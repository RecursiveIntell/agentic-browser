"""
Supervisor agent that orchestrates specialized agents.

Routes tasks to appropriate agents and synthesizes results.
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import AgentState
from ..domain_router import DomainRouter
from ..config import AgentConfig


class Supervisor:
    """Orchestrator that routes tasks to specialized agents.
    
    The supervisor analyzes the user's goal and delegates to the
    appropriate specialized agent (browser, os, research, code).
    """
    
    SYSTEM_PROMPT = """You are a SUPERVISOR agent that orchestrates specialized workers.

Your job is to:
1. Analyze the user's COMPLETE goal
2. Break it down into steps if needed
3. Route to agents IN ORDER until ALL parts are done
4. Only mark done when the ENTIRE goal is complete

Available agents:
- browser: Web navigation, clicking, form filling, simple lookups
- os: Local filesystem operations, shell commands, file reading/writing
- research: Multi-source web research, comparing information, synthesis
- code: Code analysis, project understanding, running tests

MULTI-STEP TASK EXAMPLES:
- "research similar to X app in my folder" = code (analyze app) → research (find similar online)
- "find X in my files and search for alternatives" = os (find) → research (alternatives)
- "analyze project and look up best practices" = code (analyze) → research (best practices)

CRITICAL: Look at the FULL goal, not just what's been done. If goal says "research on internet" you MUST route to research agent!

Respond with JSON:
{
  "route_to": "browser|os|research|code|done",
  "rationale": "reason - what part of goal this addresses",
  "remaining_steps": "what still needs to be done after this"
}

ONLY mark done when ALL parts of the goal are complete:
{
  "route_to": "done",
  "final_answer": "comprehensive response covering all parts of the goal"
}"""

    def __init__(self, config: AgentConfig):
        """Initialize supervisor.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.llm = ChatOpenAI(
            base_url=config.model_endpoint,
            api_key=config.api_key or "not-required",
            model=config.model,
            temperature=0.1,
            max_tokens=500,
        )
        self.domain_router = DomainRouter(config)
    
    def route(self, state: AgentState) -> AgentState:
        """Decide which agent should handle the current state.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with routing decision
        """
        # Check if we're done
        if state.get("task_complete"):
            return {
                **state,
                "current_domain": "done",
                "active_agent": "supervisor",
            }
        
        # Check step limit
        if state["step_count"] >= state["max_steps"]:
            return {
                **state,
                "current_domain": "done",
                "task_complete": True,
                "final_answer": self._force_summary(state),
            }
        
        # First routing: use heuristic domain router
        if state["step_count"] == 0:
            decision = self.domain_router.route(state["goal"])
            initial_domain = self._map_domain(decision.domain)
            return {
                **state,
                "current_domain": initial_domain,
                "active_agent": "supervisor",
                "step_count": state["step_count"] + 1,
            }
        
        # Subsequent routing: check if worker finished
        messages = self._build_messages(state)
        
        try:
            response = self.llm.invoke(messages)
            decision = self._parse_decision(response.content)
            
            if decision.get("route_to") == "done":
                return {
                    **state,
                    "current_domain": "done",
                    "task_complete": True,
                    "final_answer": decision.get("final_answer", "Task completed"),
                    "messages": [AIMessage(content=response.content)],
                }
            
            return {
                **state,
                "current_domain": decision.get("route_to", "browser"),
                "active_agent": "supervisor",
                "step_count": state["step_count"] + 1,
                "messages": [AIMessage(content=response.content)],
            }
            
        except Exception as e:
            # On error, try to continue with current domain
            return {
                **state,
                "error": f"Supervisor error: {str(e)}",
                "step_count": state["step_count"] + 1,
            }
    
    def _build_messages(self, state: AgentState) -> list:
        """Build messages for supervisor decision."""
        # Summarize current state
        context = f"""
User's Goal: {state['goal']}
Current Step: {state['step_count']} / {state['max_steps']}
Current Domain: {state['current_domain']}

Data Collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

URLs Visited: {len(state['visited_urls'])}
Files Accessed: {len(state['files_accessed'])}

Last Error: {state.get('error', 'None')}

Recent Messages:
{self._format_recent_messages(state)}

Should the task continue? If so, which agent should handle it next?
If the worker agent completed with a final answer, synthesize and return done.
"""
        
        return [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    
    def _format_recent_messages(self, state: AgentState) -> str:
        """Format recent messages for context."""
        messages = state.get("messages", [])[-3:]
        formatted = []
        for msg in messages:
            content = msg.content[:200] if hasattr(msg, 'content') else str(msg)[:200]
            formatted.append(f"- {type(msg).__name__}: {content}")
        return "\n".join(formatted) or "(no messages)"
    
    def _parse_decision(self, response: str) -> dict:
        """Parse supervisor decision from response."""
        try:
            content = response.strip()
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            return json.loads(content)
        except json.JSONDecodeError:
            return {"route_to": "done", "final_answer": "Task completed"}
    
    def _map_domain(self, domain: str) -> str:
        """Map DomainRouter domain to agent names."""
        mapping = {
            "browser": "browser",
            "os": "os",
            "research": "research",
            "code": "code",
        }
        return mapping.get(domain, "browser")
    
    def _force_summary(self, state: AgentState) -> str:
        """Generate a summary when step limit is reached."""
        data = state.get("extracted_data", {})
        if data:
            return f"Reached step limit. Collected data: {json.dumps(data)[:500]}"
        return "Reached maximum step limit without completing the task."


def supervisor_node(state: AgentState) -> AgentState:
    """LangGraph node function for supervisor."""
    from .tool_registry import get_tools
    
    # Get config from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        # Fallback: create default config
        agent_config = AgentConfig()
    
    supervisor = Supervisor(agent_config)
    return supervisor.route(state)


def route_to_agent(state: AgentState) -> Literal["browser", "os", "research", "code", "__end__"]:
    """Conditional edge function for routing.
    
    Returns the name of the next node based on current_domain.
    """
    domain = state.get("current_domain", "")
    
    if domain == "done" or state.get("task_complete"):
        return "__end__"
    
    if domain in ("browser", "os", "research", "code"):
        return domain
    
    # Default to browser
    return "browser"
