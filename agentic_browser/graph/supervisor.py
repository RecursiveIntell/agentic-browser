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
    
    SYSTEM_PROMPT = """You are a SUPERVISOR that routes tasks to specialized agents and synthesizes results.

AVAILABLE AGENTS:
- os: Find files, list directories, read/write local files, run shell commands
- code: Analyze code projects, understand what apps do, read source files
- research: Search the internet using DuckDuckGo, visit websites, gather info
- browser: Navigate to specific URLs, interact with web pages

ROUTING STRATEGY:
Analyze the goal and route to the most appropriate agent:
- Local file operations → os
- Code analysis, "what does X do?" → code
- "Find similar", "research", "search online" → research
- Navigate to specific URL → browser

MULTI-STEP TASKS:
Some goals require multiple agents:
- "Analyze X and find similar online" → code first, then research
- "Find X in files and summarize" → os first, then code

INTELLIGENT COMPLETION:
- Check extracted_data to see what's been gathered
- If agents are cycling (same agent 3+ times without new data), move on
- If you have enough information, synthesize a report and complete
- A good partial answer is better than infinite loops

Respond with JSON:
{
  "route_to": "os|code|research|browser|done",
  "rationale": "why this agent or why completing"
}

When completing, synthesize ALL gathered data into a useful report:
{
  "route_to": "done",
  "final_answer": "## Report Title\\n\\n[Synthesized findings from all agents]"
}"""

    def __init__(self, config: AgentConfig):
        """Initialize supervisor.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        
        # Build LLM kwargs - some models (o1, o3) don't support temperature
        llm_kwargs = {
            "base_url": config.model_endpoint,
            "api_key": config.api_key or "not-required",
            "model": config.model,
            "max_tokens": 500,
        }
        
        # Only add temperature for models that support it
        model_lower = (config.model or "").lower()
        is_reasoning_model = any(x in model_lower for x in ["o1", "o3", "o4"])
        if not is_reasoning_model:
            llm_kwargs["temperature"] = 0.1
            
        self.llm = ChatOpenAI(**llm_kwargs)
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
            initial_domain = self._refine_initial_domain(decision.domain, state["goal"])
            
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
            
            # DEBUG: Print Supervisor decision
            print(f"\n{'='*60}")
            print(f"[DEBUG] Supervisor - Raw LLM Response:")
            print(f"{'='*60}")
            print(response.content[:1000] if len(response.content) > 1000 else response.content)
            print(f"{'='*60}\n")
            
            decision = self._parse_decision(response.content)
            
            # DEBUG: Print parsed decision
            print(f"[DEBUG] Supervisor - Parsed Decision: {decision}")
            print(f"[DEBUG] Supervisor - Current step: {state['step_count']}, extracted_data keys: {list(state['extracted_data'].keys())}")
            
            # HARD BLOCK: Minimum steps before allowing completion
            MIN_STEPS_BEFORE_DONE = 5
            
            if decision.get("route_to") == "done":
                if state["step_count"] < MIN_STEPS_BEFORE_DONE:
                    print(f"[DEBUG] Supervisor - BLOCKED 'done': step_count ({state['step_count']}) < MIN_STEPS ({MIN_STEPS_BEFORE_DONE})")
                    # Force continue with code agent to gather more data
                    decision = {"route_to": "code", "rationale": "Need more exploration before completing"}
                elif not state['extracted_data'] or len(state['extracted_data']) == 0:
                    print(f"[DEBUG] Supervisor - BLOCKED 'done': No extracted_data yet")
                    decision = {"route_to": "code", "rationale": "No data collected yet"}
            
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
                "current_domain": decision.get("route_to", state["current_domain"]),
                "active_agent": "supervisor",
                "step_count": state["step_count"] + 1,
                "messages": [AIMessage(content=response.content)],
            }
            
        except Exception as e:
            print(f"[DEBUG] Supervisor - ERROR: {e}")
            # On error, try to continue with current domain
            return {
                **state,
                "error": f"Supervisor error: {str(e)}",
                "step_count": state["step_count"] + 1,
            }
    
    def _refine_initial_domain(self, domain: str, goal: str) -> str:
        """Refine the rough domain decision into a specific agent."""
        goal_lower = goal.lower()
        
        if domain == "browser":
            # Check for research intent
            research_keywords = ["research", "find out", "look up", "synthesize", "compare", "what is", "who is", "search"]
            if any(k in goal_lower for k in research_keywords):
                return "research"
            return "browser"
            
        elif domain == "os":
            # Check for code intent
            code_keywords = ["analyze", "debug", "fix", "code", "repo", "project", "explain", "how does"]
            if any(k in goal_lower for k in code_keywords):
                return "code"
            return "os"
            
        elif domain == "both":
            # Disambiguate based on primary intent
            if "research" in goal_lower or "find" in goal_lower:
                return "research"
            if "analyze" in goal_lower or "code" in goal_lower:
                return "code"
            # Default to research for mixed queries as it's safer
            return "research"
            
        return "browser"  # Fallback
    
    def _build_messages(self, state: AgentState) -> list:
        """Build messages for supervisor decision."""
        # Format extracted data
        data = state['extracted_data']
        important_keys = [k for k in data.keys() if any(x in k for x in ['analysis', 'findings', 'summary', 'report'])]
        other_keys = [k for k in data.keys() if k not in important_keys]
        
        formatted_data = []
        for k in important_keys:
            formatted_data.append(f"--- {k} ---\n{str(data[k])[:2000]}")
            
        remaining_chars = 1000
        if other_keys:
            other_data = {k: data[k] for k in other_keys}
            formatted_data.append(f"--- Other Data ---\n{json.dumps(other_data, indent=2)[:remaining_chars]}")
            
        data_str = "\n\n".join(formatted_data)

        # Summarize current state
        context = f"""
User's Goal: {state['goal']}
Current Step: {state['step_count']} / {state['max_steps']}
Current Domain: {state['current_domain']}

Data Collected:
{data_str}

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
            if not content:
                print("[DEBUG] Supervisor - Empty response, continuing with code agent")
                return {"route_to": "code", "rationale": "Empty LLM response, continuing exploration"}
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            return json.loads(content)
        except json.JSONDecodeError:
            # On parse failure, continue exploration instead of quitting
            print("[DEBUG] Supervisor - JSON parse failed, continuing with code agent")
            return {"route_to": "code", "rationale": "Failed to parse response"}
    
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
