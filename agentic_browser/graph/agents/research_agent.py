"""
Research Agent for multi-source information gathering.

Coordinates browser actions to research topics from multiple sources.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class ResearchAgentNode(BaseAgent):
    """Specialized agent for research tasks.
    
    Coordinates browser operations to gather information from
    multiple sources and synthesize findings.
    """
    
    AGENT_NAME = "research"
    MAX_STEPS_PER_INVOCATION = 8
    
    SYSTEM_PROMPT = """You are a specialized RESEARCH agent. Your job is to find and synthesize information from multiple sources.

WORKFLOW:
1. Search using DuckDuckGo (goto "https://duckduckgo.com")
2. Visit 2-3 relevant sources (not just search snippets!)
3. Extract key information from each
4. Synthesize findings into a comprehensive answer

Available actions:
- goto: { "url": "https://..." }
- extract_visible_text: { "max_chars": 5000 }
- done: { "summary": "synthesized findings from all sources" }

CRITICAL RULES:
1. ALWAYS use DuckDuckGo - Google blocks AI agents!
2. Visit ACTUAL websites, not just read search snippets
3. Only use complete URLs starting with https://
4. After visiting 2-3 sources, call "done" with synthesis
5. If a site times out or 404s, skip it and try another

Respond with JSON:
{
  "action": "goto|extract_visible_text|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config, browser_tools=None):
        """Initialize research agent.
        
        Args:
            config: Agent configuration
            browser_tools: Browser tools for web access
        """
        super().__init__(config)
        self._browser_tools = browser_tools
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_browser_tools(self, browser_tools) -> None:
        """Set browser tools after initialization."""
        self._browser_tools = browser_tools
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute research workflow.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with research findings
        """
        if not self._browser_tools:
            return self._update_state(
                state,
                error="Browser tools not available for research",
            )
        
        # Get current page state
        try:
            page_state = self._browser_tools.get_page_state()
        except Exception:
            page_state = {}
        
        # Build context showing progress
        sources_visited = len([u for u in state['visited_urls'] if u and 'duckduckgo' not in u])
        
        task_context = f"""
RESEARCH TASK: {state['goal']}

Sources visited so far: {sources_visited}/3
Visited URLs: {chr(10).join(state['visited_urls'][-5:]) or '(none)'}

Current page: {page_state.get('title', 'Unknown')}
URL: {page_state.get('url', 'about:blank')}

Visible content (truncated):
{page_state.get('visible_text', '')[:2000]}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

REMINDER: Visit 2-3 actual sources, then synthesize with "done".
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.llm.invoke(messages)
            action_data = self._parse_action(response.content)
            
            if action_data.get("action") == "done":
                return self._update_state(
                    state,
                    message=AIMessage(content=response.content),
                    task_complete=True,
                    final_answer=action_data.get("args", {}).get("summary", "Research completed"),
                )
            
            # Execute browser action
            result = self._browser_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            visited = None
            if action_data.get("action") == "goto":
                visited = action_data.get("args", {}).get("url")
            
            extracted = None
            if action_data.get("action") == "extract_visible_text" and result.success:
                key = f"research_source_{sources_visited + 1}"
                extracted = {key: result.data[:1000] if result.data else result.message[:500]}
            
            return self._update_state(
                state,
                message=AIMessage(content=response.content),
                visited_url=visited,
                extracted_data=extracted,
                error=result.message if not result.success else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Research agent error: {str(e)}",
            )
    
    def _parse_action(self, response: str) -> dict:
        """Parse LLM response into action dict."""
        try:
            content = response.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"action": "done", "args": {"summary": "Failed to parse action"}}


def research_agent_node(state: AgentState, config: dict) -> AgentState:
    """LangGraph node function for research agent."""
    agent_config = config.get("agent_config")
    browser_tools = config.get("browser_tools")
    
    agent = ResearchAgentNode(agent_config, browser_tools)
    return agent.execute(state)
