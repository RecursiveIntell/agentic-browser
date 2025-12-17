"""
Browser Agent for web navigation and data extraction.

Wraps existing BrowserTools for LangGraph integration.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState
from ...tools import BrowserTools, ToolResult
from ...llm_client import LLMClient


class BrowserAgentNode(BaseAgent):
    """Specialized agent for web browsing tasks.
    
    Uses Playwright-based BrowserTools for navigation, clicking,
    typing, and data extraction.
    """
    
    AGENT_NAME = "browser"
    MAX_STEPS_PER_INVOCATION = 5
    
    SYSTEM_PROMPT = """You are a specialized BROWSER agent. Your ONLY job is web navigation.

You have access to these browser actions:
- goto: { "url": "https://..." }
- click: { "selector": "css or text selector" }
- type: { "selector": "...", "text": "..." }
- press: { "key": "Enter|Tab|..." }
- scroll: { "amount": 800 }
- extract_visible_text: { "max_chars": 8000 }
- done: { "summary": "what you accomplished" }

CRITICAL RULES:
1. ALWAYS use DuckDuckGo for search (https://duckduckgo.com) - Google blocks AI agents!
2. Only use COMPLETE URLs starting with https://
3. If you get a 404, move to the next site - don't retry
4. After 3-5 actions, call "done" with what you found

Respond with JSON:
{
  "action": "goto|click|type|press|scroll|extract_visible_text|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config, browser_tools: BrowserTools | None = None):
        """Initialize browser agent.
        
        Args:
            config: Agent configuration
            browser_tools: Optional pre-initialized BrowserTools
        """
        super().__init__(config)
        self._browser_tools = browser_tools
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_browser_tools(self, browser_tools: BrowserTools) -> None:
        """Set browser tools after initialization."""
        self._browser_tools = browser_tools
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute browser navigation steps.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with browser action results
        """
        if not self._browser_tools:
            return self._update_state(
                state,
                error="Browser tools not initialized",
            )
        
        # Build context from current page state
        page_state = self._get_page_state()
        task_context = f"""
Current URL: {page_state.get('url', 'about:blank')}
Page Title: {page_state.get('title', '')}

Visible Text (truncated):
{page_state.get('visible_text', '')[:2000]}

Top Links:
{self._format_links(page_state.get('links', []))}

Already Visited:
{chr(10).join(f'- {url}' for url in state['visited_urls'][-5:])}

Your task: {state['goal']}
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
                    final_answer=action_data.get("args", {}).get("summary", "Task completed"),
                )
            
            # Execute the browser action
            result = self._execute_action(action_data)
            
            # Track visited URL
            visited = None
            if action_data.get("action") == "goto":
                visited = action_data.get("args", {}).get("url")
            
            # Store extraction results
            extracted = None
            if action_data.get("action") == "extract_visible_text" and result.success:
                key = f"browser_extract_{len(state['extracted_data'])}"
                extracted = {key: result.data or result.message[:500]}
            
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
                error=f"Browser agent error: {str(e)}",
            )
    
    def _get_page_state(self) -> dict[str, Any]:
        """Get current browser page state."""
        if not self._browser_tools:
            return {}
        
        try:
            return self._browser_tools.get_page_state()
        except Exception:
            return {}
    
    def _format_links(self, links: list[dict]) -> str:
        """Format links for context."""
        if not links:
            return "(no links)"
        
        formatted = []
        for link in links[:10]:
            text = link.get("text", "")[:40]
            href = link.get("href", "")
            formatted.append(f"- [{text}]({href})")
        return "\n".join(formatted)
    
    def _parse_action(self, response: str) -> dict:
        """Parse LLM response into action dict."""
        try:
            # Extract JSON from response
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
    
    def _execute_action(self, action_data: dict) -> ToolResult:
        """Execute a browser action."""
        action = action_data.get("action", "")
        args = action_data.get("args", {})
        
        return self._browser_tools.execute(action, args)


def browser_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for browser agent."""
    from ..tool_registry import get_tools
    
    # Get tools from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
        browser_tools = tools.browser_tools
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig()
        browser_tools = None
    
    agent = BrowserAgentNode(agent_config, browser_tools)
    return agent.execute(state)


