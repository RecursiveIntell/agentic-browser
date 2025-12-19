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
- click: { "selector": "text=Link Text" }  # USE text= PREFIX FOR LINK TEXT!
- type: { "selector": "input[name=q]", "text": "search query" }
- press: { "key": "Enter|Tab|..." }
- scroll: { "amount": 800 }
- extract_visible_text: { "max_chars": 8000 }
- download_image: { }  # Downloads the largest image on the page to ~/Downloads
- download_image: { "selector": "img.main" }  # Download specific image by selector
- download_image: { "url": "https://example.com/image.jpg" }  # Download by direct URL
- done: { "summary": "what you accomplished" }

=== CLICK SELECTOR FORMAT ===
ALWAYS use the correct prefix for click selectors:
- text=Link Text → Click link by visible text (MOST COMMON)
- #elementId → Click by ID
- .className → Click by class
- button:has-text("Submit") → Click button with text

EXAMPLES:
✅ CORRECT: {"action": "click", "args": {"selector": "text=Read More"}}
✅ CORRECT: {"action": "click", "args": {"selector": "text=r/artificial"}}
❌ WRONG: {"action": "click", "args": {"selector": "r/artificial"}}
❌ WRONG: {"action": "click", "args": {"selector": "Read More"}}

=== IMAGE DOWNLOAD ===
To download an image:
1. Navigate to a page with images (e.g., Pixabay, Pexels, Unsplash)
2. Optionally navigate to the image detail page for higher resolution
3. Call download_image with no args to auto-download the largest image

CRITICAL RULES:
1. ALWAYS use DuckDuckGo for search (https://duckduckgo.com) - Google blocks AI agents!
2. Only use COMPLETE URLs starting with https://
3. If you get a 404, move to the next site - don't retry
4. After 3-5 actions, call "done" with what you found
5. For clicking links, ALWAYS use text= prefix!

Respond with JSON:
{
  "action": "goto|click|type|press|scroll|extract_visible_text|download_image|done",
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
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            if action_data.get("action") == "done":
                # Store findings but DON'T mark task_complete - let supervisor decide
                summary = action_data.get("args", {}).get("summary", "Task completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"browser_findings": summary},
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
                
                content_str = ""
                if isinstance(result.data, dict):
                    content_str = result.data.get("text", str(result.data))
                else:
                    content_str = str(result.data) if result.data else str(result.message)
                    
                extracted = {key: content_str[:2000]}
                
            # Create tool output message so agent knows result
            tool_content = str(result.message) if result.message else "Action successful"
            if result.data and not extracted:
                tool_content = str(result.data)[:1000]
            
            tool_msg = HumanMessage(content=f"Tool output: {tool_content}")
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
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
            
            data = json.loads(content)
            
            # Validate action
            if not data.get("action"):
                # Default to text extraction instead of done
                return {
                    "action": "extract_visible_text",
                    "args": {"max_chars": 5000}
                }
                
            return data
        except json.JSONDecodeError:
            # On parse failure, extract instead of quitting
            return {"action": "extract_visible_text", "args": {"max_chars": 5000}}
    
    def _execute_action(self, action_data: dict) -> ToolResult:
        """Execute a browser action with auto-fix for common selector issues."""
        action = action_data.get("action", "")
        args = action_data.get("args", {}).copy()  # Copy to avoid mutating original
        
        # Auto-fix click selectors that are missing prefix
        if action == "click" and "selector" in args:
            selector = args["selector"]
            # Check if selector looks like plain text (not CSS/xpath/text=)
            # Common patterns that need text= prefix:
            # - Contains spaces and no special CSS chars
            # - Contains / without being xpath
            # - Doesn't start with common prefixes
            needs_prefix = (
                not selector.startswith(("text=", "xpath=", "#", ".", "[", "button", "a[", "input"))
                and not selector.startswith("//")  # xpath
                and ("/" in selector or " " in selector or selector[0].islower())
            )
            
            if needs_prefix:
                args["selector"] = f"text={selector}"
        
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


