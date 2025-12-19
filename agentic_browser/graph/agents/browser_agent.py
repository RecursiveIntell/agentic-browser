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
        import logging
        logger = logging.getLogger(__name__)
        
        if not self._browser_tools:
            return self._update_state(
                state,
                error="Browser tools not initialized",
            )
        
        # Build context from current page state
        page_state = self._get_page_state()
        current_url = page_state.get('current_url', '') or page_state.get('url', 'about:blank')
        
        # Get step count - NOTE: state uses 'step_count' not 'current_step'
        step_count = state.get('step_count', 0)
        
        # Detect if we're on an images page
        url_lower = current_url.lower()
        is_images_page = any(x in url_lower for x in [
            '/images', 'images.', 'pixabay', 'pexels', 'unsplash', 
            'ia=images', 'iax=images', 'tbm=isch',  # DuckDuckGo and Google image params
            'image', 'photo', 'pic', 'jpg', 'png'
        ])
        
        # Check if goal involves downloading images
        goal_lower = state['goal'].lower()
        is_image_download_goal = any(w in goal_lower for w in [
            'download', 'save', 'get image', 'get picture', 'find picture', 
            'find image', 'picture of', 'image of', 'photo of'
        ])
        
        # Debug logging for auto-download conditions
        print(f"[BROWSER DEBUG] URL: {current_url[:80]}...")
        print(f"[BROWSER DEBUG] step_count={step_count}, is_images_page={is_images_page}, is_image_download_goal={is_image_download_goal}")
        
        # === SMART AUTO-DOWNLOAD ===
        # If we're on an images page, goal is to download an image, and we've been trying for a while
        # Just automatically download the image instead of asking the LLM (which keeps ignoring it)
        if is_images_page and is_image_download_goal and step_count >= 6:
            print(f"[BROWSER] 🔄 AUTO-DOWNLOAD TRIGGERED! step={step_count}, executing download_image")
            logger.info(f"Auto-download: images page detected, goal involves images, step {step_count}")
            
            result = self._browser_tools.execute("download_image", {})
            
            if result.success:
                download_path = result.data.get("path", "") if result.data else ""
                print(f"[BROWSER] ✅ Auto-downloaded image to: {download_path}")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=f"Auto-downloaded image to: {download_path}")],
                    extracted_data={
                        "downloaded_image": download_path,
                        "image_filename": result.data.get("filename", "") if result.data else "",
                    },
                    # Mark as complete since we downloaded the image
                    task_complete=True,
                    final_answer=f"Downloaded image to: {download_path}",
                )
            else:
                # Download failed, log and let agent try other approaches
                print(f"[BROWSER] ⚠️ Auto-download failed: {result.message}")
        
        image_download_hint = ""
        if is_images_page:
            image_download_hint = """
⚠️ YOU ARE ON AN IMAGES PAGE! If the goal is to download an image:
- Call download_image with NO arguments to auto-download the largest image
- Example: {"action": "download_image", "args": {}}
"""
        
        task_context = f"""
Current URL: {current_url}
Page Title: {page_state.get('page_title', '') or page_state.get('title', '')}
{image_download_hint}
Visible Text (truncated):
{page_state.get('visible_text', '')[:2000]}

Top Links:
{self._format_links(page_state.get('top_links', []) or page_state.get('links', []))}

Already Visited:
{chr(10).join(f'- {url}' for url in state['visited_urls'][-5:])}

Your task: {state['goal']}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            # Log the action for debugging
            action = action_data.get("action", "unknown")
            args = action_data.get("args", {})
            logger.debug(f"Browser agent action: {action} with args: {args}")
            print(f"[BROWSER] Action: {action}, Args: {args}")  # Visible debug output
            
            # === LOOP DETECTION ===
            # Check if we're repeating the same action or stuck in a goto loop
            action_key = f"{action}:{json.dumps(args, sort_keys=True)}"
            recent_actions = state.get("_recent_actions", [])
            
            # Count how many times this exact action was done recently
            repeat_count = sum(1 for a in recent_actions[-5:] if a == action_key)
            
            # Detect: repeating goto to URL we're already on
            if action == "goto" and args.get("url"):
                goto_url = args.get("url", "").split("?")[0].rstrip("/")
                current_base = current_url.split("?")[0].rstrip("/")
                if goto_url == current_base or repeat_count >= 1:
                    logger.warning(f"Loop detected: already on {current_url}, forcing download_image")
                    print(f"[BROWSER] ⚠️ Loop detected - already on this URL or repeated action!")
                    
                    # If it's an images page and goal involves downloading, force download_image
                    goal_lower = state['goal'].lower()
                    if is_images_page and any(w in goal_lower for w in ['download', 'save', 'get image', 'get picture', 'find picture', 'find image']):
                        print(f"[BROWSER] 🔄 Auto-executing download_image to break loop")
                        action_data = {"action": "download_image", "args": {}}
                        action = "download_image"
                        args = {}
            
            # If repeated 3+ times, force a scroll or download
            if repeat_count >= 2:
                print(f"[BROWSER] ⚠️ Action repeated {repeat_count + 1}x - breaking loop")
                if is_images_page:
                    action_data = {"action": "download_image", "args": {}}
                    action = "download_image"
                    args = {}
                else:
                    action_data = {"action": "scroll", "args": {"amount": 500}}
                    action = "scroll"
                    args = {"amount": 500}
            
            # Track this action for loop detection (store in state)
            recent_actions.append(action_key)
            if len(recent_actions) > 10:
                recent_actions = recent_actions[-10:]
            
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
            
            # Handle download_image results - store download path
            if action_data.get("action") == "download_image" and result.success:
                download_path = result.data.get("path", "") if result.data else ""
                filename = result.data.get("filename", "") if result.data else ""
                extracted = {
                    "downloaded_image": download_path,
                    "image_filename": filename,
                }
                # Log success
                print(f"[BROWSER] Downloaded image to: {download_path}")
                
            # Create tool output message so agent knows result
            tool_content = str(result.message) if result.message else "Action successful"
            if result.data and not extracted:
                tool_content = str(result.data)[:1000]
            
            tool_msg = HumanMessage(content=f"Tool output: {tool_content}")
            
            # Build updated state with loop tracking
            new_state = self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                visited_url=visited,
                extracted_data=extracted,
                error=result.message if not result.success else None,
            )
            
            # Persist recent actions for loop detection
            new_state["_recent_actions"] = recent_actions
            
            return new_state
            
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


