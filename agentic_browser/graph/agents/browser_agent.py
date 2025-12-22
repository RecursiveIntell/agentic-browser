"""
Browser Agent for web navigation and data extraction.

Wraps existing BrowserTools for LangGraph integration.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ...tools import BrowserTools, ToolResult
from ..state import AgentState
from .base import BaseAgent


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

=== LOGIN & AUTHENTICATION ===
If you encounter a login page or wall:
1. Don't panic or stop. Many sites (Google, Social Media) require logins.
2. The user's profile often has active sessions. Try interacting with the page first.
3. If not logged in, you can use 'type' and 'click' to handle login forms.
4. If you are stuck or need the user's manual intervention, call 'done' with a summary like "Halted at login wall: needs user credentials".
5. NEVER refuse to visit a site just because it has a "privacy/security" notification or requires login.

CRITICAL RULES:
1. ALWAYS use DuckDuckGo for search (https://duckduckgo.com) - Google blocks AI agents!
2. Only use COMPLETE URLs starting with https://
3. If you get a 404, move to the next site - don't retry
4. After 3-5 actions, call "done" with what you accomplished
5. For clicking links, ALWAYS use text= prefix!
6. SKIP SPONSORED/AD LINKS! Never click links containing "Sponsored", "Ad", "Advertisement", or marked as ads

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

        # === AUTO-NAVIGATION FIX ===
        # If we're on about:blank or empty page, FORCE navigation to DuckDuckGo
        # This prevents agents from extracting empty pages and getting stuck
        if current_url == 'about:blank' or not current_url or current_url.startswith('about:'):
            # Build search query from goal
            goal = state['goal']
            search_query = goal.replace(' ', '+')[:100]
            search_url = f"https://duckduckgo.com/?q={search_query}"
            
            print(f"[BROWSER] 🚀 AUTO-NAVIGATING: Browser on blank page, opening DuckDuckGo")
            self._browser_tools.execute("goto", {"url": search_url})
            
            return self._update_state(
                state,
                messages=[AIMessage(content=f"Auto-navigating to search: {search_url}")],
                visited_url=search_url,
            )

        # Get step count - NOTE: state uses 'step_count' not 'current_step'
        step_count = state.get('step_count', 0)

        # Detect if we're on an images page
        url_lower = current_url.lower()

        # Detect blocked/CAPTCHA pages (Google, etc.) - these are NOT image pages!
        is_blocked_page = any(x in url_lower for x in [
            '/sorry/', '/captcha', 'unusual_traffic', 'recaptcha',
            'challenge-platform', 'blocked'
        ])

        # Only flag as images page if NOT blocked and has specific image site patterns
        is_images_page = not is_blocked_page and any(x in url_lower for x in [
            '/images', 'images.', 'pixabay.com', 'pexels.com', 'unsplash.com',
            'ia=images', 'iax=images', 'tbm=isch',  # DuckDuckGo and Google image params
        ])


        # Check if goal involves downloading images
        goal_lower = state['goal'].lower()
        is_image_download_goal = any(w in goal_lower for w in [
            'download', 'save', 'get image', 'get picture', 'find picture',
            'find image', 'picture of', 'image of', 'photo of'
        ])
        
        # === MULTI-IMAGE DETECTION ===
        # Parse how many images the user wants (e.g., "download 2 pictures")
        import re
        required_images = 1  # Default to 1
        image_count_patterns = [
            r'(\d+)\s*(?:pictures?|images?|photos?)',  # "2 pictures", "3 images"
            r'(?:download|save|get)\s*(\d+)',  # "download 2", "save 3"
        ]
        for pattern in image_count_patterns:
            match = re.search(pattern, goal_lower)
            if match:
                required_images = int(match.group(1))
                required_images = max(1, min(required_images, 10))  # Cap at 10
                break
        
        # Track how many images we've already downloaded
        downloaded_count = state.get('browser_downloaded_image_count', 0)

        # Debug logging for auto-download conditions
        logger.debug(f"URL: {current_url[:80]}...")
        logger.debug(f"step_count={step_count}, is_images_page={is_images_page}, is_image_download_goal={is_image_download_goal}")

        # === HANDLE BLOCKED PAGES ===
        # If we're on a CAPTCHA/blocked page, redirect to DuckDuckGo Images
        if is_blocked_page and is_image_download_goal:
            logger.warning("Blocked page detected! Redirecting to DuckDuckGo Images")
            # Extract search terms from goal
            search_terms = state['goal'].replace('find', '').replace('picture', '').replace('image', '')
            search_terms = search_terms.replace('save', '').replace('download', '').strip()[:50]
            search_query = search_terms.replace(' ', '+') or 'rare+cat'

            redirect_url = f"https://duckduckgo.com/?q={search_query}&iax=images&ia=images"
            self._browser_tools.execute("goto", {"url": redirect_url})

            return self._update_state(
                state,
                messages=[AIMessage(content="Blocked by search engine, redirecting to DuckDuckGo Images")],
                error=None,  # Clear any previous error
            )

        # Track download failures for loop breaking
        failed_downloads = state.get('browser_failed_download_count', 0)

        # === SMART AUTO-DOWNLOAD ===
        # If we're on an images page, goal is to download an image, and we've been trying for a while
        # Just automatically download the image instead of asking the LLM (which keeps ignoring it)
        if is_images_page and is_image_download_goal and step_count >= 6:
            # If we've already failed 2+ times, navigate to a reliable image site
            if failed_downloads >= 2:
                logger.info(f"download_image failed {failed_downloads}x, navigating to Pixabay")
                search_query = state['goal'].replace(' ', '%20')[:40]
                pixabay_url = f"https://pixabay.com/images/search/{search_query}/"
                self._browser_tools.execute("goto", {"url": pixabay_url})

                new_state = self._update_state(
                    state,
                    messages=[AIMessage(content=f"Download failed {failed_downloads}x, navigating to Pixabay")],
                )
                new_state['browser_failed_download_count'] = 0  # Reset counter
                return new_state

            logger.info(f"AUTO-DOWNLOAD TRIGGERED! step={step_count}, executing download_image")
            logger.info(f"Auto-download: images page detected, goal involves images, step {step_count}")
            logger.info(f"Progress: {downloaded_count}/{required_images} images downloaded")

            result = self._browser_tools.execute("download_image", {})

            if result.success:
                download_path = result.data.get("path", "") if result.data else ""
                downloaded_count += 1
                logger.info(f"Auto-downloaded image {downloaded_count}/{required_images} to: {download_path}")
                
                # Check if we've downloaded enough images
                is_complete = downloaded_count >= required_images
                
                # If NOT complete, SCROLL to get different images into view
                # This prevents downloading the same image repeatedly
                if not is_complete:
                    logger.info(f"Need {required_images - downloaded_count} more images - scrolling to find new images")
                    scroll_result = self._browser_tools.execute("scroll", {"amount": 800})
                    logger.info(f"Scrolled page to reveal new images: {scroll_result.success}")
                
                new_state = self._update_state(
                    state,
                    messages=[AIMessage(content=f"Auto-downloaded image {downloaded_count}/{required_images} to: {download_path}")],
                    extracted_data={
                        f"downloaded_image_{downloaded_count}": download_path,
                        f"image_filename_{downloaded_count}": result.data.get("filename", "") if result.data else "",
                    },
                    task_complete=is_complete,
                    final_answer=f"Downloaded {downloaded_count}/{required_images} images" if is_complete else None,
                )
                new_state['browser_downloaded_image_count'] = downloaded_count
                
                return new_state
            else:
                # Download failed - increment counter and continue
                failed_downloads += 1
                logger.warning(f"Auto-download failed ({failed_downloads}x): {result.message}")
                # Store failure count for next iteration
                new_state = self._update_state(state, error=result.message)
                new_state['browser_failed_download_count'] = failed_downloads
                new_state['browser_downloaded_image_count'] = downloaded_count
                return new_state


        image_download_hint = ""
        if is_images_page:
            image_download_hint = """
⚠️ YOU ARE ON AN IMAGES PAGE! If the goal is to download an image:
- Call download_image with NO arguments to auto-download the largest image
- Example: {"action": "download_image", "args": {}}
"""

        # === ROOT CAUSE FIX: Show LLM what it already tried ===
        # Without this, LLM doesn't know it's repeating actions
        recent_actions = state.get("browser_recent_actions", [])
        if recent_actions:
            # Parse action keys back to readable format
            action_history = []
            for action_key in recent_actions[-5:]:  # Last 5 actions
                try:
                    action_name, args_json = action_key.split(":", 1)
                    action_history.append(f"  - {action_name}: {args_json[:60]}")
                except ValueError:
                    action_history.append(f"  - {action_key[:60]}")
            actions_str = "\n".join(action_history)
            action_hint = f"""
⚠️ PREVIOUS ACTIONS ON THIS SESSION (DO NOT REPEAT!):
{actions_str}

You MUST choose a DIFFERENT action. If previous actions didn't work:
- Try a different selector with text= prefix
- Navigate to a different URL
- Use 'back' to return to search results
- Use 'done' if you've gathered enough information
"""
        else:
            action_hint = ""

        task_context = f"""
Current URL: {current_url}
Page Title: {page_state.get('page_title', '') or page_state.get('title', '')}
{image_download_hint}
{action_hint}
Visible Text (truncated):
{page_state.get('visible_text', '')[:800]}

Top Links:
{self._format_links(page_state.get('top_links', []) or page_state.get('links', []))}

Recently Visited (last 10):
{chr(10).join(f'- {url}' for url in list(state['visited_urls'])[-10:])}

Your task: {state['goal']}
"""
        # CRITICAL: Guard against task_context explosion
        MAX_TASK_CONTEXT = 8000
        if len(task_context) > MAX_TASK_CONTEXT:
            task_context = task_context[:MAX_TASK_CONTEXT] + "\n...[Task context truncated]"


        # Vision mode: capture screenshot for LLM
        # Now cost-effective with detail=low (~500 tokens vs ~5k with high)
        screenshot_b64 = None

        if self.config.vision_mode and self._browser_tools:
            screenshot_b64 = self.capture_screenshot_base64(self._browser_tools)
            if screenshot_b64:
                task_context += """

[VISION MODE] A screenshot of the current page is attached.
CRITICAL VISION RULES:
- ONLY click elements that are ACTUALLY VISIBLE in both the screenshot AND visible text below
- DO NOT make up button/link names - use the EXACT text you see
- If you can't find an element in the visible text, it doesn't exist
- Verify element text matches EXACTLY before clicking
"""
                print("[BROWSER] Vision mode: screenshot captured")

        # Build messages with optional vision
        # NOTE: This creates the messages list including the new HumanMessage prompt at the end
        messages = self._build_messages(state, task_context)

        # Create a copy for specific LLM invocation (to add image)
        invoke_messages = list(messages)

        # Replace last HumanMessage with vision message if we have screenshot
        if screenshot_b64 and self.config.vision_mode:
            # Pop the last message and replace with vision-enabled one for the LLM ONLY
            last_msg = invoke_messages.pop()
            if hasattr(last_msg, 'content'):
                invoke_messages.append(self.build_vision_message(last_msg.content, screenshot_b64))

        try:
            response = self.safe_invoke(invoke_messages)

            # Update token usage
            token_usage = self.update_token_usage(state, response)

            action_data = self._parse_action(response.content)

            # Log the action for debugging
            action = action_data.get("action", "unknown")
            args = action_data.get("args", {})
            logger.debug(f"Browser agent action: {action} with args: {args}")
            logger.debug(f"Action: {action}, Args: {args}")
            print(f"[BROWSER] Action: {action}, Args: {json.dumps(args)[:80]}")

            # === NAVIGATION-BASED LOOP DETECTION ===
            # Instead of comparing URLs (which fails due to params/fragments),
            # track actions and only reset after a SUCCESSFUL navigation action
            recent_actions = list(state.get("browser_recent_actions", []))
            
            action_key = f"{action}:{json.dumps(args, sort_keys=True)}"

            # Count how many times this exact action was done recently
            repeat_count = sum(1 for a in recent_actions[-5:] if a == action_key)
            
            # Also count total actions without navigation
            actions_without_nav = state.get("_actions_without_nav", 0)

            # === LOOP DETECTION AT 3X - FORCE BACK ===
            if repeat_count >= 2:
                print(f"[BROWSER] ⚠️ Action repeated {repeat_count + 1}x - forcing BACK to break loop")
                action_data = {"action": "back", "args": {}}
                action = "back"
                args = {}
                # Clear recent actions since we're forcing navigation
                recent_actions = []
            
            # Detect: repeating goto to URL we're already on
            elif action == "goto" and args.get("url"):
                goto_url = args.get("url", "").split("?")[0].rstrip("/")
                current_base = current_url.split("?")[0].rstrip("/")
                if goto_url == current_base:
                    print("[BROWSER] ⚠️ Loop detected - already on this URL! Forcing BACK")
                    action_data = {"action": "back", "args": {}}
                    action = "back"
                    args = {}
                    recent_actions = []

            # Warn if too many actions without navigation
            if actions_without_nav >= 6:
                print(f"[BROWSER] ⚠️ {actions_without_nav} actions without navigating away!")
            
            # Track this action
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
                    token_usage=token_usage,
                    step_update={
                        "status": "completed",
                        "outcome": f"Browser task completed: {summary}",
                        "notes": summary
                    }
                )

            # Execute the browser action
            result = self._execute_action(action_data, state)

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

            # Track clicked selectors (share with research agent)
            existing_clicked = list(state.get('clicked_selectors', []))
            if action_data.get("action") == "click":
                selector = action_data.get("args", {}).get("selector", "")
                if selector and selector not in existing_clicked:
                    logger.debug(f"Click {'successful' if result.success else 'FAILED'}, tracking selector: {selector[:50]}...")
                    existing_clicked.append(selector)

            # === URL-CHANGE DETECTION (NEW) ===
            # Track if clicks succeed but don't actually navigate
            failed_nav_count = state.get("browser_failed_nav_clicks", 0)
            same_page_actions = state.get("browser_same_page_actions", 0)

            if action_data.get("action") == "click" and result.success:
                # Get URL after click
                try:
                    new_page_state = self._get_page_state()
                    new_url = new_page_state.get('current_url', '') or new_page_state.get('url', '')

                    # Compare base URLs (without query params)
                    old_base = current_url.split('?')[0].rstrip('/')
                    new_base = new_url.split('?')[0].rstrip('/')

                    if old_base == new_base and new_url == current_url:
                        # Click succeeded but URL didn't change at all
                        failed_nav_count += 1
                        print(f"[BROWSER] ⚠️ Click succeeded but URL unchanged ({failed_nav_count}x)")

                        if failed_nav_count >= 3:
                            # Too many failed navigation clicks - force back or extract
                            print("[BROWSER] 🔄 Forcing back navigation - stuck on same page!")
                            tool_content += "\n\n⚠️ WARNING: Multiple clicks succeeded but page didn't navigate. Try using 'back' to return to search results, or use 'goto' with a direct URL."
                            failed_nav_count = 0  # Reset
                    else:
                        # Navigation worked, reset counter
                        failed_nav_count = 0
                except Exception as e:
                    print(f"[BROWSER] URL check warning: {e}")

            # Track same-page action count
            page_base = current_url.split('?')[0].rstrip('/')
            last_page_base = state.get("browser_last_page_base", "")

            if page_base == last_page_base:
                same_page_actions += 1
            else:
                same_page_actions = 1  # Reset on new page

            # If too many actions on same page without progress, warn strongly
            if same_page_actions >= 6:
                print(f"[BROWSER] ⚠️ {same_page_actions} actions on same page without navigating away!")
                tool_content += f"\n\n⚠️ WARNING: You've taken {same_page_actions} actions on this page without navigating. Consider using 'back' to return to search results or 'done' to complete with current findings."

            # Build updated state with loop tracking
            new_state = self._update_state(
                state,
                messages=[
                    messages[-1],  # The TEXT-ONLY prompt (crucial for history/memory!)
                    AIMessage(content=response.content),
                    tool_msg
                ],
                visited_url=visited,
                extracted_data=extracted,
                error=result.message if not result.success else None,
                token_usage=token_usage,
            )

            # Set the updated clicked selectors list
            new_state['clicked_selectors'] = existing_clicked

            # Persist recent actions for loop detection
            new_state["browser_recent_actions"] = recent_actions

            # Persist URL tracking for page-scoped loop detection
            new_state["_browser_last_url"] = current_url

            # Persist URL-change detection state
            new_state["browser_failed_nav_clicks"] = failed_nav_count
            new_state["browser_same_page_actions"] = same_page_actions
            new_state["browser_last_page_base"] = page_base


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
            
            # Detect LLM refusals before trying JSON parse
            refusal_patterns = [
                "i'm unable to", "i cannot", "i can't", "i am unable",
                "i'm not able", "i am not able", "cannot assist",
                "unable to assist", "cannot help", "sorry, but"
            ]
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in refusal_patterns):
                print(f"[BROWSER] ⚠️ LLM refused request, using scroll to find alternative")
                return {"action": "scroll", "args": {"amount": 500}}
            
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            else:
                # No JSON found at all, use scroll
                print("[BROWSER] ⚠️ No JSON in response, using scroll")
                return {"action": "scroll", "args": {"amount": 500}}

            data = json.loads(content)

            # Validate action - must have valid action type
            action = data.get("action", "")
            valid_actions = ["goto", "click", "type", "press", "scroll", "extract_visible_text", "download_image", "done", "back"]
            
            if action not in valid_actions:
                # Invalid action - log and return scroll to break potential loops
                print(f"[BROWSER] ⚠️ Invalid action '{action}' parsed, using scroll fallback")
                return {"action": "scroll", "args": {"amount": 500}}

            return data
        except json.JSONDecodeError as e:
            # On parse failure, log the issue and use scroll (not extract which causes loops!)
            print(f"[BROWSER] ⚠️ JSON parse failed: {e}. Using scroll fallback.")
            return {"action": "scroll", "args": {"amount": 500}}

    def _execute_action(self, action_data: dict, state: dict = None) -> ToolResult:
        """Execute a browser action with auto-fix for common selector issues."""
        action = action_data.get("action", "")
        args = action_data.get("args", {}).copy()  # Copy to avoid mutating original

        # Auto-fix click selectors that are missing prefix
        if action == "click" and "selector" in args:
            selector = args["selector"]
            # Check if selector looks like plain text (not CSS/xpath/text=)
            # ... (unchanged logic) ...
            needs_prefix = (
                not selector.startswith(("text=", "xpath=", "#", ".", "[", "button", "a[", "input"))
                and not selector.startswith("//")  # xpath
                and ("/" in selector or " " in selector or selector[0].islower())
            )

            if needs_prefix:
                args["selector"] = f"text={selector}"

        # PROVIDER-SPECIFIC OPTIMIZATION:
        # If accessing Anthropic (known for strict TPM limits), reduce extraction size
        if action == "extract_visible_text":
            current_max = args.get("max_chars", 8000)

            # Re-use the provider detection logic from base.py (simplified here)
            endpoint_lower = (self.config.model_endpoint or "").lower()
            model_lower = (self.config.model or "").lower()
            is_local = "localhost" in endpoint_lower or "127.0.0.1" in endpoint_lower
            is_anthropic = "anthropic" in endpoint_lower or ("claude" in model_lower and not is_local)

            if is_anthropic and current_max > 4500:
                print(f"[BROWSER] Optimizing for Anthropic TPM: Reducing extract size {current_max} -> 4500")
                args["max_chars"] = 4500

            # CHECK SHARED STATE: Skip auto_scroll if research already scrolled this page
            if state and "auto_scroll" not in args:
                try:
                    page_state = self._browser_tools.get_page_state()
                    current_url = page_state.get('current_url', '') or page_state.get('url', '')
                    scrolled_urls = state.get('scrolled_urls', [])
                    url_base = current_url.split('?')[0].rstrip('/')
                    already_scrolled = any(u.split('?')[0].rstrip('/') == url_base for u in scrolled_urls)

                    if already_scrolled:
                        args["auto_scroll"] = False
                        print("[BROWSER] Skipping scroll (research already scrolled this page)")
                except Exception:
                    pass  # Fall through if can't check

        # SKIP SCROLL if research already scrolled this page
        if action == "scroll" and state:
            try:
                page_state = self._browser_tools.get_page_state()
                current_url = page_state.get('current_url', '') or page_state.get('url', '')
                scrolled_urls = state.get('scrolled_urls', [])
                url_base = current_url.split('?')[0].rstrip('/')
                already_scrolled = any(u.split('?')[0].rstrip('/') == url_base for u in scrolled_urls)

                if already_scrolled:
                    print("[BROWSER] Skipping scroll action (research already scrolled this page)")
                    return ToolResult(success=True, message="Scroll skipped - page already scrolled by research")
            except Exception:
                pass

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


