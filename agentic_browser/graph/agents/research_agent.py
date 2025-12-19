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
    MAX_STEPS_PER_INVOCATION = 15
    
    SYSTEM_PROMPT = """You are a RESEARCH agent. Find information from the web SYSTEMATICALLY.

Available actions:
- goto: { "url": "https://..." } - Navigate to URL
- click: { "selector": "text=Link Text" } - Click links (USE text= prefix!)
- back: { } - Go back to previous page (e.g., to return to search results)
- extract_visible_text: { "max_chars": 8000 } - Get page text
- done: { "summary": "your research findings" } - Complete with report

=== SYSTEMATIC RESEARCH WORKFLOW ===

STEP 1: SEARCH
- Go to DuckDuckGo: {"action": "goto", "args": {"url": "https://duckduckgo.com/?q=your+search+terms"}}

STEP 2: EXTRACT SEARCH RESULTS
- ALWAYS extract the search page: {"action": "extract_visible_text", "args": {"max_chars": 8000}}
- Look for clickable link text in the results

STEP 3: CLICK RESULTS (PREFERRED) or GOTO
- PREFERRED: Click links directly from search results:
  {"action": "click", "args": {"selector": "text=Article Title Here"}}
- After visiting a page, use BACK to return to search results:
  {"action": "back", "args": {}}
- Then click the next result

NOTE: Using click+back is MORE RELIABLE than direct goto because:
- Search result URLs can be tracking links that redirect
- Direct URLs may be outdated or 404
- Clicking the actual link follows the correct redirect

STEP 4: VISIT MULTIPLE SOURCES
- Click result 1 → Extract → Back → Click result 2 → Extract → etc.
- Continue until you have 3-5 sources

STEP 5: SYNTHESIZE
- Only call "done" after visiting 3+ different websites
- Include SPECIFIC facts and examples from each source
- Name your sources in the summary

=== CRITICAL RULES ===

1. CLICKING: Always use text= prefix for link text
   - CORRECT: {"action": "click", "args": {"selector": "text=Read More"}}
   - WRONG: {"action": "click", "args": {"selector": "r/artificial"}}

2. USE BACK: After extracting from a page, go BACK to search results to click the next link

3. SYSTEMATIC ORDERING: Visit results 1, 2, 3, 4, 5 in order - don't skip around

4. EXTRACT BEFORE NAVIGATING: Always extract_visible_text before leaving a page

5. TRACK PROGRESS: Count how many sources you've actually extracted content from

6. FOLLOW USER REQUIREMENTS: If user asks for "10 examples", gather 10+ examples

=== ERROR RECOVERY ===
- If a site fails (404, CAPTCHA, paywall), go BACK and try the next result
- After 3 failures in a row, synthesize what you have and call "done"
- A partial report from 3 good sources beats nothing

Respond with JSON:
{
  "action": "goto|click|back|extract_visible_text|done",
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
        current_url = page_state.get('current_url', '') or page_state.get('url', 'about:blank')
        
        # Determine next action hint
        # Loop Protection & Intelligent Hints
        visited_search = any('duckduckgo.com' in u for u in state['visited_urls'])
        
        if visited_search and ('duckduckgo.com' in current_url or current_url == 'about:blank'):
            if sources_visited == 0:
                 action_hint = """
CRITICAL: You have search results but haven't visited any sites yet.
1. Extract text to find links: {"action": "extract_visible_text", ...}
2. MUST visit at least 1 result: {"action": "goto", "args": {"url": "https://..."}}
DO NOT call "done" yet.
"""
            else:
                 action_hint = """
1. If you need more info, visit another result.
2. If satisfied, synthesize findings: {"action": "done", ...}
"""
        elif current_url == 'about:blank' or not current_url or current_url.startswith('about:'):
            action_hint = """
ACTION REQUIRED: Start your search immediately!
Construct a search URL: {"action": "goto", "args": {"url": "https://duckduckgo.com/?q=your+search+query"}}
"""
        elif 'duckduckgo.com' in current_url:
            # We are on search results
             action_hint = """
You are on search results.
1. Extract text: {"action": "extract_visible_text", "args": {"max_chars": 5000}}
2. Then visit a result: {"action": "goto", "args": {"url": "https://..."}}
"""
        elif 'duckduckgo.com' not in current_url and current_url != 'about:blank':
            # Content page logic - MUST extract before anything else
            # Check if we have any research content yet
            has_research_content = any(
                'research_source' in k for k in state['extracted_data'].keys()
            )
            
            if not has_research_content:
                action_hint = """
MANDATORY: You are on a content page but have NOT extracted any research data yet.
You MUST extract text from this page:
{"action": "extract_visible_text", "args": {"max_chars": 8000}}

DO NOT call "done". DO NOT navigate away. EXTRACT FIRST.
"""
            else:
                action_hint = """
You have some research data. Options:
1. Extract this page too: {"action": "extract_visible_text", ...}
2. Visit another result to get more info.
3. If you have ENOUGH data (2+ sources), synthesize: {"action": "done", "args": {"summary": "..."}}
"""
        
        elif sources_visited >= 3:
            action_hint = """
MANDATORY COMPLETION: You have visited 3+ sources.
You MUST stop researching now.
Action: {"action": "done", "args": {"summary": "## Research Report\\n\\n[Your detailed findings here]"}}
"""
        else:
            action_hint = "Continue research."
        
        # Minimum sources required for completion
        MIN_SOURCES_REQUIRED = 3
        
        # Calculate progress based on UNIQUE content URLs visited (excluding search engines)
        search_engines = ['duckduckgo.com', 'google.com/search', 'bing.com/search', 'yahoo.com/search']
        content_urls = [
            u for u in state.get('visited_urls', []) 
            if u and not any(se in u.lower() for se in search_engines)
        ]
        # Dedupe by base URL (before query params)
        unique_sites = set(u.split('?')[0].rstrip('/') for u in content_urls)
        sources_visited = len(unique_sites)
        sources_needed = max(0, MIN_SOURCES_REQUIRED - sources_visited)
        
        # Strong progress indicator
        if sources_needed > 0:
            progress_msg = f"⚠️ NEED {sources_needed} MORE SOURCES before you can call done!"
        else:
            progress_msg = "✅ You have enough sources. You may call done with a comprehensive summary."
        
        # Track clicked link selectors to avoid re-clicking
        clicked_selectors = state.get('clicked_selectors', [])
        
        # Deduplicate for display (operator.add can create duplicates)
        unique_clicked = list(dict.fromkeys(clicked_selectors))  # Preserves order
        print(f"[RESEARCH DEBUG] clicked_selectors in state: {len(clicked_selectors)} total, {len(unique_clicked)} unique")
        
        # Format clicked links warning with deduped list
        if unique_clicked:
            clicked_warning = f"""
⛔ ALREADY CLICKED ({len(unique_clicked)} unique links) - DO NOT CLICK THESE AGAIN:
{chr(10).join(f'  - {sel}' for sel in unique_clicked[-10:])}

You MUST click a DIFFERENT link from the visible results!
"""
            # If already clicked 2+ unique links, suggest scroll then click
            if len(unique_clicked) >= 2:
                clicked_warning += f"""
📋 After scrolling, click a NEW link you haven't clicked before!
Example: {{"action": "click", "args": {{"selector": "text=Some New Article Title"}}}}
"""
        else:
            clicked_warning = ""
        
        # Add error hint if there have been consecutive failures
        consecutive_err = state.get('consecutive_errors', 0)
        if consecutive_err >= 1:
            clicked_warning += f"""
⚠️ LAST ACTION FAILED! You may be clicking links that don't exist.
Use ONLY selectors from the visible content below. Look for real link text like:
  - "Prevalence and Characteristics..."  
  - "Harm Reduction Journal..."
DO NOT make up link titles!
"""
        
        task_context = f"""
RESEARCH TASK: {state['goal']}

=== PROGRESS ===
Unique websites visited: {sources_visited}/{MIN_SOURCES_REQUIRED}
{progress_msg}
{clicked_warning}
Visited URLs: {chr(10).join(state['visited_urls'][-5:]) or '(none)'}

Current page: {page_state.get('page_title', '') or page_state.get('title', 'Unknown')}
URL: {current_url}

{action_hint}

Visible content (truncated):
{page_state.get('visible_text', '')[:2000]}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            # Debug: show what action was chosen
            print(f"[RESEARCH] Action: {action_data.get('action')}, Args: {action_data.get('args', {})}")
            
            if action_data.get("action") == "done":
                # HARD BLOCK: Reject 'done' if insufficient research content
                # Check unique content sites visited (reuse logic from above)
                search_engines = ['duckduckgo.com', 'google.com/search', 'bing.com/search', 'yahoo.com/search']
                content_urls = [
                    u for u in state.get('visited_urls', []) 
                    if u and not any(se in u.lower() for se in search_engines)
                ]
                unique_sites = set(u.split('?')[0].rstrip('/') for u in content_urls)
                sites_visited = len(unique_sites)
                
                MIN_SOURCES = 3
                
                if sites_visited < MIN_SOURCES:
                    # Need to visit more sites - force extraction won't help, need to navigate
                    print(f"[RESEARCH] Blocking done: only visited {sites_visited}/{MIN_SOURCES} unique sites")
                    action_data = {
                        "action": "back",  # Go back to find more sources
                        "args": {}
                    }
                    # Fall through to execute this action
                else:
                    # Allow completion with research data
                    summary = action_data.get("args", {}).get("summary", "Research completed")
                    return self._update_state(
                        state,
                        messages=[AIMessage(content=response.content)],
                        task_complete=True,
                        final_answer=summary,
                        extracted_data={"research_findings": summary},
                    )
            
            # If there are recent failed clicks, force scroll to find real links
            # Check last 3 messages for click failure patterns AND check error field
            recent_msgs = state.get('messages', [])[-6:]  # Last 3 pairs of messages
            recent_fails = sum(1 for m in recent_msgs if hasattr(m, 'content') and 'Could not click' in str(m.content))
            # Also check if current state has a click error
            current_error = state.get('error', '') or ''
            if 'Could not click' in current_error:
                recent_fails += 1
            
            # Check if we just scrolled - if so, force extract to update context
            last_action_was_scroll = state.get('last_action_was_scroll', False)
            print(f"[RESEARCH DEBUG] last_action_was_scroll={last_action_was_scroll}, action={action_data.get('action')}")
            if last_action_was_scroll and action_data.get("action") == "click":
                # LLM is trying to click but hasn't seen new content after scroll
                print(f"[RESEARCH] 📋 Just scrolled - forcing extract to show new visible content before clicking")
                action_data = {"action": "extract_visible_text", "args": {"max_chars": 4000}}
                # Will clear the flag after execution below
                
            if recent_fails >= 1 and action_data.get("action") == "click":
                print(f"[RESEARCH] ⚠️ {recent_fails} recent click failure(s) - forcing scroll to find real links")
                action_data = {"action": "scroll", "args": {"amount": 800}}
            
            # Detect duplicate click and force scroll instead
            if action_data.get("action") == "click":
                selector = action_data.get("args", {}).get("selector", "")
                clicked_selectors = state.get('clicked_selectors', [])
                unique_clicked = list(dict.fromkeys(clicked_selectors))
                
                if selector in unique_clicked:
                    print(f"[RESEARCH] 🔄 Duplicate click detected: {selector[:50]}... - forcing scroll instead")
                    action_data = {"action": "scroll", "args": {"amount": 800}}
            
            # Auto-extract before going back if on a content page
            if action_data.get("action") == "back":
                try:
                    page_state_now = self._browser_tools.get_page_state()
                    current_url_now = page_state_now.get('current_url', '') or page_state_now.get('url', '')
                    is_search_page = any(se in current_url_now.lower() for se in ['duckduckgo.com', 'google.com/search', 'bing.com/search'])
                    
                    if current_url_now and not is_search_page and current_url_now != 'about:blank':
                        # Check if we already visited this URL (means we already extracted)
                        normalized_url = current_url_now.split('?')[0].rstrip('/')
                        already_visited = any(
                            u.split('?')[0].rstrip('/') == normalized_url
                            for u in state.get('visited_urls', []) if u
                        )
                        if not already_visited:
                            print(f"[RESEARCH] 📦 Auto-extracting content before going back from: {current_url_now[:50]}...")
                            # Force extract first
                            action_data = {"action": "extract_visible_text", "args": {"max_chars": 8000}}
                        else:
                            print(f"[RESEARCH] ✅ Already visited {current_url_now[:50]}..., proceeding with back")
                except Exception as e:
                    print(f"[RESEARCH] Warning: auto-extract check failed: {e}")
            
            # Execute browser action
            result = self._browser_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            visited = None
            if action_data.get("action") == "goto":
                target_url = action_data.get("args", {}).get("url", "")
                # Check for duplicate URL (skip search engines)
                if target_url and 'duckduckgo' not in target_url:
                    # Normalize URL for comparison
                    normalized = target_url.split('?')[0].rstrip('/')
                    already_visited = any(
                        u.split('?')[0].rstrip('/') == normalized 
                        for u in state['visited_urls'] if u
                    )
                    if already_visited:
                        # Skip duplicate, try to continue
                        return self._update_state(
                            state,
                            messages=[AIMessage(content=f"Skipped duplicate URL: {target_url}")],
                            error=f"Already visited {target_url}, try a different source",
                        )
                visited = target_url
            
            extracted = None
            if action_data.get("action") == "extract_visible_text" and result.success:
                # Get current URL to check if we're on a search engine
                try:
                    page_state = self._browser_tools.get_page_state()
                    current_url = page_state.get('current_url', '') or page_state.get('url', '')
                except:
                    current_url = ''
                
                # DON'T save content from search engine pages
                search_engines = ['duckduckgo.com', 'google.com/search', 'bing.com/search', 'yahoo.com/search']
                is_search_page = any(se in current_url.lower() for se in search_engines)
                
                if is_search_page:
                    # This is a search results page - don't save as research source
                    # Just return the content for the agent to process and click a result
                    tool_msg = HumanMessage(content=f"Search results extracted. Now click on a relevant result link to get actual content.")
                    return self._update_state(
                        state,
                        messages=[AIMessage(content=response.content), tool_msg],
                        visited_url=visited,
                    )
                
                # Get content
                content_str = ""
                if isinstance(result.data, dict):
                    content_str = result.data.get("text", str(result.data))
                else:
                    content_str = str(result.data) if result.data else str(result.message)
                
                # Quality check: detect paywalls, access denied, CAPTCHA, etc.
                lower_content = content_str.lower()[:1000]
                is_low_quality = any(phrase in lower_content for phrase in [
                    "access denied", "403 forbidden", "404 not found",
                    "subscribe to read", "subscription required", "paywall",
                    "please enable javascript", "browser not supported",
                    "captcha", "robot", "cookies must be enabled",
                    "sign in to continue", "create an account to",
                    "verify you", "human verification", "are you human",
                    "security check", "checking your browser", "just a moment",
                ])
                
                if is_low_quality and len(content_str) < 500:
                    # Low quality content - note but don't save
                    # IMPORTANT: Mark URL as visited to prevent auto-extract loop on CAPTCHA pages
                    return self._update_state(
                        state,
                        messages=[AIMessage(content=f"Low quality content (CAPTCHA/paywall) at {current_url[:50]}..., skipping this source")],
                        visited_url=current_url,  # Mark as visited to break auto-extract loop
                        error="Content blocked or inaccessible, try another URL",
                    )
                
                # Also filter out content that's just navigation menus
                nav_junk_phrases = [
                    "open menu all images", "shopping videos more news maps",
                    "search assist duck.ai", "never tracks your searches",
                    "main menu search", "contents hide", "toggle the table",
                ]
                is_nav_junk = any(phrase in lower_content for phrase in nav_junk_phrases)
                
                if is_nav_junk and len(content_str) < 800:
                    return self._update_state(
                        state,
                        messages=[AIMessage(content="Page content appears to be navigation/UI only, try extracting from a content page")],
                        visited_url=current_url,  # Mark as visited to prevent loops
                        error="No substantial content found, click a result link first",
                    )
                
                # Save good content - number based on existing research_sources
                existing_sources = len([k for k in state['extracted_data'].keys() if 'research_source' in k])
                key = f"research_source_{existing_sources + 1}"
                extracted = {key: content_str[:2000]}
                
                # Mark URL as visited so we don't re-extract (fixes auto-extract loop)
                visited = current_url
            
            # Create tool output message
            tool_content = "Action successful."
            if result.message:
                tool_content = str(result.message)
            elif result.data and not extracted:
                 # If we didn't extract to extracted_data, show it here
                tool_content = str(result.data)[:1000]
                
            tool_msg = HumanMessage(content=f"Tool output: {tool_content}")
            
            # Track clicked selectors to avoid re-clicking (including FAILED clicks)
            # NOTE: Without operator.add, we manually copy and extend the list
            existing_clicked = list(state.get('clicked_selectors', []))
            if action_data.get("action") == "click":
                selector = action_data.get("args", {}).get("selector", "")
                if selector and selector not in existing_clicked:
                    if result.success:
                        print(f"[RESEARCH DEBUG] Click successful, adding selector: {selector}")
                    else:
                        print(f"[RESEARCH DEBUG] Click FAILED, marking selector as tried: {selector}")
                    existing_clicked.append(selector)  # Add both success AND failed clicks
            
            new_state = self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                visited_url=visited,
                extracted_data=extracted,
                error=result.message if not result.success else None,
            )
            
            # Set the updated clicked selectors list
            new_state['clicked_selectors'] = existing_clicked
            
            # Track scroll state - set flag when scroll, clear when extract
            if action_data.get("action") == "scroll":
                new_state['last_action_was_scroll'] = True
                print("[RESEARCH] 📜 Scrolled - will force extract on next step to update context")
            elif action_data.get("action") == "extract_visible_text":
                new_state['last_action_was_scroll'] = False
            
            return new_state
            
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
            
            data = json.loads(content)
            
            # Validate action
            if not data.get("action"):
                # Default to text extraction instead of done
                return {
                    "action": "extract_visible_text",
                    "args": {"max_chars": 8000}
                }
                
            return data
        except json.JSONDecodeError:
            # On parse failure, extract instead of quitting
            return {"action": "extract_visible_text", "args": {"max_chars": 8000}}


def research_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for research agent."""
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
    
    agent = ResearchAgentNode(agent_config, browser_tools)
    return agent.execute(state)


