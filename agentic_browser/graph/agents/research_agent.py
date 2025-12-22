"""
Research Agent for multi-source information gathering.

Coordinates browser actions to research topics from multiple sources.
"""

import json
import re
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
- scroll_down: { } - Scroll down to see more content (DO THIS OFTEN!)
- extract_visible_text: { "max_chars": 8000 } - Get page text
- done: { "summary": "your research findings" } - Complete with report
- search_runs: { "query": "python docs" } - Find past successful runs
- get_run_details: { "run_id": "..." } - Read steps from a past run

⬇️ SCROLL MORE! Pages often have important content below the fold.
- After landing on a content page, scroll 2-3 times to get ALL the information
- If you can't find links, SCROLL to reveal more results
- Extract text AFTER scrolling to capture the full page

=== SYSTEMATIC RESEARCH WORKFLOW ===

STEP 0: CHECK MEMORY (AUTO-INJECTED)
- I will auto-inject PROVEN STRATEGIES from your encrypted bank.
- I will auto-inject MISTAKES TO AVOID from your Apocalypse bank.
- I will also show RAW INSIGHTS from recent runs.
- PRIORITIZE encrypted banks, but consider raw insights for innovation!

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
- Visit the number of sources specified by the user (default: 3-5)

STEP 5: SYNTHESIZE INTO A CONTENT-FOCUSED REPORT
- CRITICAL: Your final report must contain ACTUAL FACTS, DATA, and FINDINGS from the extracted content
- DO NOT just describe what each page "is about" - that's useless!
- GOOD: "According to MIT research, LLMs achieved 95% accuracy on X task using technique Y"
- BAD: "The MIT article discusses advancements in AI and LLMs"
- Include specific statistics, dates, names, quotes, and examples
- Structure as: Introduction → Key Findings → Details from Each Source → Conclusion

=== CRITICAL RULES ===

1. CLICKING: Always use text= prefix for link text
   - CORRECT: {"action": "click", "args": {"selector": "text=Read More"}}
   - WRONG: {"action": "click", "args": {"selector": "r/artificial"}}

2. USE BACK: After extracting from a page, go BACK to search results to click the next link

3. SYSTEMATIC ORDERING: Visit results 1, 2, 3, 4, 5 in order - don't skip around

4. EXTRACT BEFORE NAVIGATING: Always extract_visible_text before leaving a page

5. TRACK PROGRESS: Count how many sources you've actually extracted content from

6. FOLLOW USER REQUIREMENTS: If user asks for "10 examples", gather 10+ examples

7. SKIP SPONSORED/AD LINKS! Never click links containing "Sponsored", "Ad", "Advertisement", or marked as ads

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

    def __init__(self, config, browser_tools=None, recall_tool=None):
        """Initialize research agent.
        
        Args:
            config: Agent configuration
            browser_tools: Browser tools for web access
            recall_tool: RecallTool for accessing past runs
        """
        super().__init__(config)
        self._browser_tools = browser_tools
        self.recall_tool = recall_tool
    
    # Default minimum sources if not specified in goal
    DEFAULT_MIN_SOURCES = 3
    
    # Class-level cache for tiered recall results (200 entries for 16GB RAM)
    _recall_cache: dict[str, str] = {}
    _RECALL_CACHE_SIZE = 200
    
    # Track last URL for screenshot change detection (Phase 4 optimization)
    _last_url: str = ""
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_browser_tools(self, browser_tools) -> None:
        """Set browser tools after initialization."""
        self._browser_tools = browser_tools
    
    def _parse_min_sources_from_goal(self, goal: str) -> int:
        """Parse the minimum source count from the user's goal.
        
        Looks for patterns like:
        - "at least 5 sources"
        - "from 10 sources" 
        - "minimum 3 sites"
        - "5+ sources"
        - "atleast 4 sources"
        
        Args:
            goal: The user's goal string
            
        Returns:
            The parsed count, or DEFAULT_MIN_SOURCES if not found
        """
        goal_lower = goal.lower()
        
        # Patterns to match (order matters - more specific first)
        patterns = [
            r'(\d+)\s+(?:[\w]+\s+)?(?:sources?|sites?|websites?|webpages?|pages?)',
            r'(?:at\s*least|atleast|minimum|min)\s+(\d+)\s+(?:sources?|sites?|websites?|webpages?|pages?)',
            r'(\d+)\+\s*(?:sources?|sites?|websites?|webpages?|pages?)',
            r'from\s+(\d+)\s+(?:sources?|sites?|websites?|webpages?|pages?)',
            r'(\d+)\s+(?:different|unique|separate)\s+(?:sources?|sites?|websites?|webpages?)',
            r'(?:gather|collect|find|get).*?from\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                count = int(match.group(1))
                # Sanity check: cap at reasonable limit
                count = max(1, min(count, 20))
                print(f"[RESEARCH] Parsed MIN_SOURCES={count} from goal")
                return count
        
        # Default fallback
        print(f"[RESEARCH] No source count in goal, using default MIN_SOURCES={self.DEFAULT_MIN_SOURCES}")
        return self.DEFAULT_MIN_SOURCES
    
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
        
        # === SIMPLE ACTION TRACKING (PARITY WITH BROWSER AGENT) ===
        # Don't use URL comparison - it's unreliable. Just track actions globally.
        recent_actions_list = list(state.get("recent_actions", []))
        
        # === AUTO-NAVIGATION FIX ===
        # If we're on about:blank or empty page, FORCE navigation to DuckDuckGo
        # This prevents agents from extracting empty pages and getting stuck
        if current_url == 'about:blank' or not current_url or current_url.startswith('about:'):
            # Build search query from goal
            goal = state['goal']
            # Extract key search terms (simple approach)
            search_query = goal.replace(' ', '+')[:100]
            search_url = f"https://duckduckgo.com/?q={search_query}"
            
            print(f"[RESEARCH] 🚀 AUTO-NAVIGATING: Browser on blank page, opening DuckDuckGo")
            self._browser_tools.execute("goto", {"url": search_url})
            
            return self._update_state(
                state,
                messages=[AIMessage(content=f"Auto-navigating to search: {search_url}")],
                visited_url=search_url,
            )
        
        # Determine next action hint
        # Loop Protection & Intelligent Hints
        visited_search = any('duckduckgo.com' in u for u in state['visited_urls'])
        
        if visited_search and ('duckduckgo.com' in current_url or current_url == 'about:blank'):
            if sources_visited == 0:
                 action_hint = """
CRITICAL: You have search results but haven't visited any sites yet.
1. Look at the visible content below to find search result titles
2. CLICK a result link using text= selector: {"action": "click", "args": {"selector": "text=Article Title Here"}}
3. If no good links visible, SCROLL first: {"action": "scroll", "args": {"amount": 500}}
DO NOT call "done" yet - you need to visit actual websites!
"""
            else:
                 action_hint = """
1. If you need more info, CLICK another search result
2. If satisfied, synthesize findings: {"action": "done", ...}
"""
        elif current_url == 'about:blank' or not current_url or current_url.startswith('about:'):
            action_hint = """
ACTION REQUIRED: Start your search immediately!
Construct a search URL: {"action": "goto", "args": {"url": "https://duckduckgo.com/?q=your+search+query"}}
"""
        elif 'duckduckgo.com' in current_url:
            # We are on search results - CLICK, don't goto
             action_hint = """
You are on SEARCH RESULTS. You MUST click a result to visit it:
1. Look at the visible content below for clickable titles
2. CLICK a result: {"action": "click", "args": {"selector": "text=Result Title Here"}}
3. If no links visible, SCROLL: {"action": "scroll", "args": {"amount": 500}}
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
3. If you have ENOUGH data, synthesize: {"action": "done", "args": {"summary": "..."}}
"""
        
        # Minimum sources required for completion - DYNAMIC based on user's goal
        MIN_SOURCES_REQUIRED = self._parse_min_sources_from_goal(state['goal'])
        
        # Count UNIQUE content sources (excluding search engines)
        search_engines = ['duckduckgo.com', 'google.com/search', 'bing.com/search', 'yahoo.com/search']
        content_urls = [
            u for u in state.get('visited_urls', []) 
            if u and not any(se in u.lower() for se in search_engines)
        ]
        unique_sites = set(u.split('?')[0].rstrip('/') for u in content_urls)
        sources_visited_actual = len(unique_sites)
        
        # Count research_source keys in extracted_data
        research_source_count = len([k for k in state['extracted_data'].keys() if 'research_source' in k])
        
        # === FORCED AUTO-COMPLETION ===
        # If we have enough sources, STOP immediately and synthesize
        if research_source_count >= MIN_SOURCES_REQUIRED:
            print(f"[RESEARCH] ✅ GOAL MET! {research_source_count} sources collected (needed {MIN_SOURCES_REQUIRED}). Auto-completing...")
            
            # Synthesize findings into a detailed report - include MORE content per source
            findings = []
            for key, value in state['extracted_data'].items():
                if 'research_source' in key or 'browser_extract' in key:
                    # Include up to 2000 chars per source for detailed content
                    content = str(value)[:2000]
                    if len(str(value)) > 2000:
                        content += "..."
                    findings.append(f"### {key}\n{content}\n")
            
            # Build a proper detailed report
            summary = f"""## Research Report: {state['goal'][:100]}

**Sources Collected:** {research_source_count}

---

{chr(10).join(findings[:5])}

---

**Note:** Above is the full extracted content from each source. Review for specific details like prices, specifications, dates, and actionable information.
"""
            return self._update_state(
                state,
                task_complete=False,  # Let supervisor decide final completion
                final_answer=summary,
                extracted_data={"research_findings": summary},
                step_update={"status": "completed", "outcome": f"Research complete with {research_source_count} sources"}
            )
        
        if sources_visited_actual >= MIN_SOURCES_REQUIRED:
            action_hint = f"""
MANDATORY COMPLETION: You have visited {MIN_SOURCES_REQUIRED}+ sources as requested.
You MUST stop researching now and synthesize your findings.
Action: {{"action": "done", "args": {{"summary": "## Research Report\\n\\n[Your detailed findings here]"}}}}
"""
        else:
            action_hint = f"Continue research. Need {MIN_SOURCES_REQUIRED - sources_visited_actual} more sources."
        
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
        
        # === ROOT CAUSE FIX: Show LLM what it already tried ===
        recent_actions_list = state.get('recent_actions', [])
        if recent_actions_list:
            actions_str = "\n".join(f"  - {a}" for a in recent_actions_list[-5:])
            recent_actions_hint = f"""
⚠️ YOUR RECENT ACTIONS (DO NOT REPEAT!):\n{actions_str}

Choose a DIFFERENT action! Options:
- Click a link you HAVEN'T clicked yet
- Use 'back' to return to search results
- Use 'done' if you have enough data
"""
        else:
            recent_actions_hint = ""
        
        # Check if we've reached the bottom of the page (from scroll detection)
        reached_bottom = state.get('_reached_page_bottom', False)
        bottom_hint = ""
        if reached_bottom:
            bottom_hint = """
⚠️ PAGE BOTTOM REACHED: You have scrolled to the bottom of this page.
DO NOT scroll again. Instead:
1. Look for "More results", "Next page", "Load more" buttons and click them: {"action": "click", "args": {"selector": "text=More results"}}
2. Extract the content now: {"action": "extract_visible_text", "args": {"max_chars": 12000}}
3. Go back to find more sources: {"action": "back", "args": {}}
"""
        
        task_context = f"""
RESEARCH TASK: {state['goal']}

=== PROGRESS ===
Unique websites visited: {sources_visited}/{MIN_SOURCES_REQUIRED}
{progress_msg}
{clicked_warning}
{recent_actions_hint}
{bottom_hint}
Visited URLs (last 10): {chr(10).join(list(state['visited_urls'])[-10:]) or '(none)'}

Current page: {page_state.get('page_title', '') or page_state.get('title', 'Unknown')}
URL: {current_url}

{action_hint}

Visible content (truncated):
{page_state.get('visible_text', '')[:800]}

Data collected (summary):
{json.dumps(state['extracted_data'], indent=2)[:1200] if state['extracted_data'] else '(none yet)'}
"""
        # CRITICAL: Guard against task_context explosion
        MAX_TASK_CONTEXT = 8000
        if len(task_context) > MAX_TASK_CONTEXT:
            task_context = task_context[:MAX_TASK_CONTEXT] + "\n...[Task context truncated]"

        # Vision mode: capture screenshot for LLM
        # OPTIMIZATION: Only capture if URL changed (saves ~100-300ms per step)
        screenshot_b64 = None
        
        if self.config.vision_mode and self._browser_tools:
            # Check if URL changed since last capture
            url_changed = current_url != self._last_url
            
            if url_changed:
                screenshot_b64 = self.capture_screenshot_base64(self._browser_tools)
                self._last_url = current_url  # Update tracking
                
                if screenshot_b64:
                    task_context += """

[VISION MODE] A screenshot of the current page is attached.
⚠️ CRITICAL VISION RULES - READ CAREFULLY:
1. ONLY click links that appear VERBATIM in the "Visible content" text below
2. DO NOT paraphrase or approximate link text - use the EXACT wording
3. If the link text you want is not in "Visible content", the element does NOT exist
4. When in doubt, use extract_visible_text first to see what's actually on the page
5. If clicks keep failing, SCROLL to load more content or use BACK to return to search results
6. NEVER make up link text - only use text you can copy from the visible content
"""
                    print("[RESEARCH] Vision mode: screenshot captured (page changed)")
            else:
                print("[RESEARCH] Vision mode: skipped screenshot (page unchanged)")
        
        # TIERED RECALL INJECTION (Strategies > Apocalypse > Raw Runs)
        # Uses caching + async for 2x faster performance
        step_count = state.get('step_count', 0)
        if step_count <= 1:
            import hashlib
            goal = state['goal']
            cache_key = hashlib.md5(goal.encode()).hexdigest()[:16]
            
            # Skip if --no-memory flag is set for faster startup
            if getattr(self.config, 'no_memory', False):
                recall_context = ""
            # Check cache first (fast path)
            elif cache_key in self._recall_cache:
                recall_context = self._recall_cache[cache_key]
                print("[RESEARCH] 🧠 Using cached tiered recall context")
            else:
                try:
                    from ..knowledge_base import get_knowledge_base
                    kb = get_knowledge_base()
                    
                    # Use parallel version with 2s total timeout
                    recall_result = kb.tiered_recall_async("research", goal)
                    recall_context = recall_result.to_prompt_injection()
                    
                    # CRITICAL: Guard against recall_context explosion
                    MAX_RECALL_CONTEXT = 2000
                    if len(recall_context) > MAX_RECALL_CONTEXT:
                        recall_context = recall_context[:MAX_RECALL_CONTEXT] + "\n...[Recall truncated]"
                    
                    # Cache with size limit
                    if len(self._recall_cache) >= self._RECALL_CACHE_SIZE:
                        # Clear half the cache
                        keys = list(self._recall_cache.keys())
                        for k in keys[:len(keys)//2]:
                            del self._recall_cache[k]
                        print(f"[RESEARCH] Evicted {len(keys)//2} cached recall entries")
                    
                    self._recall_cache[cache_key] = recall_context
                    print("[RESEARCH] 🧠 Computed (parallel) and cached tiered recall")
                except Exception as e:
                    recall_context = ""
                    print(f"[RESEARCH] ⚠️ Tiered recall failed: {e}")
            
            if recall_context:
                task_context = f"{recall_context}\n\n---\n\n{task_context}"
        
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
            
            # Debug: show what action was chosen
            print(f"[RESEARCH] Action: {action_data.get('action')}, Args: {action_data.get('args', {})}")
            
            if action_data.get("action") == "done":
                # HARD BLOCK: Reject 'done' if insufficient research content
                # Count research_source_N keys in extracted_data (more reliable than visited_urls
                # because visited_urls doesn't include the current extraction yet)
                research_sources_count = len([k for k in state.get('extracted_data', {}).keys() if 'research_source' in k])
                
                # Use dynamic MIN_SOURCES from goal
                MIN_SOURCES = self._parse_min_sources_from_goal(state['goal'])
                
                if research_sources_count < MIN_SOURCES:
                    # Need to visit more sites - force extraction won't help, need to navigate
                    print(f"[RESEARCH] Blocking done: only have {research_sources_count}/{MIN_SOURCES} research sources extracted (user requested {MIN_SOURCES})")
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
                        token_usage=token_usage,
                        step_update={
                            "status": "completed",
                            "outcome": f"Research completed: {summary[:100]}...",
                            "notes": summary
                        }
                    )
            
            # If there are recent failed clicks, force scroll to find real links
            # Check last 3 messages for click failure patterns AND check error field
            recent_msgs = state.get('messages', [])[-6:]  # Last 3 pairs of messages
            recent_fails = sum(1 for m in recent_msgs if hasattr(m, 'content') and 'Could not click' in str(m.content))
            # Also check if current state has a click error
            current_error = state.get('error', '') or ''
            
            # Auto-scroll logic removed to prevent loops
            
            # Map 'scroll_down' action to 'scroll' tool
            if action_data.get("action") == "scroll_down":
                action_data["action"] = "scroll"
                action_data["args"] = {"amount": 800}
            
            # RATE LIMIT: Detect consecutive scroll actions and force extract instead
            if action_data.get("action") == "scroll":
                recent_actions_list = state.get('recent_actions', [])
                # Count recent scrolls in last 3 actions
                recent_scrolls = sum(1 for a in recent_actions_list[-3:] if a == "scroll")
                if recent_scrolls >= 2:
                    print(f"[RESEARCH] 🛑 Blocking repeated scroll (seen {recent_scrolls} scrolls) - forcing extract")
                    action_data = {"action": "extract_visible_text", "args": {"max_chars": 8000, "auto_scroll": False}}
            
            # Detect duplicate click and force scroll instead
            if action_data.get("action") == "click":
                selector = action_data.get("args", {}).get("selector", "")
                clicked_selectors = state.get('clicked_selectors', [])
                unique_clicked = list(dict.fromkeys(clicked_selectors))
                
                # PRE-CHECK: Verify element exists before attempting click
                # This prevents wasting time on hallucinated selectors from vision mode
                if selector and self._browser_tools:
                    from agentic_browser.utils import format_selector
                    formatted_selector = format_selector(selector)
                    try:
                        element_count = self._browser_tools.page.locator(formatted_selector).count()
                        if element_count == 0:
                            print(f"[RESEARCH] ⚠️ Element NOT FOUND: {selector[:60]}... - forcing extract to see available links")
                            # Force extract to show what's actually on the page
                            action_data = {"action": "extract_visible_text", "args": {"max_chars": 8000}}
                    except Exception as e:
                        print(f"[RESEARCH] Element pre-check failed: {e}")
                
                if selector in unique_clicked:
                    print(f"[RESEARCH] 🔄 Duplicate click detected: {selector[:50]}... - using BACK instead")
                    action_data = {"action": "back", "args": {}}
            
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
                            # Force extract first with auto-scroll
                            action_data = {"action": "extract_visible_text", "args": {"max_chars": 12000, "auto_scroll": True}}
                        else:
                            print(f"[RESEARCH] ✅ Already visited {current_url_now[:50]}..., proceeding with back")
                except Exception as e:
                    print(f"[RESEARCH] Warning: auto-extract check failed: {e}")
            
            # Execute browser action
            final_args = action_data.get("args", {})
            if action_data.get("action") == "extract_visible_text":
                # Only auto-scroll on first extraction of a page (avoid repeated slow scrolling)
                current_url = page_state.get('current_url', '') or page_state.get('url', '')
                scrolled_urls = state.get('scrolled_urls', [])
                
                # Check if we've already scrolled this URL
                url_base = current_url.split('?')[0].rstrip('/')
                already_scrolled = any(u.split('?')[0].rstrip('/') == url_base for u in scrolled_urls)
                
                if "auto_scroll" not in final_args:
                    final_args["auto_scroll"] = not already_scrolled  # Only scroll first time
                    if already_scrolled:
                        print(f"[RESEARCH] Skipping auto_scroll (already scrolled this page)")
                
                # Also increase chars if not specified
                if "max_chars" not in final_args:
                    final_args["max_chars"] = 12000
            
            result = self._browser_tools.execute(
                action_data.get("action", ""),
                final_args,
            )
            
            # Track this URL as scrolled if we just did auto_scroll
            if action_data.get("action") == "extract_visible_text" and final_args.get("auto_scroll"):
                scrolled_url = current_url  # Will be added to state later
            
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
                            token_usage=token_usage,
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
                        token_usage=token_usage,
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
                
            # Enhanced error handling for click failures - show available alternatives
            if action_data.get("action") == "click" and not result.success:
                # Extract available links from the error data
                available_links = []
                suggestion = ""
                if result.data:
                    available_links = result.data.get("available_links", [])
                    suggestion = result.data.get("suggestion", "")
                
                if available_links:
                    links_list = "\n".join(f'  - text="{link}"' for link in available_links[:6])
                    tool_content += f"\n\n🔗 AVAILABLE LINKS on this page (pick one of these EXACTLY):\n{links_list}"
                    if suggestion:
                        tool_content += f"\n\n💡 Suggestion: {suggestion}"
            
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
            
            # === SCROLL STATE TRACKING (Simplified) ===
            # Limit scrolled_urls to last 10 to prevent unbounded growth
            existing_scrolled = state.get('scrolled_urls', [])[-9:]
            
            if action_data.get("action") == "scroll":
                new_state['last_action_was_scroll'] = True
                new_state['scrolled_urls'] = existing_scrolled + [current_url]
            
            # Track scrolled_urls for extract_visible_text with auto_scroll
            if action_data.get("action") == "extract_visible_text" and final_args.get("auto_scroll"):
                new_state['scrolled_urls'] = existing_scrolled + [current_url]
                
                # Simplified: just mark as scrolled, skip expensive position checking
                # The RATE LIMIT check at line 473 already handles scroll loops
                pass  # No expensive page.evaluate needed
            else:
                # Clear scroll state for non-scroll actions
                new_state['last_action_was_scroll'] = False
            
            # === PERSIST URL AND ACTION TRACKING (PARITY WITH BROWSER AGENT) ===
            new_state['_research_last_url'] = current_url
            
            # Track action for loop detection
            action_name = action_data.get("action", "unknown")
            recent_actions_list.append(action_name)
            if len(recent_actions_list) > 10:
                recent_actions_list = recent_actions_list[-10:]
            new_state['recent_actions'] = recent_actions_list
            
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
            
            # Detect LLM refusals before trying JSON parse
            refusal_patterns = [
                "i'm unable to", "i cannot", "i can't", "i am unable",
                "i'm not able", "i am not able", "cannot assist",
                "unable to assist", "cannot help", "sorry, but"
            ]
            content_lower = content.lower()
            if any(pattern in content_lower for pattern in refusal_patterns):
                print(f"[RESEARCH] ⚠️ LLM refused request, using scroll to find alternative")
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
                print("[RESEARCH] ⚠️ No JSON in response, using scroll")
                return {"action": "scroll", "args": {"amount": 500}}
            
            data = json.loads(content)
            
            # Validate action - must be valid type
            action = data.get("action", "")
            valid_actions = ["goto", "click", "back", "extract_visible_text", "scroll", "done"]
            
            if action not in valid_actions:
                print(f"[RESEARCH] ⚠️ Invalid action '{action}', using scroll fallback")
                return {"action": "scroll", "args": {"amount": 500}}
                
            return data
        except json.JSONDecodeError as e:
            # On parse failure, use scroll (back causes blank page loops)
            print(f"[RESEARCH] ⚠️ JSON parse failed: {e}. Using scroll fallback.")
            return {"action": "scroll", "args": {"amount": 500}}


def research_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for research agent."""
    from ..tool_registry import get_tools
    
    # Get tools from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
        browser_tools = tools.browser_tools
        recall_tool = tools.recall_tool
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig()
        browser_tools = None
        recall_tool = None
    
    agent = ResearchAgentNode(agent_config, browser_tools, recall_tool)
    return agent.execute(state)


