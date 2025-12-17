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
    
    SYSTEM_PROMPT = """You are a RESEARCH agent. Find information from the web.

Available actions:
- goto: { "url": "https://..." } - Navigate to URL
- type: { "selector": "selector", "text": "text" } - Type into input
- press: { "key": "Enter" } - Press a key
- click: { "selector": "selector" } - Click an element
- extract_visible_text: { "max_chars": 5000 } - Get page text
- done: { "summary": "your research findings" } - Complete with report

SEARCH WORKFLOW:
1. Construct search URL: "https://duckduckgo.com/?q=your+query"
2. Navigate directly to it using "goto"
3. Extract search results to find real URLs (do not guess!)
4. Visit 1-3 actual websites from search results
5. Synthesize findings into a comprehensive report

ADAPTIVE DEPTH:
- SIMPLE query ("what is X?"): 1 search, 1-2 sites, quick summary
- COMPLEX query ("compare A vs B"): multiple searches, more sites, detailed comparison

CRITICAL RULES:
1. ALWAYS start with DuckDuckGo - don't make up URLs!
2. Only visit URLs you actually see in search results
3. If a site fails (ERR_NAME_NOT_RESOLVED, 404), skip it and try another
4. After 3+ errors on same URL, move on

ERROR RECOVERY:
- If research stalls, synthesize what you have and call "done"
- A partial report is better than no report

Respond with JSON:
{
  "action": "goto|type|press|click|extract_visible_text|done",
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
        current_url = page_state.get('url', 'about:blank')
        
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
        
        elif sources_visited >= 2:
            action_hint = """
You have visited enough sources. Synthesize your findings now:
{"action": "done", "args": {"summary": "## Research Report\\n\\n[Your findings here]"}}
"""
        else:
            action_hint = "Continue research."
        
        task_context = f"""
RESEARCH TASK: {state['goal']}

Sources visited: {sources_visited}/2-3 recommended
Visited URLs: {chr(10).join(state['visited_urls'][-5:]) or '(none)'}

Current page: {page_state.get('title', 'Unknown')}
URL: {current_url}

{action_hint}

Visible content (truncated):
{page_state.get('visible_text', '')[:2000]}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.llm.invoke(messages)
            action_data = self._parse_action(response.content)
            
            if action_data.get("action") == "done":
                # HARD BLOCK: Reject 'done' if no research content collected
                has_research_content = any(
                    'research_source' in k for k in state['extracted_data'].keys()
                )
                
                if not has_research_content:
                    # Force extraction instead of allowing premature completion
                    action_data = {
                        "action": "extract_visible_text",
                        "args": {"max_chars": 8000}
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
                # content might be dict {"text": "..."} or string
                content_str = ""
                if isinstance(result.data, dict):
                    content_str = result.data.get("text", str(result.data))
                else:
                    content_str = str(result.data) if result.data else str(result.message)
                
                extracted = {key: content_str[:2000]} # Increased capture size
            
            # Create tool output message
            tool_content = "Action successful."
            if result.message:
                tool_content = str(result.message)
            elif result.data and not extracted:
                 # If we didn't extract to extracted_data, show it here
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
                return {
                    "action": "done",
                    "args": {"summary": "Agent error: Failed to generate valid action"}
                }
                
            return data
        except json.JSONDecodeError:
            return {"action": "done", "args": {"summary": "Failed to parse JSON response"}}


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


