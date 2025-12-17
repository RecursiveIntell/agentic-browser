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
1. Go to https://duckduckgo.com
2. Type your search query in input[name='q']
3. Press Enter to search
4. Extract results to find real URLs
5. Visit 1-3 actual websites from search results
6. Synthesize findings into a comprehensive report

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
        if current_url == 'about:blank' or not current_url or current_url.startswith('about:'):
            action_hint = """
ACTION REQUIRED: You are on a blank page!
Your FIRST action MUST be: {"action": "goto", "args": {"url": "https://duckduckgo.com"}}
"""
        elif 'duckduckgo.com' in current_url and not state['visited_urls']:
            action_hint = """
ACTION REQUIRED: You are on DuckDuckGo.
Type your search: {"action": "type", "args": {"selector": "input[name='q']", "text": "your search query"}}
Then press Enter: {"action": "press", "args": {"key": "Enter"}}
"""
        elif sources_visited >= 2:
            action_hint = """
You have visited enough sources. Synthesize your findings now:
{"action": "done", "args": {"summary": "## Research Report\\n\\n[Your findings here]"}}
"""
        else:
            action_hint = "Continue research or call done if you have enough info."
        
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
                # Research agent CAN mark complete - it's typically the final step
                summary = action_data.get("args", {}).get("summary", "Research completed")
                return self._update_state(
                    state,
                    message=AIMessage(content=response.content),
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


