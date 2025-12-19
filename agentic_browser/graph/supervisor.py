"""
Supervisor agent that orchestrates specialized agents.

Routes tasks to appropriate agents and synthesizes results.
"""

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import AgentState
from ..domain_router import DomainRouter
from ..config import AgentConfig

logger = logging.getLogger("agentic_browser.supervisor")


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
- data: Convert file formats (JSON/CSV/YAML), text search, compress/extract
- network: Network diagnostics (ping, DNS, HTTP requests, SSL checks)
- sysadmin: System monitoring (services, processes, disk/memory usage, logs)
- media: Video/audio conversion (ffmpeg), image resize/compress
- package: Python venv, pip install, git clone, project setup
- automation: Desktop notifications, scheduling, reminders, timers

ROUTING STRATEGY:
Analyze the goal and route to the most appropriate agent:
- Local file operations → os
- Code analysis, "what does X do?" → code
- "Find similar", "research", "search online" → research
- Navigate to specific URL → browser
- Convert JSON to CSV, compress files → data
- Ping, check ports, test API → network
- Check service status, memory usage → sysadmin
- Convert video, resize image → media
- Create venv, install packages → package
- Remind me, notify, schedule, timer → automation

MULTI-STEP TASKS:
Some goals require multiple agents:
- "Analyze X and find similar online" → code first, then research
- "Find X in files and summarize" → os first, then code

INTELLIGENT COMPLETION:
- Check extracted_data to see what's been gathered
- If agents are cycling (same agent 3+ times without new data), move on
- If you have enough information, synthesize a report and complete
- A good partial answer is better than infinite loops!
- SATISFICING: If you have 3-4 solid facts/sources, that is ENOUGH. Do not aim for perfection.
- If total step count > 15, aggressively wrap up and complete.

Respond with JSON:
{
  "route_to": "os|code|research|browser|data|network|sysadmin|media|package|automation|done",
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
        
        # Use factory function to create provider-appropriate LLM client
        from .agents.base import create_llm_client
        self.llm = create_llm_client(config, max_tokens=500)
        self.domain_router = DomainRouter(config)
    
    def safe_invoke(self, messages: list) -> AIMessage:
        """Invoke LLM with fallback handling for 404 errors and empty responses.
        
        If the configured model is not found (404), this will:
        1. Log the error
        2. Switch to a fallback model (e.g. claude-3-haiku)
        3. Re-initialize the LLM client
        4. Retry the invocation
        """
        try:
            response = self.llm.invoke(messages)
            
            # Handle empty responses
            if response is None or (hasattr(response, 'content') and not response.content):
                print("[WARN] Supervisor LLM returned empty response, providing fallback")
                return AIMessage(content='{"route_to": "done", "rationale": "Model returned empty response", "final_answer": "Unable to complete - model returned empty response"}')
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle empty response errors from provider
            if "empty" in error_msg or "must contain" in error_msg:
                print(f"[WARN] Empty response error: {e}")
                return AIMessage(content='{"route_to": "done", "rationale": "Model error", "final_answer": "Model returned empty response - please try again"}')
            
            # Check for 404 / model not found errors
            if "404" in error_msg or "not_found" in error_msg or "model" in error_msg and "not found" in error_msg:
                print(f"[WARN] detailed error: {str(e)}")
                print(f"[WARN] Model {self.config.model} not found. Attempting fallback...")
                
                # Determine fallback model based on current provider/model
                fallback_model = None
                
                if "claude" in (self.config.model or ""):
                    # Fallback chain for Anthropic
                    if "sonnet-20241022" in self.config.model:
                        fallback_model = "claude-3-5-sonnet-20240620"
                    elif "sonnet" in self.config.model:
                        fallback_model = "claude-3-haiku-20240307"
                    else:
                        fallback_model = "claude-3-haiku-20240307"
                elif "gpt" in (self.config.model or ""):
                    # Fallback for OpenAI
                    fallback_model = "gpt-4o-mini"
                elif "gemini" in (self.config.model or ""):
                    # Fallback for Google
                    fallback_model = "gemini-1.5-flash"
                
                if fallback_model and fallback_model != self.config.model:
                    print(f"[INFO] Switching to fallback model: {fallback_model}")
                    
                    # Update config and re-initialize LLM
                    from .agents.base import create_llm_client
                    self.config.model = fallback_model
                    self.llm = create_llm_client(self.config, max_tokens=500)
                    
                    # Retry
                    try:
                        return self.llm.invoke(messages)
                    except Exception as retry_err:
                        print(f"[WARN] Retry also failed: {retry_err}")
                        return AIMessage(content='{"route_to": "done", "final_answer": "Model error during retry - please try again"}')
            
            # Catch-all: return a fallback response instead of crashing
            print(f"[WARN] Unhandled LLM error, returning fallback: {e}")
            return AIMessage(content='{"route_to": "done", "rationale": "LLM error", "final_answer": "An error occurred with the model. Please try again."}')
    
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
            response = self.safe_invoke(messages)
            
            # DEBUG: Print Supervisor decision
            print(f"\n{'='*60}")
            print(f"[DEBUG] Supervisor - Raw LLM Response:")
            print(f"{'='*60}")
            logger.debug(response.content[:1000] if len(response.content) > 1000 else response.content)
            print(f"{'='*60}\n")
            
            decision = self._parse_decision(response.content, state.get("current_domain", "research"))
            
            # DEBUG: Print parsed decision
            print(f"[DEBUG] Supervisor - Parsed Decision: {decision}")
            print(f"[DEBUG] Supervisor - Current step: {state['step_count']}, extracted_data keys: {list(state['extracted_data'].keys())}")
            
            # HARD ENFORCEMENT: Force completion if criteria met
            research_sources = len([k for k in state['extracted_data'].keys() if 'research_source' in k])
            
            # Calculate percentage-based thresholds
            max_steps = state['max_steps']
            soft_limit = int(max_steps * 0.5)  # 50% of max_steps
            hard_limit = int(max_steps * 0.7)  # 70% of max_steps
            
            if state['step_count'] >= soft_limit and research_sources >= 3:
                print(f"[ENFORCE] Soft limit ({soft_limit}+ steps, 50% of {max_steps}) AND sufficient data ({research_sources} sources). Forcing completion.")
                decision = {
                    "route_to": "done",
                    "final_answer": self._synthesize_report(state)
                }
            elif state['step_count'] >= hard_limit:
                print(f"[ENFORCE] Hard limit ({hard_limit}+ steps, 70% of {max_steps}). Forcing completion regardless of data.")
                # Return immediately - don't let the extracted_data check override this
                return {
                    **state,
                    "current_domain": "done",
                    "task_complete": True,
                    "final_answer": self._synthesize_report(state),
                    "messages": [AIMessage(content=response.content)],
                }
            
            # HARD BLOCK: Minimum steps before allowing completion
            MIN_STEPS_BEFORE_DONE = 5
            
            if decision.get("route_to") == "done":
                goal_lower = state['goal'].lower()
                
                # Check if this is a research task
                is_research_goal = any(kw in goal_lower for kw in [
                    'look up', 'research', 'find out', 'search', 'what is', 
                    'who is', 'give me', 'report', 'summarize', 'website', 'websites'
                ])
                
                # Count research sources
                research_sources = len([k for k in state['extracted_data'].keys() if 'research_source' in k])
                
                # Check if user specified minimum sources (e.g., "at least 3 websites")
                import re
                source_match = re.search(r'at least (\d+)', goal_lower) or re.search(r'(\d+)\s*(?:websites?|sources?|pages?)', goal_lower)
                min_required_sources = int(source_match.group(1)) if source_match else 3
                
                # For research tasks, require minimum sources
                if is_research_goal and research_sources < min_required_sources:
                    print(f"[DEBUG] Supervisor - BLOCKED 'done': research task has only {research_sources}/{min_required_sources} sources")
                    decision = {"route_to": "research", "rationale": f"Need {min_required_sources - research_sources} more sources before completing"}
                
                elif state["step_count"] < MIN_STEPS_BEFORE_DONE:
                    print(f"[DEBUG] Supervisor - BLOCKED 'done': step_count ({state['step_count']}) < MIN_STEPS ({MIN_STEPS_BEFORE_DONE})")
                    # Determine which agent to use based on goal
                    fallback_agent = "research" if is_research_goal else "code"
                    decision = {"route_to": fallback_agent, "rationale": "Need more exploration before completing"}
                    
                elif not state['extracted_data'] or len(state['extracted_data']) == 0:
                    print(f"[DEBUG] Supervisor - BLOCKED 'done': No extracted_data yet")
                    # Same logic - use research for research goals
                    fallback_agent = "research" if is_research_goal else "code"
                    decision = {"route_to": fallback_agent, "rationale": "No data collected yet"}
            
            if decision.get("route_to") == "done":
                return {
                    **state,
                    "current_domain": "done",
                    "task_complete": True,
                    "final_answer": decision.get("final_answer", "Task completed"),
                    "messages": [AIMessage(content=response.content)],
                }
            
            # Success - reset consecutive errors
            state["consecutive_errors"] = 0
            
            return {
                **state,
                "current_domain": decision.get("route_to", state["current_domain"]),
                "active_agent": "supervisor",
                "step_count": state["step_count"] + 1,
                "messages": [AIMessage(content=response.content)],
                "consecutive_errors": 0,
            }
            
        except Exception as e:
            print(f"[DEBUG] Supervisor - ERROR: {e}")
            
            # Increment error count
            errors = state.get("consecutive_errors", 0) + 1
            
            if errors >= 3:
                print(f"[ERROR] Supervisor - Too many consecutive errors ({errors}). Aborting.")
                return {
                    **state,
                    "task_complete": True,
                    "final_answer": f"Task failed due to persistent errors: {str(e)}",
                    "error": str(e),
                    "consecutive_errors": errors,
                }
            
            # On error, try to continue with current domain
            return {
                **state,
                "error": f"Supervisor error: {str(e)}",
                "step_count": state["step_count"] + 1,
                "consecutive_errors": errors,
            }
    
    def _refine_initial_domain(self, domain: str, goal: str) -> str:
        """Refine the rough domain decision into a specific agent."""
        goal_lower = goal.lower()
        
        # Check for new specialized agent keywords first
        # Network agent keywords
        network_keywords = ["ping", "dns", "port", "http", "api", "ssl", "certificate", "network", "traceroute"]
        if any(k in goal_lower for k in network_keywords):
            return "network"
        
        # SysAdmin agent keywords
        sysadmin_keywords = ["service", "systemctl", "process", "memory", "disk usage", "cpu", "journal", "logs", "uptime"]
        if any(k in goal_lower for k in sysadmin_keywords):
            return "sysadmin"
        
        # Media agent keywords
        media_keywords = ["video", "audio", "ffmpeg", "convert mp", "resize image", "compress image", "extract audio"]
        if any(k in goal_lower for k in media_keywords):
            return "media"
        
        # Data agent keywords
        data_keywords = ["json", "csv", "yaml", "convert", "compress", "extract", "tar", "zip", "text search"]
        if any(k in goal_lower for k in data_keywords):
            return "data"
        
        # Package agent keywords
        package_keywords = ["pip", "venv", "virtual environment", "npm", "install package", "git clone", "requirements"]
        if any(k in goal_lower for k in package_keywords):
            return "package"
        
        # Automation agent keywords
        automation_keywords = ["remind", "notify", "notification", "schedule", "timer", "countdown", "cron", "at "]
        if any(k in goal_lower for k in automation_keywords):
            return "automation"
        
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
    
    def _parse_decision(self, response: str, current_domain: str = "research") -> dict:
        """Parse supervisor decision from response.
        
        Args:
            response: LLM response to parse
            current_domain: Current active domain to fallback to on parse failure
        """
        try:
            content = response.strip()
            if not content:
                logger.debug(f"Supervisor - Empty response, continuing with {current_domain} agent")
                return {"route_to": current_domain, "rationale": "Empty LLM response, continuing exploration"}
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            return json.loads(content)
        except json.JSONDecodeError:
            # On parse failure, continue with current agent instead of switching to code
            logger.debug(f"Supervisor - JSON parse failed, continuing with {current_domain} agent")
            return {"route_to": current_domain, "rationale": "Failed to parse response"}
    
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
    
    def _synthesize_report(self, state: AgentState) -> str:
        """Synthesize all collected data into a report."""
        data = state.get("extracted_data", {})
        
        if not data:
            return "No data collected during the task."
        
        def clean_content(text: str, max_len: int = 500) -> str:
            """Clean up webpage content for display."""
            text = str(text)
            # Remove DuckDuckGo and search engine noise
            noise_patterns = [
                "Main menu", "Jump to content", "Toggle the table",
                "Contents hide", "Create account", "Log in", "Personal tools",
                "All Images Videos News Maps Shopping",
                "DuckDuckGo Protection. Privacy. Peace of mind.",
                "DuckDuckGo never tracks your searches",
                "Search Assist Duck.ai Search Settings",
                "Protected DuckDuckGo protects", "Learn More You can hide this",
                "reminder in Search Settings All regions",
                "Open menu All Images Videos News Maps Shopping Search Assist",
                "Never tracks your searches", "Search Settings", "All regions",
                "Argentina Australia", "Belgium Brazil Bulgaria Canada",
                "Chile China Colombia", "France Germany Greece",
                "Media subscription Light novel Volumes Manga Volumes Anime Episodes",
                "References External links", "Categories:",
                "This Thursday , we ask you to join the 2%",
                "gave just $2.75", "donate", "Donate",
                "Wikipedia", "From Wikipedia, the free encyclopedia",
                "Sorry to interrupt, but we're short on time",
            ]
            for noise in noise_patterns:
                text = text.replace(noise, " ")
            # Clean up whitespace
            text = ' '.join(text.split())
            return text[:max_len]
        
        # Categorize data by source type
        research_data = {k: v for k, v in data.items() if 'research' in k.lower()}
        code_data = {k: v for k, v in data.items() if 'code' in k.lower()}
        network_data = {k: v for k, v in data.items() if 'network' in k.lower()}
        sysadmin_data = {k: v for k, v in data.items() if 'sysadmin' in k.lower()}
        other_data = {k: v for k, v in data.items() 
                      if not any(x in k.lower() for x in ['research', 'code', 'network', 'sysadmin'])}
        
        # Build report
        report = f"## Report: {state['goal']}\n\n"
        report += f"**Total items collected:** {len(data)}\n\n"
        
        # Include network diagnostics if present
        if network_data:
            report += f"### Network Results\n\n"
            for key, content in network_data.items():
                if isinstance(content, dict):
                    report += f"- **{key}**: {json.dumps(content)[:300]}\n"
                else:
                    report += f"- **{key}**: {clean_content(str(content), 300)}\n"
            report += "\n"
        
        # Include sysadmin data if present
        if sysadmin_data:
            report += f"### System Information\n\n"
            for key, content in sysadmin_data.items():
                report += f"- **{key}**: {clean_content(str(content), 300)}\n"
            report += "\n"
        
        # Include code analysis if present
        if 'code_analysis' in data:
            report += f"### Analysis Summary\n\n{data['code_analysis'][:1500]}\n\n"
        
        # Include research findings if present  
        if research_data:
            report += f"### Research Findings ({len(research_data)} sources)\n\n"
            for i, (key, content) in enumerate(list(research_data.items())[:3], 1):
                cleaned = clean_content(content, 600)
                report += f"**Source {i}:** {cleaned}\n\n"
        
        # Include code findings if present (exclude code_analysis already shown)
        code_findings = {k: v for k, v in code_data.items() if k != 'code_analysis'}
        if code_findings:
            report += f"### Code Exploration ({len(code_findings)} files examined)\n\n"
            for i, (key, content) in enumerate(list(code_findings.items())[:3], 1):
                content_preview = clean_content(str(content), 300)
                report += f"- {key}: {content_preview}...\n"
            report += "\n"
        
        # Include other data
        if other_data:
            report += f"### Additional Findings\n\n"
            for key, content in list(other_data.items())[:3]:
                report += f"- **{key}**: {clean_content(str(content), 200)}\n"
            report += "\n"
        
        return report


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


def route_to_agent(state: AgentState) -> Literal[
    "browser", "os", "research", "code", "sysadmin", 
    "network", "media", "package", "automation", "data", "__end__"
]:
    """Conditional edge function for routing.
    
    Returns the name of the next node based on current_domain.
    """
    domain = state.get("current_domain", "")
    
    if domain == "done" or state.get("task_complete"):
        return "__end__"
    
    # All valid agent routes
    valid_agents = {
        "browser", "os", "research", "code", "sysadmin",
        "network", "media", "package", "automation", "data"
    }
    
    if domain in valid_agents:
        return domain
    
    # Determine smart default based on goal
    goal = state.get("goal", "").lower()
    
    # OS-oriented keywords → don't open browser
    os_keywords = ["file", "memory", "disk", "cpu", "process", "service", "list", "run"]
    if any(kw in goal for kw in os_keywords):
        return "os"
    
    # Web-oriented → browser
    return "browser"
