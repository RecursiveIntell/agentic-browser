"""
OS Agent for local filesystem and shell operations.

Wraps existing OSTools for LangGraph integration.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState
from ...os_tools import OSTools
from ...llm_client import LLMClient


class OSAgentNode(BaseAgent):
    """Specialized agent for OS/filesystem tasks.
    
    Uses OSTools for shell commands, file operations,
    and system inspection.
    """
    
    AGENT_NAME = "os"
    MAX_STEPS_PER_INVOCATION = 5
    
    SYSTEM_PROMPT = """You are a specialized LINUX OS agent. Your ONLY job is local system operations.

You have access to these OS actions:
- os_exec: { "cmd": "command string", "timeout_s": 30 }
- os_list_dir: { "path": "/path/to/dir" }
- os_read_file: { "path": "/path/to/file" }
- os_write_file: { "path": "/path/to/file", "content": "...", "mode": "overwrite|append" }
- done: { "summary": "what you found/accomplished" }

CRITICAL RULES:
1. Use 3-5 commands max, then call "done" with your findings
2. CASE SENSITIVITY: "Coding" = "coding" = "CODING" - match by intent!
3. FUZZY MATCHING: "cat app" could be CatOS, cat-tracker, kitty-project, etc.
4. EXPLORE: If user says "find X in Y", actually go into Y directory!
5. Don't give up easily - look inside directories before saying "not found"

Safe commands (just do them): ls, cat, grep, find, df, du, ps, head, tail
Risky commands (require approval): rm, sudo, chmod, chown

Respond with JSON:
{
  "action": "os_exec|os_list_dir|os_read_file|os_write_file|done",
  "args": { ... },
  "rationale": "brief reason",
  "risk": "low|medium|high"
}"""

    def __init__(self, config, os_tools: OSTools | None = None):
        """Initialize OS agent.
        
        Args:
            config: Agent configuration
            os_tools: Optional pre-initialized OSTools
        """
        super().__init__(config)
        self._os_tools = os_tools
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_os_tools(self, os_tools: OSTools) -> None:
        """Set OS tools after initialization."""
        self._os_tools = os_tools
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute OS operations.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with OS action results
        """
        if not self._os_tools:
            return self._update_state(
                state,
                error="OS tools not initialized",
            )
        
        # Build context with error history
        last_error = state.get('error', '')
        
        task_context = f"""
Your task: {state['goal']}

USER HOME DIRECTORY: /home/sikmindz
COMMON DIRECTORIES: ~/Coding, ~/Documents, ~/Downloads

IMPORTANT - Previous error (if any): {last_error or '(none)'}

Files already accessed:
{chr(10).join(f'- {f}' for f in state['files_accessed'][-10:]) or '(none yet)'}

Data collected so far:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

EXPLORATION STRATEGY:
1. If you get "Path does not exist", use ABSOLUTE paths starting with /home/sikmindz/
2. Start by listing /home/sikmindz/Coding to find projects
3. Use "os_exec" with "find /home/sikmindz/Coding -name '*cat*' -type d" to search
4. DON'T repeat the same failed path - try a different approach!
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
            
            # Execute the OS action
            result = self._os_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            # Track file access
            file_accessed = None
            if action_data.get("action") in ("os_read_file", "os_list_dir"):
                file_accessed = action_data.get("args", {}).get("path")
            
            # Store results
            extracted = None
            if result.success and result.data:
                key = f"os_{action_data.get('action')}_{len(state['extracted_data'])}"
                extracted = {key: str(result.data)[:500]}
            
            return self._update_state(
                state,
                message=AIMessage(content=response.content),
                file_accessed=file_accessed,
                extracted_data=extracted,
                error=result.message if not result.success else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"OS agent error: {str(e)}",
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


def os_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for OS agent."""
    agent_config = state.get("_config")
    os_tools = state.get("_os_tools")
    
    if not agent_config:
        from ...config import AgentConfig
        agent_config = AgentConfig()
    
    agent = OSAgentNode(agent_config, os_tools)
    return agent.execute(state)

