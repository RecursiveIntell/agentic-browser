"""
Code Agent for code analysis and execution.

Specializes in understanding codebases and running code.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class CodeAgentNode(BaseAgent):
    """Specialized agent for code analysis tasks.
    
    Analyzes codebases, reads documentation, and can execute
    sandboxed code for testing.
    """
    
    AGENT_NAME = "code"
    MAX_STEPS_PER_INVOCATION = 8
    
    SYSTEM_PROMPT = """You are a specialized CODE agent. Your job is to analyze, understand, and work with code.

Available actions:
- os_list_dir: { "path": "/path/to/project" }
- os_read_file: { "path": "/path/to/file" }
- os_exec: { "cmd": "command", "cwd": "/project/dir" }
- done: { "summary": "your analysis/findings" }

WORKFLOW FOR CODE ANALYSIS:
1. List the project directory to understand structure
2. Read README.md, package.json, pyproject.toml, or main entry files
3. Read key source files to understand the codebase
4. Provide summary of what the project does, tech stack, structure

SAFE COMMANDS:
- ls, cat, head, tail, grep, find, wc
- python --version, node --version, pip list
- Running tests: pytest, npm test (read-only)

AVOID (require approval):
- Any write operations
- Installing packages
- Running arbitrary scripts

FUZZY MATCHING:
- User's description may not match exact names
- "cat app" could be: CatOS, cat-tracker, feline-*, meow-*, etc.
- Look at README/docs to understand what a project actually does

Respond with JSON:
{
  "action": "os_list_dir|os_read_file|os_exec|done",
  "args": { ... },
  "rationale": "brief reason",
  "risk": "low|medium|high"
}"""

    def __init__(self, config, os_tools=None):
        """Initialize code agent.
        
        Args:
            config: Agent configuration
            os_tools: OS tools for file access
        """
        super().__init__(config)
        self._os_tools = os_tools
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_os_tools(self, os_tools) -> None:
        """Set OS tools after initialization."""
        self._os_tools = os_tools
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute code analysis.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with code analysis results
        """
        if not self._os_tools:
            return self._update_state(
                state,
                error="OS tools not available for code analysis",
            )
        
        task_context = f"""
CODE TASK: {state['goal']}

Files examined:
{chr(10).join(state['files_accessed'][-10:]) or '(none yet)'}

Information gathered:
{json.dumps(state['extracted_data'], indent=2)[:1500]}

TIPS:
- Start with os_list_dir to see project structure
- Read README.md or main config files first
- Look for entry points (main.py, index.js, etc.)
- After understanding the project, call "done" with your analysis
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
                    final_answer=action_data.get("args", {}).get("summary", "Analysis completed"),
                )
            
            # Execute OS action
            result = self._os_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            file_accessed = None
            if action_data.get("action") in ("os_read_file", "os_list_dir"):
                file_accessed = action_data.get("args", {}).get("path")
            
            extracted = None
            if result.success:
                action_name = action_data.get("action", "unknown")
                key = f"code_{action_name}_{len(state['extracted_data'])}"
                data = result.data if result.data else result.message
                extracted = {key: str(data)[:800]}
            
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
                error=f"Code agent error: {str(e)}",
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


def code_agent_node(state: AgentState, config: dict) -> AgentState:
    """LangGraph node function for code agent."""
    agent_config = config.get("agent_config")
    os_tools = config.get("os_tools")
    
    agent = CodeAgentNode(agent_config, os_tools)
    return agent.execute(state)
