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
    MAX_STEPS_PER_INVOCATION = 10
    
    SYSTEM_PROMPT = """You are a CODE agent. Analyze projects at the appropriate depth.

Available actions:
- os_list_dir: { "path": "." } - List current directory
- os_read_file: { "path": "README.md" } - Read a file
- os_exec: { "cmd": "command" } - Run a shell command
- done: { "summary": "your analysis" } - Complete with findings

ADAPTIVE WORKFLOW:
For SIMPLE tasks ("what is this app?"):
  1. List current directory: {"action": "os_list_dir", "args": {"path": "."}}
  2. Read README: {"action": "os_read_file", "args": {"path": "README.md"}}
  3. Done with summary

For COMPLEX tasks ("analyze architecture"):
  1. List structure (.)
  2. Read config files
  3. Read source
  4. Done

CRITICAL RULES:
1. ALWAYS start with: { "action": "os_list_dir", "args": { "path": "." } }
2. PROJECT VALIDATION:
   - Check if the files in "." match the user's request.
   - If User asks for "Cat App" but you see "Agentic Browser" code -> WRONG DIRECTORY.
   - ACTION: Search nearby: 
     { "action": "os_exec", "args": { "cmd": "find .. -name '*cat*' -type d -maxdepth 3" } }
   - Then listing the CORRECT directory.
3. Do not guess paths! Use "os_exec find ..." if you are lost.
4. "os_list_dir" output is in your HISTORY. Do not re-list same dir repeatedly.
5. Store findings in "code_analysis" key of extracted_data.

FUZZY MATCHING:
- "cat app" could match: CatOS, Cat Info App, cat-tracker, etc.
- Check folder names and READMEs to identify the right project

COMPLETION RULES:
- Always provide a USEFUL summary when calling "done"
- Include: what the project does, tech stack, key features
- If you can't find something, say so and complete with what you have

Respond with JSON:
{
  "action": "os_list_dir|os_read_file|os_exec|done",
  "args": { ... },
  "rationale": "brief reason"
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
                # Store findings but DON'T mark task_complete - let supervisor decide
                summary = action_data.get("args", {}).get("summary", "Analysis completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"code_analysis": summary},
                    # Note: NOT setting task_complete - supervisor handles that
                )
            
            # Execute OS action
            result = self._os_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            file_accessed = None
            if action_data.get("action") in ("os_read_file", "os_list_dir"):
                file_accessed = action_data.get("args", {}).get("path")
            
            # Use tool output as a message, not extracted_data
            action_name = action_data.get("action", "unknown")
            output_content = str(result.data if result.data else result.message)
            tool_msg = HumanMessage(content=f"Tool '{action_name}' output:\n{output_content[:2000]}")
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                file_accessed=file_accessed,
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
            
            data = json.loads(content)
            
            # Validate action exists
            if not data.get("action"):
                return {
                    "action": "done", 
                    "args": {"summary": "Agent error: Failed to generate valid action"}
                }
                
            return data
        except json.JSONDecodeError:
            return {"action": "done", "args": {"summary": "Failed to parse JSON response"}}


def code_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for code agent."""
    from ..tool_registry import get_tools
    
    # Get tools from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
        os_tools = tools.os_tools
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig()
        os_tools = None
    
    agent = CodeAgentNode(agent_config, os_tools)
    return agent.execute(state)


