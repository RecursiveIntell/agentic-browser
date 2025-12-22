"""
OS Agent for local filesystem and shell operations.

Wraps existing OSTools for LangGraph integration.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState
from ...os_tools import OSTools
from ...tool_schemas import RunCommandRequest, ListDirRequest, ReadFileRequest, WriteFileRequest

logger = logging.getLogger("agentic_browser.os_agent")


class OSAgentNode(BaseAgent):
    """Specialized agent for OS/filesystem tasks.
    
    Uses OSTools for shell commands, file operations,
    and system inspection.
    """
    
    AGENT_NAME = "os"
    MAX_STEPS_PER_INVOCATION = 5
    
    SYSTEM_PROMPT = """You are a specialized LINUX OS agent. Your ONLY job is local system operations.

You have access to these OS actions:
- os_exec: { "argv": ["ls", "-la"], "timeout_s": 30 }  # PREFERRED: use argv list
- os_exec: { "cmd": "ls -la" }  # DEPRECATED: avoid shell strings
- os_list_dir: { "path": "." } - DEFAULT to current directory!
- os_read_file: { "path": "filename.txt" }
- os_write_file: { "path": "file.txt", "content": "..." }
- done: { "summary": "detailed findings" }

STARTING STRATEGY:
1. ALWAYS start by listing the current directory: {"action": "os_list_dir", "args": {"path": "."}}
2. Then drill down into relevant folders
3. Never guess absolute paths - check first!
4. Use 3-5 commands max, then call "done" with your findings
5. CASE SENSITIVITY: "Coding" = "coding" = "CODING" - match by intent!
6. FUZZY MATCHING: "cat app" could be CatOS, cat-tracker, kitty-project, etc.
7. EXPLORE: If user says "find X in Y", actually go into Y directory!
8. Don't give up easily - look inside directories before saying "not found"

Safe commands (just do them): ls, cat, grep, find, df, du, ps, head, tail, free, top
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
        
        # Build context with error history - use dynamic home path
        last_error = state.get('error', '')
        home_dir = str(Path.home())
        
        task_context = f"""
Your task: {state['goal']}

USER HOME DIRECTORY: {home_dir}
COMMON DIRECTORIES: ~/Coding, ~/Documents, ~/Downloads

IMPORTANT - Previous error (if any): {last_error or '(none)'}

Files already accessed:
{chr(10).join(f'- {f}' for f in state['files_accessed'][-10:]) or '(none yet)'}

Data collected so far:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

EXPLORATION STRATEGY:
1. If you get "Path does not exist", use ABSOLUTE paths starting with {home_dir}/
2. Start by listing {home_dir}/Coding to find projects (if it exists)
3. Use "os_exec" with argv format: {{"argv": ["find", "{home_dir}/Coding", "-name", "*cat*", "-type", "d"]}}
4. DON'T repeat the same failed path - try a different approach!
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            if action_data.get("action") == "done":
                # Store findings but DON'T mark task_complete - let supervisor decide
                summary = action_data.get("args", {}).get("summary", "Task completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"os_findings": summary},
                )
            
            # Execute the OS action using typed interface when possible
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            # === SAFETY CHECK: Block destructive commands without approval ===
            from ..safety import GraphSafetyChecker, RiskLevel
            safety_checker = GraphSafetyChecker()
            risk_level, reason, requires_approval = safety_checker.check_action(action, args, domain="os")
            
            # Check if action is blocked entirely (denylist)
            if safety_checker.classifier.is_blocked(action, args):
                return self._update_state(
                    state,
                    messages=[AIMessage(content=f"BLOCKED: {reason}")],
                    error=f"Action blocked by safety system: {reason}",
                )
            
            # If HIGH or MEDIUM risk, require approval
            if requires_approval:
                # Check if already approved
                action_key = f"{action}:{hash(str(args))}"
                if action_key not in state.get("approved_actions", []):
                    # Set pending approval and return - DO NOT EXECUTE
                    print(f"[OS-AGENT] ⚠️ Action requires approval: {action} (risk: {risk_level.value})")
                    print(f"[OS-AGENT] Reason: {reason}")
                    
                    return self._update_state(
                        state,
                        messages=[AIMessage(content=f"⚠️ Action requires approval: {action}\nRisk: {risk_level.value}\nReason: {reason}")],
                        pending_approval={
                            "action": action,
                            "args": args,
                            "risk_level": risk_level.value,
                            "reason": reason,
                        },
                    )
                else:
                    print(f"[OS-AGENT] ✅ Action already approved: {action}")
            
            # For os_exec, prefer typed execution with argv
            if action == "os_exec":
                # Convert cmd to argv if needed (backward compatibility)
                if "cmd" in args and "argv" not in args:
                    import shlex
                    try:
                        args["argv"] = shlex.split(args["cmd"])
                        del args["cmd"]
                    except ValueError:
                        pass  # Keep cmd if shlex fails
                
                # Use typed request if we have argv
                if "argv" in args:
                    try:
                        request = RunCommandRequest(
                            argv=args["argv"],
                            timeout_s=args.get("timeout_s", 30),
                            cwd=args.get("cwd"),
                        )
                        result = self._os_tools.execute_typed(request)
                    except Exception as e:
                        # Fallback to legacy
                        result = self._os_tools.execute(action, args)
                else:
                    result = self._os_tools.execute(action, args)
            else:
                result = self._os_tools.execute(action, args)
            
            # Track file access
            file_accessed = None
            if action_data.get("action") in ("os_read_file", "os_list_dir"):
                file_accessed = action_data.get("args", {}).get("path")
            
            # Use tool output as a message, not extracted_data
            action_name = action_data.get("action", "unknown")
            tool_content = str(result.data) if result.data else result.message
            tool_msg = HumanMessage(content=f"Tool '{action_name}' output:\n{str(tool_content)[:2000]}")
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                file_accessed=file_accessed,
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
            
            data = json.loads(content)
            
            # Validate action
            if not data.get("action"):
                # Default to exploration instead of done
                return {
                    "action": "os_list_dir",
                    "args": {"path": "."}
                }
                
            return data
        except json.JSONDecodeError:
            # On parse failure, explore instead of quitting
            return {"action": "os_list_dir", "args": {"path": "."}}


def os_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for OS agent."""
    from ..tool_registry import get_tools
    
    # Get tools from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
        os_tools = tools.os_tools
    else:
        # Fallback: create minimal config from state
        from ...config import AgentConfig
        goal = state.get("goal", "OS operation")
        agent_config = AgentConfig(
            goal=goal,
            profile="default",
            max_steps=state.get("max_steps", 30),
        )
        os_tools = OSTools(agent_config)
        logger.warning("OS agent created fallback config - tool registry returned None")
    
    agent = OSAgentNode(agent_config, os_tools)
    return agent.execute(state)


