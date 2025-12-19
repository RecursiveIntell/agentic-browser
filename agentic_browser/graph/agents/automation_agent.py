"""
Automation & Workflow Agent for scheduling and notifications.

Handles task scheduling, desktop notifications, and simple workflows.
"""

import json
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class AutomationAgentNode(BaseAgent):
    """Specialized agent for automation and scheduling tasks.
    
    Handles desktop notifications, simple scheduling,
    and basic workflow coordination.
    """
    
    AGENT_NAME = "automation"
    MAX_STEPS_PER_INVOCATION = 5
    
    SYSTEM_PROMPT = """You are an AUTOMATION agent. You help with scheduling and notifications.

Available actions:
- notify_desktop: { "title": "Reminder", "body": "Check backup status" }
- delay: { "seconds": 5 }  # Wait before next action
- at_schedule: { "time": "14:00", "command": "notify-send 'Meeting!'" }  # One-time
- cron_show: {}  # Show current crontab
- reminder_list: {}  # List pending reminders
- countdown: { "seconds": 30, "message": "Timer done!" }  # Countdown notification
- done: { "summary": "what was set up" }

NOTES:
1. Desktop notifications work immediately
2. 'at' scheduling requires the 'at' daemon (may need sudo)
3. Cron jobs are user-level only
4. This agent is for simple automations - complex workflows should be scripts

Respond with JSON:
{
  "action": "notify_desktop|at_schedule|...|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config):
        """Initialize automation agent."""
        super().__init__(config)
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute automation task."""
        task_context = f"""
AUTOMATION TASK: {state['goal']}

Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            if action == "done":
                summary = args.get("summary", "Automation task completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"automation_result": summary},
                )
            
            # Execute the automation action
            result = self._execute_action(action, args)
            
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"automation_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Automation agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute an automation action."""
        handlers = {
            "notify_desktop": self._notify_desktop,
            "delay": self._delay,
            "at_schedule": self._at_schedule,
            "cron_show": self._cron_show,
            "reminder_list": self._reminder_list,
            "countdown": self._countdown,
        }
        
        handler = handlers.get(action)
        if not handler:
            return {"success": False, "message": f"Unknown action: {action}"}
        
        try:
            return handler(args)
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _notify_desktop(self, args: dict) -> dict:
        """Send a desktop notification."""
        title = args.get("title", "Agent Notification")
        body = args.get("body", "")
        urgency = args.get("urgency", "normal")  # low, normal, critical
        
        # Try notify-send (Linux)
        cmd = ["notify-send", "-u", urgency, title, body]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return {"success": False, "message": f"Notification failed: {result.stderr}"}
        
        return {
            "success": True,
            "message": f"Notification sent: {title}",
            "data": {"title": title, "body": body}
        }
    
    def _delay(self, args: dict) -> dict:
        """Delay execution."""
        import time
        seconds = min(args.get("seconds", 1), 30)  # Max 30 seconds
        
        time.sleep(seconds)
        
        return {
            "success": True,
            "message": f"Waited {seconds} seconds",
            "data": {"seconds": seconds}
        }
    
    def _at_schedule(self, args: dict) -> dict:
        """Schedule a one-time command using 'at'.
        
        Uses stdin to pass the command safely to 'at', avoiding shell=True.
        """
        time_str = args.get("time", "")
        command = args.get("command", "")
        
        if not time_str or not command:
            return {"success": False, "message": "Both time and command are required"}
        
        # Validate time_str to prevent injection (basic alphanumeric + colon/space)
        import re
        if not re.match(r'^[\w\s:+-]+$', time_str):
            return {"success": False, "message": f"Invalid time format: {time_str}"}
        
        # Safe subprocess call: pass command via stdin, not shell
        try:
            result = subprocess.run(
                ["at", time_str],
                input=command + "\n",
                text=True,
                capture_output=True,
                timeout=10
            )
        except FileNotFoundError:
            return {
                "success": False,
                "message": "'at' command not found. Install it with: sudo apt install at"
            }
        
        if result.returncode != 0:
            return {
                "success": False, 
                "message": f"at scheduling failed: {result.stderr}. Make sure 'at' is installed and atd is running."
            }
        
        return {
            "success": True,
            "message": f"Scheduled '{command}' at {time_str}",
            "data": {"time": time_str, "command": command}
        }
    
    def _cron_show(self, args: dict) -> dict:
        """Show current user's crontab."""
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            if "no crontab" in result.stderr.lower():
                return {"success": True, "message": "No crontab for current user", "data": {}}
            return {"success": False, "message": f"crontab failed: {result.stderr}"}
        
        return {
            "success": True,
            "message": result.stdout or "No cron jobs",
            "data": {"crontab": result.stdout}
        }
    
    def _reminder_list(self, args: dict) -> dict:
        """List pending 'at' jobs."""
        result = subprocess.run(["atq"], capture_output=True, text=True, timeout=5)
        
        return {
            "success": True,
            "message": result.stdout or "No pending jobs",
            "data": {"jobs": result.stdout}
        }
    
    def _countdown(self, args: dict) -> dict:
        """Start a countdown and notify when done."""
        import time
        
        seconds = min(args.get("seconds", 30), 300)  # Max 5 minutes
        message = args.get("message", "Timer done!")
        
        time.sleep(seconds)
        
        # Send notification
        subprocess.run(
            ["notify-send", "-u", "critical", "Timer", message],
            capture_output=True,
            timeout=5
        )
        
        return {
            "success": True,
            "message": f"Countdown complete after {seconds}s: {message}",
            "data": {"seconds": seconds, "message": message}
        }
    
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
            return {"action": "done", "args": {"summary": "Unable to parse response"}}


def automation_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for automation agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = AutomationAgentNode(agent_config)
    return agent.execute(state)
