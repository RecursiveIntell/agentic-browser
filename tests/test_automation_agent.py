"""
Tests for the automation agent.

Verifies security and correctness of automation operations.
"""

import subprocess
import pytest
from unittest.mock import patch, MagicMock

from agentic_browser.graph.agents.automation_agent import AutomationAgentNode


class TestAutomationAgentSecurity:
    """Tests for security of automation agent operations."""
    
    def test_at_schedule_no_shell_true(self):
        """Verify _at_schedule does not use shell=True (security requirement)."""
        import inspect
        from agentic_browser.graph.agents import automation_agent
        
        # Read the source code of the module
        source = inspect.getsource(automation_agent)
        
        # Count occurrences of shell=True (should be 0)
        shell_true_count = source.count("shell=True")
        assert shell_true_count == 0, (
            f"Found {shell_true_count} occurrences of shell=True in automation_agent.py. "
            "This is a security risk - use subprocess.run with args list instead."
        )
    
    def test_at_schedule_uses_stdin(self):
        """Verify _at_schedule passes command via stdin, not command line."""
        from agentic_browser.config import AgentConfig
        
        config = AgentConfig(goal="test")
        agent = AutomationAgentNode(config)
        
        # Mock subprocess.run to capture the call
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            
            result = agent._at_schedule({
                "time": "14:00",
                "command": "echo 'Hello World'"
            })
            
            # Verify subprocess.run was called with args list, not shell command
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            
            # First positional arg should be ["at", "14:00"]
            assert call_args[0][0] == ["at", "14:00"], (
                f"Expected args list ['at', '14:00'], got {call_args[0][0]}"
            )
            
            # Should use input= parameter for the command
            assert "input" in call_args[1], "Expected 'input' keyword argument for stdin"
            assert "echo 'Hello World'" in call_args[1]["input"], (
                f"Command not found in stdin input: {call_args[1]['input']}"
            )
            
            # Should NOT have shell=True
            assert call_args[1].get("shell") is not True, "shell=True should not be used"
    
    def test_at_schedule_special_characters(self):
        """Verify special characters in commands are handled safely."""
        from agentic_browser.config import AgentConfig
        
        config = AgentConfig(goal="test")
        agent = AutomationAgentNode(config)
        
        # Test with potentially dangerous command
        dangerous_command = 'echo "test"; rm -rf /'
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            
            result = agent._at_schedule({
                "time": "14:00",
                "command": dangerous_command
            })
            
            # The command should be passed as-is to stdin (at will execute it)
            # This is still "dangerous" but it's not our job to block it here
            # The safety classifier and policy engine handle that
            call_args = mock_run.call_args
            assert dangerous_command in call_args[1]["input"]
            
            # Critically, shell=True was NOT used
            assert call_args[1].get("shell") is not True
    
    def test_at_schedule_invalid_time_format(self):
        """Verify invalid time formats are rejected."""
        from agentic_browser.config import AgentConfig
        
        config = AgentConfig(goal="test")
        agent = AutomationAgentNode(config)
        
        # Test with injection attempt in time field
        result = agent._at_schedule({
            "time": "14:00; rm -rf /",
            "command": "echo test"
        })
        
        # Should fail due to invalid time format
        assert result["success"] is False
        assert "Invalid time format" in result["message"]
    
    def test_at_schedule_missing_at_command(self):
        """Verify graceful handling when 'at' is not installed."""
        from agentic_browser.config import AgentConfig
        
        config = AgentConfig(goal="test")
        agent = AutomationAgentNode(config)
        
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("at not found")
            
            result = agent._at_schedule({
                "time": "14:00",
                "command": "echo test"
            })
            
            assert result["success"] is False
            assert "'at' command not found" in result["message"]


class TestAutomationAgentFunctionality:
    """Tests for basic automation agent functionality."""
    
    def test_notify_desktop_uses_args_list(self):
        """Verify notify_desktop uses subprocess with args list."""
        from agentic_browser.config import AgentConfig
        
        config = AgentConfig(goal="test")
        agent = AutomationAgentNode(config)
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            
            result = agent._notify_desktop({
                "title": "Test",
                "body": "Test body"
            })
            
            # Should use args list
            call_args = mock_run.call_args[0][0]
            assert isinstance(call_args, list)
            assert call_args[0] == "notify-send"
