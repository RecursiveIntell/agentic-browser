"""
Tests for OS tools functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from agentic_browser.os_tools import OSTools, ToolResult


class TestOSToolsExec:
    """Tests for os_exec action."""
    
    @pytest.fixture
    def tools(self, tmp_path):
        return OSTools(sandbox_dir=tmp_path)
    
    def test_exec_simple_command(self, tools):
        """Test executing a simple command."""
        result = tools.execute("os_exec", {"cmd": "echo hello"})
        
        assert result.success is True
        assert "hello" in result.data["stdout"]
    
    def test_exec_missing_cmd(self, tools):
        """Test error when cmd is missing."""
        result = tools.execute("os_exec", {})
        
        assert result.success is False
        assert "missing" in result.message.lower()
    
    def test_exec_command_not_found(self, tools):
        """Test error when command doesn't exist."""
        result = tools.execute("os_exec", {"cmd": "nonexistent_command_xyz"})
        
        assert result.success is False
        assert "not found" in result.message.lower()
    
    def test_exec_timeout(self, tools):
        """Test command timeout handling."""
        import subprocess
        from unittest.mock import patch
        
        with patch("agentic_browser.os_tools.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
            result = tools.execute("os_exec", {"cmd": "sleep 100", "timeout_s": 1})
        
        assert result.success is False
        assert "timed out" in result.message.lower() or "timeout" in result.message.lower()
    
    def test_exec_respects_max_timeout(self, tools):
        """Test that timeout is capped at MAX_TIMEOUT_S."""
        # Request a very long timeout
        result = tools.execute("os_exec", {"cmd": "echo test", "timeout_s": 9999})
        
        # Should still work (uses capped timeout internally)
        assert result.success is True
    
    def test_exec_custom_cwd(self, tools, tmp_path):
        """Test executing in custom working directory."""
        result = tools.execute("os_exec", {"cmd": "pwd", "cwd": str(tmp_path)})
        
        assert result.success is True
        assert str(tmp_path) in result.data["stdout"]
    
    def test_exec_nonexistent_cwd(self, tools):
        """Test error when cwd doesn't exist."""
        result = tools.execute("os_exec", {"cmd": "ls", "cwd": "/nonexistent/path"})
        
        assert result.success is False
        assert "does not exist" in result.message


class TestOSToolsListDir:
    """Tests for os_list_dir action."""
    
    @pytest.fixture
    def tools(self, tmp_path):
        return OSTools(sandbox_dir=tmp_path)
    
    def test_list_dir_basic(self, tools, tmp_path):
        """Test listing directory contents."""
        # Create some files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "subdir").mkdir()
        
        result = tools.execute("os_list_dir", {"path": str(tmp_path)})
        
        assert result.success is True
        assert result.data["count"] == 3
        
        names = [e["name"] for e in result.data["entries"]]
        assert "file1.txt" in names
        assert "file2.txt" in names
        assert "subdir" in names
    
    def test_list_dir_missing_path(self, tools):
        """Test error when path is missing."""
        result = tools.execute("os_list_dir", {})
        
        assert result.success is False
        assert "missing" in result.message.lower()
    
    def test_list_dir_nonexistent(self, tools):
        """Test error when path doesn't exist."""
        result = tools.execute("os_list_dir", {"path": "/nonexistent/path"})
        
        assert result.success is False
        assert "does not exist" in result.message
    
    def test_list_dir_not_a_directory(self, tools, tmp_path):
        """Test error when path is a file."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        
        result = tools.execute("os_list_dir", {"path": str(file_path)})
        
        assert result.success is False
        assert "not a directory" in result.message


class TestOSToolsReadFile:
    """Tests for os_read_file action."""
    
    @pytest.fixture
    def tools(self, tmp_path):
        return OSTools(sandbox_dir=tmp_path)
    
    def test_read_file_basic(self, tools, tmp_path):
        """Test reading a file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")
        
        result = tools.execute("os_read_file", {"path": str(file_path)})
        
        assert result.success is True
        assert "Hello, World!" in result.data["content"]
        assert result.data["truncated"] is False
    
    def test_read_file_missing_path(self, tools):
        """Test error when path is missing."""
        result = tools.execute("os_read_file", {})
        
        assert result.success is False
        assert "missing" in result.message.lower()
    
    def test_read_file_nonexistent(self, tools):
        """Test error when file doesn't exist."""
        result = tools.execute("os_read_file", {"path": "/nonexistent/file.txt"})
        
        assert result.success is False
        assert "does not exist" in result.message
    
    def test_read_file_truncation(self, tools, tmp_path):
        """Test file truncation for large files."""
        file_path = tmp_path / "large.txt"
        file_path.write_text("x" * 10000)
        
        result = tools.execute("os_read_file", {"path": str(file_path), "max_bytes": 100})
        
        assert result.success is True
        assert len(result.data["content"]) == 100
        assert result.data["truncated"] is True


class TestOSToolsWriteFile:
    """Tests for os_write_file action."""
    
    @pytest.fixture
    def tools(self, tmp_path):
        return OSTools(
            sandbox_dir=tmp_path,
            allow_outside_home=True,  # Allow writes for testing
        )
    
    def test_write_file_overwrite(self, tools, tmp_path):
        """Test writing a file (overwrite mode)."""
        file_path = tmp_path / "test.txt"
        
        result = tools.execute("os_write_file", {
            "path": str(file_path),
            "content": "Hello!",
            "mode": "overwrite",
        })
        
        assert result.success is True
        assert file_path.read_text() == "Hello!"
    
    def test_write_file_append(self, tools, tmp_path):
        """Test appending to a file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello")
        
        result = tools.execute("os_write_file", {
            "path": str(file_path),
            "content": " World!",
            "mode": "append",
        })
        
        assert result.success is True
        assert file_path.read_text() == "Hello World!"
    
    def test_write_file_missing_path(self, tools):
        """Test error when path is missing."""
        result = tools.execute("os_write_file", {"content": "test"})
        
        assert result.success is False
        assert "missing" in result.message.lower()
    
    def test_write_file_missing_content(self, tools, tmp_path):
        """Test error when content is missing."""
        result = tools.execute("os_write_file", {"path": str(tmp_path / "test.txt")})
        
        assert result.success is False
        assert "missing" in result.message.lower()
    
    def test_write_file_invalid_mode(self, tools, tmp_path):
        """Test error for invalid mode."""
        result = tools.execute("os_write_file", {
            "path": str(tmp_path / "test.txt"),
            "content": "test",
            "mode": "invalid",
        })
        
        assert result.success is False
        assert "invalid mode" in result.message.lower()
    
    def test_write_file_creates_parent_dirs(self, tools, tmp_path):
        """Test that parent directories are created."""
        file_path = tmp_path / "subdir" / "nested" / "test.txt"
        
        result = tools.execute("os_write_file", {
            "path": str(file_path),
            "content": "nested content",
        })
        
        assert result.success is True
        assert file_path.read_text() == "nested content"


class TestOSToolsPathRestrictions:
    """Tests for path-based safety restrictions."""
    
    def test_write_outside_home_blocked(self, tmp_path):
        """Test that writes outside home are blocked by default."""
        tools = OSTools(
            sandbox_dir=tmp_path,
            allow_outside_home=False,
        )
        
        result = tools.execute("os_write_file", {
            "path": "/tmp/blocked.txt",
            "content": "should fail",
        })
        
        assert result.success is False
        assert "outside home" in result.message.lower() or "blocked" in result.message.lower()
    
    def test_write_to_etc_blocked(self, tmp_path):
        """Test that writes to /etc are blocked."""
        tools = OSTools(sandbox_dir=tmp_path, allow_outside_home=True)
        
        result = tools.execute("os_write_file", {
            "path": "/etc/test.txt",
            "content": "should fail",
        })
        
        assert result.success is False
        assert "protected" in result.message.lower() or "blocked" in result.message.lower()


class TestOSToolsRiskClassification:
    """Tests for risk classification method."""
    
    @pytest.fixture
    def tools(self, tmp_path):
        return OSTools(sandbox_dir=tmp_path)
    
    def test_classify_rm_rf_high(self, tools):
        """Test rm -rf classified as high risk."""
        risk = tools.classify_risk("os_exec", {"cmd": "rm -rf /"})
        assert risk == "high"
    
    def test_classify_sudo_high(self, tools):
        """Test sudo classified as high risk."""
        risk = tools.classify_risk("os_exec", {"cmd": "sudo apt update"})
        assert risk == "high"
    
    def test_classify_mv_medium(self, tools):
        """Test mv classified as medium risk."""
        risk = tools.classify_risk("os_exec", {"cmd": "mv file1 file2"})
        assert risk == "medium"
    
    def test_classify_ls_low(self, tools):
        """Test ls classified as low risk."""
        risk = tools.classify_risk("os_exec", {"cmd": "ls -la"})
        assert risk == "low"
    
    def test_classify_write_file_medium(self, tools):
        """Test write_file classified as medium risk."""
        import os
        home = os.path.expanduser("~")
        risk = tools.classify_risk("os_write_file", {"path": f"{home}/test.txt"})
        assert risk == "medium"
    
    def test_classify_list_dir_low(self, tools):
        """Test list_dir is always low risk."""
        risk = tools.classify_risk("os_list_dir", {"path": "/etc"})
        assert risk == "low"


class TestUnknownAction:
    """Tests for unknown action handling."""
    
    @pytest.fixture
    def tools(self, tmp_path):
        return OSTools(sandbox_dir=tmp_path)
    
    def test_unknown_action_error(self, tools):
        """Test error for unknown action."""
        result = tools.execute("os_unknown", {})
        
        assert result.success is False
        assert "unknown" in result.message.lower()
