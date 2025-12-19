"""
Tests for tool router functionality.
"""

import pytest
from unittest.mock import MagicMock

from agentic_browser.tool_router import ToolRouter


class TestToolRouterRouting:
    """Tests for action routing decisions."""
    
    @pytest.fixture
    def router(self):
        return ToolRouter()
    
    def test_route_browser_actions(self, router):
        """Test browser actions route to browser domain."""
        browser_actions = [
            "goto", "click", "type", "press", "scroll",
            "wait_for", "extract", "extract_visible_text",
            "screenshot", "back", "forward", "done",
            "download_file", "download_image",
        ]
        
        for action in browser_actions:
            domain = router.route_action(action)
            assert domain == "browser", f"Failed for: {action}"
    
    def test_route_os_actions(self, router):
        """Test OS actions route to OS domain."""
        os_actions = [
            "os_exec", "os_list_dir", "os_read_file", "os_write_file",
            "os_move_file", "os_copy_file", "os_delete_file",
        ]
        
        for action in os_actions:
            domain = router.route_action(action)
            assert domain == "os", f"Failed for: {action}"

    def test_route_memory_actions(self, router):
        """Test memory actions route to memory domain."""
        memory_actions = [
            "memory_get_site", "memory_save_site", "memory_get_directory",
        ]

        for action in memory_actions:
            domain = router.route_action(action)
            assert domain == "memory", f"Failed for: {action}"
    
    def test_unknown_action_raises(self, router):
        """Test that unknown actions raise ValueError."""
        with pytest.raises(ValueError, match="Unknown action"):
            router.route_action("unknown_action")


class TestToolRouterExecution:
    """Tests for action execution dispatch."""
    
    def test_execute_browser_action(self):
        """Test browser action dispatches to browser tools."""
        mock_browser_tools = MagicMock()
        mock_browser_tools.execute.return_value = MagicMock(
            success=True, message="Clicked element"
        )
        
        router = ToolRouter(browser_tools=mock_browser_tools)
        result = router.execute("click", {"selector": "#button"})
        
        mock_browser_tools.execute.assert_called_once_with(
            "click", {"selector": "#button"}
        )
        assert result.success is True
    
    def test_execute_os_action(self):
        """Test OS action dispatches to OS tools."""
        mock_os_tools = MagicMock()
        mock_os_tools.execute.return_value = MagicMock(
            success=True, message="Listed directory"
        )
        
        router = ToolRouter(os_tools=mock_os_tools)
        result = router.execute("os_list_dir", {"path": "/home"})
        
        mock_os_tools.execute.assert_called_once_with(
            "os_list_dir", {"path": "/home"}
        )
        assert result.success is True
    
    def test_execute_without_browser_tools(self):
        """Test error when browser tools not configured."""
        router = ToolRouter()  # No browser tools
        
        result = router.execute("click", {"selector": "#button"})
        
        assert result.success is False
        assert "not configured" in result.message.lower()
    
    def test_execute_without_os_tools(self):
        """Test error when OS tools not configured."""
        router = ToolRouter()  # No OS tools
        
        result = router.execute("os_exec", {"cmd": "ls"})
        
        assert result.success is False
        assert "not configured" in result.message.lower()


class TestToolRouterAvailability:
    """Tests for available actions reporting."""
    
    def test_get_available_actions_none(self):
        """Test no actions when tools not configured."""
        router = ToolRouter()
        actions = router.get_available_actions()
        
        assert actions["browser"] == []
        assert actions["os"] == []
        assert actions["memory"] == []
    
    def test_get_available_actions_browser_only(self):
        """Test browser actions when only browser configured."""
        router = ToolRouter(browser_tools=MagicMock())
        actions = router.get_available_actions()
        
        assert "click" in actions["browser"]
        assert "goto" in actions["browser"]
        assert actions["os"] == []
        assert actions["memory"] == []
    
    def test_get_available_actions_os_only(self):
        """Test OS actions when only OS configured."""
        router = ToolRouter(os_tools=MagicMock())
        actions = router.get_available_actions()
        
        assert actions["browser"] == []
        assert "os_exec" in actions["os"]
        assert "os_list_dir" in actions["os"]
        assert actions["memory"] == []

    def test_get_available_actions_memory_only(self):
        """Test memory actions when only memory configured."""
        router = ToolRouter(memory_tools=MagicMock())
        actions = router.get_available_actions()

        assert actions["browser"] == []
        assert actions["os"] == []
        assert "memory_get_site" in actions["memory"]
    
    def test_get_available_actions_both(self):
        """Test all actions when both configured."""
        router = ToolRouter(
            browser_tools=MagicMock(),
            os_tools=MagicMock(),
        )
        actions = router.get_available_actions()
        
        assert len(actions["browser"]) > 0
        assert len(actions["os"]) > 0
        assert len(actions["memory"]) == 0


class TestToolRouterSetters:
    """Tests for tool setter methods."""
    
    def test_set_browser_tools(self):
        """Test setting browser tools."""
        router = ToolRouter()
        assert not router.has_browser_tools()
        
        router.set_browser_tools(MagicMock())
        assert router.has_browser_tools()
    
    def test_set_os_tools(self):
        """Test setting OS tools."""
        router = ToolRouter()
        assert not router.has_os_tools()
        
        router.set_os_tools(MagicMock())
        assert router.has_os_tools()
