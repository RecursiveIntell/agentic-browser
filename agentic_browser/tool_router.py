"""
Tool router for Agentic Browser.

Routes actions to the appropriate tool class (BrowserTools or OSTools).
"""

from typing import Any, Optional

from .domain_router import DomainRouter


# Import ToolResult from tools module to maintain consistency
# (we use a duck-typed approach to avoid circular imports)


class ToolRouter:
    """Routes actions to Browser or OS tools.
    
    Provides a unified interface for executing actions regardless of domain.
    """
    
    def __init__(
        self,
        browser_tools: Optional[Any] = None,
        os_tools: Optional[Any] = None,
        memory_tools: Optional[Any] = None,
    ):
        """Initialize the tool router.
        
        Args:
            browser_tools: BrowserTools instance (optional)
            os_tools: OSTools instance (optional)
        """
        self.browser_tools = browser_tools
        self.os_tools = os_tools
        self.memory_tools = memory_tools
    
    def set_browser_tools(self, browser_tools: Any) -> None:
        """Set the browser tools instance.
        
        Args:
            browser_tools: BrowserTools instance
        """
        self.browser_tools = browser_tools
    
    def set_os_tools(self, os_tools: Any) -> None:
        """Set the OS tools instance.
        
        Args:
            os_tools: OSTools instance
        """
        self.os_tools = os_tools

    def set_memory_tools(self, memory_tools: Any) -> None:
        """Set the memory tools instance.

        Args:
            memory_tools: MemoryTools instance
        """
        self.memory_tools = memory_tools
    
    def route_action(self, action: str) -> str:
        """Determine which domain an action belongs to.
        
        Args:
            action: Action name
            
        Returns:
            "browser" or "os"
            
        Raises:
            ValueError: If action is unknown
        """
        if DomainRouter.is_browser_action(action):
            return "browser"
        elif DomainRouter.is_os_action(action):
            return "os"
        elif DomainRouter.is_memory_action(action):
            return "memory"
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def execute(self, action: str, args: dict[str, Any]) -> Any:
        """Execute an action via the appropriate tool class.
        
        Args:
            action: Action name
            args: Action arguments
            
        Returns:
            ToolResult from the appropriate tool class
            
        Raises:
            ValueError: If action is unknown or tools not configured
        """
        domain = self.route_action(action)
        
        if domain == "browser":
            if self.browser_tools is None:
                # Return error result in ToolResult-like format
                from dataclasses import dataclass
                
                @dataclass
                class ErrorResult:
                    success: bool = False
                    message: str = "Browser tools not configured"
                    data: Any = None
                    
                    def to_dict(self):
                        return {"success": self.success, "message": self.message}
                
                return ErrorResult()
            
            return self.browser_tools.execute(action, args)
        
        elif domain == "os":
            if self.os_tools is None:
                from dataclasses import dataclass
                
                @dataclass
                class ErrorResult:
                    success: bool = False
                    message: str = "OS tools not configured"
                    data: Any = None
                    
                    def to_dict(self):
                        return {"success": self.success, "message": self.message}
                
                return ErrorResult()
            
            return self.os_tools.execute(action, args)

        else:  # domain == "memory"
            if self.memory_tools is None:
                from dataclasses import dataclass

                @dataclass
                class ErrorResult:
                    success: bool = False
                    message: str = "Memory tools not configured"
                    data: Any = None

                    def to_dict(self):
                        return {"success": self.success, "message": self.message}

                return ErrorResult()

            return self.memory_tools.execute(action, args)
    
    def get_available_actions(self) -> dict[str, list[str]]:
        """Get lists of available actions by domain.
        
        Returns:
            Dict with "browser" and "os" keys containing action lists
        """
        browser_actions = []
        os_actions = []
        
        if self.browser_tools is not None:
            browser_actions = [
                "goto", "click", "type", "press", "scroll",
                "wait_for", "extract", "extract_visible_text",
                "screenshot", "back", "forward", "done",
                "download_file", "download_image",
            ]
        
        if self.os_tools is not None:
            os_actions = [
                "os_exec", "os_list_dir", "os_read_file", "os_write_file",
                "os_move_file", "os_copy_file", "os_delete_file",
            ]

        memory_actions = []
        if self.memory_tools is not None:
            memory_actions = [
                "memory_get_site", "memory_save_site", "memory_get_directory",
            ]
        
        return {
            "browser": browser_actions,
            "os": os_actions,
            "memory": memory_actions,
        }
    
    def has_browser_tools(self) -> bool:
        """Check if browser tools are configured."""
        return self.browser_tools is not None
    
    def has_os_tools(self) -> bool:
        """Check if OS tools are configured."""
        return self.os_tools is not None

    def has_memory_tools(self) -> bool:
        """Check if memory tools are configured."""
        return self.memory_tools is not None
