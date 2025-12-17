"""
Tool registry for LangGraph serialization compatibility.

Stores tool instances outside of state to enable checkpointing.
"""

import threading
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class ToolSet:
    """Container for tools associated with a session."""
    config: Any
    browser_tools: Any = None
    os_tools: Any = None


class ToolRegistry:
    """Singleton registry for tool instances.
    
    Tools are registered by session_id and looked up by nodes.
    This allows state to store only the session_id (a string)
    which is serializable for checkpointing.
    
    Usage:
        # Register tools when starting a session
        registry = ToolRegistry.get_instance()
        registry.register("session-123", config, os_tools=my_os_tools)
        
        # Look up tools in node functions
        tools = registry.get("session-123")
        os_tools = tools.os_tools
    """
    
    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._tools: dict[str, ToolSet] = {}
        self._tools_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register(
        self,
        session_id: str,
        config: Any,
        browser_tools: Any = None,
        os_tools: Any = None,
    ) -> None:
        """Register tools for a session.
        
        Args:
            session_id: Unique session identifier
            config: AgentConfig instance
            browser_tools: Optional BrowserTools instance
            os_tools: Optional OSTools instance
        """
        with self._tools_lock:
            self._tools[session_id] = ToolSet(
                config=config,
                browser_tools=browser_tools,
                os_tools=os_tools,
            )
    
    def get(self, session_id: str) -> Optional[ToolSet]:
        """Get tools for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ToolSet or None if not found
        """
        with self._tools_lock:
            return self._tools.get(session_id)
    
    def update_browser_tools(self, session_id: str, browser_tools: Any) -> None:
        """Update browser tools for a session."""
        with self._tools_lock:
            if session_id in self._tools:
                self._tools[session_id].browser_tools = browser_tools
    
    def update_os_tools(self, session_id: str, os_tools: Any) -> None:
        """Update OS tools for a session."""
        with self._tools_lock:
            if session_id in self._tools:
                self._tools[session_id].os_tools = os_tools
    
    def unregister(self, session_id: str) -> None:
        """Remove tools for a session.
        
        Args:
            session_id: Session identifier
        """
        with self._tools_lock:
            self._tools.pop(session_id, None)
    
    def clear(self) -> None:
        """Clear all registered tools."""
        with self._tools_lock:
            self._tools.clear()


# Convenience function
def get_tools(session_id: str) -> Optional[ToolSet]:
    """Get tools for a session from the global registry."""
    return ToolRegistry.get_instance().get(session_id)
