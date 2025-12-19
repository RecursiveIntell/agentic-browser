"""
Memory tools for Agentic Browser.

Provides a simple tool wrapper around StructuredMemoryStore.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .memory_store import StructuredMemoryStore, KnownSite


@dataclass
class ToolResult:
    """Result of a memory tool execution."""

    success: bool
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "success": self.success,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


class MemoryTools:
    """Executes memory actions via StructuredMemoryStore."""

    def __init__(self, store: StructuredMemoryStore):
        self.store = store

    def execute(self, action: str, args: dict[str, Any]) -> ToolResult:
        """Execute a memory action."""
        action_map = {
            "memory_get_site": self._get_site,
            "memory_save_site": self._save_site,
            "memory_get_directory": self._get_directory,
        }
        handler = action_map.get(action)
        if handler is None:
            return ToolResult(success=False, message=f"Unknown memory action: {action}")
        try:
            return handler(args)
        except Exception as e:
            return ToolResult(success=False, message=f"Error executing {action}: {e}")

    def _get_site(self, args: dict[str, Any]) -> ToolResult:
        domain = args.get("domain")
        if not domain:
            return ToolResult(success=False, message="Missing required argument: domain")
        site = self.store.get_site(domain)
        if site is None:
            return ToolResult(success=False, message=f"No site found for {domain}")
        return ToolResult(
            success=True,
            message=f"Loaded site for {domain}",
            data=site.model_dump(),
        )

    def _save_site(self, args: dict[str, Any]) -> ToolResult:
        domain = args.get("domain")
        data = args.get("data")
        if not domain or data is None:
            return ToolResult(
                success=False,
                message="Missing required arguments: domain and data",
            )

        if "domain" not in data:
            data = {**data, "domain": domain}

        try:
            site = KnownSite(**data)
        except Exception as e:
            return ToolResult(success=False, message=f"Invalid site data: {e}")

        self.store.save_site(site)
        return ToolResult(
            success=True,
            message=f"Saved site for {domain}",
            data={"domain": domain},
        )

    def _get_directory(self, args: dict[str, Any]) -> ToolResult:
        name = args.get("name")
        if not name:
            return ToolResult(success=False, message="Missing required argument: name")
        directory = self.store.get_directory(name)
        if directory is None:
            return ToolResult(success=False, message=f"No directory found for {name}")
        return ToolResult(
            success=True,
            message=f"Loaded directory {name}",
            data=directory.model_dump(),
        )
