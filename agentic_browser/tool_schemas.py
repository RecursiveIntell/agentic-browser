"""
Typed tool schemas for Agentic Browser.

Provides Pydantic models for all tool arguments with validation.
All actions MUST use these schemas - no freeform dicts or shell strings.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, HttpUrl


# =============================================================================
# Enums and Base Types
# =============================================================================

class RiskLevel(str, Enum):
    """Risk level for an action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionDomain(str, Enum):
    """Domain for action execution."""
    BROWSER = "browser"
    OS = "os"
    MEMORY = "memory"


# =============================================================================
# OS Tool Schemas
# =============================================================================

class ListDirRequest(BaseModel):
    """Request to list directory contents."""
    
    path: str = Field(
        description="Directory path to list (absolute or relative to home)"
    )
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Expand ~ and validate path format."""
        if not v:
            raise ValueError("Path cannot be empty")
        return str(Path(v).expanduser())


class ReadFileRequest(BaseModel):
    """Request to read file contents."""
    
    path: str = Field(description="File path to read")
    max_bytes: int = Field(
        default=1048576,
        ge=1,
        le=10485760,
        description="Maximum bytes to read (1-10MB)"
    )
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        return str(Path(v).expanduser())


class WriteFileRequest(BaseModel):
    """Request to write content to a file."""
    
    path: str = Field(description="File path to write")
    content: str = Field(description="Content to write")
    mode: Literal["overwrite", "append"] = Field(
        default="overwrite",
        description="Write mode"
    )
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        return str(Path(v).expanduser())
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        # Allow empty content but warn
        return v


class RunCommandRequest(BaseModel):
    """Request to run a command.
    
    IMPORTANT: Commands must be specified as argv list, NOT shell strings.
    This prevents shell injection and makes commands deterministic.
    
    Example:
        ✅ Correct: argv=["ls", "-la", "/home"]
        ❌ Wrong: cmd="ls -la /home"  # Shell string - dangerous!
    """
    
    argv: list[str] = Field(
        description="Command as list of arguments (NOT a shell string)"
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Working directory for command execution"
    )
    timeout_s: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Timeout in seconds (1-120)"
    )
    
    @field_validator("argv")
    @classmethod
    def validate_argv(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("argv cannot be empty")
        if any(not isinstance(arg, str) for arg in v):
            raise ValueError("All argv elements must be strings")
        return v
    
    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return str(Path(v).expanduser())
        return v


class MoveFileRequest(BaseModel):
    """Request to move/rename a file."""
    
    source: str = Field(description="Source path")
    destination: str = Field(description="Destination path")
    
    @field_validator("source", "destination")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        return str(Path(v).expanduser())


class CopyFileRequest(BaseModel):
    """Request to copy a file."""
    
    source: str = Field(description="Source path")
    destination: str = Field(description="Destination path")
    
    @field_validator("source", "destination")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        return str(Path(v).expanduser())


class DeleteFileRequest(BaseModel):
    """Request to delete a file.
    
    NOTE: This is a HIGH risk operation and will always require approval.
    """
    
    path: str = Field(description="File path to delete")
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Path cannot be empty")
        return str(Path(v).expanduser())


# =============================================================================
# Browser Tool Schemas
# =============================================================================

class GotoRequest(BaseModel):
    """Request to navigate to a URL."""
    
    url: str = Field(description="URL to navigate to")
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v:
            raise ValueError("URL cannot be empty")
        # Basic validation - allow http, https, and file protocols
        if not v.startswith(("http://", "https://", "file://")):
            # Assume https if no protocol
            v = f"https://{v}"
        return v


class ClickRequest(BaseModel):
    """Request to click an element."""
    
    selector: str = Field(description="Element selector (CSS, text=, xpath=)")
    timeout_ms: int = Field(
        default=10000,
        ge=100,
        le=60000,
        description="Timeout in milliseconds"
    )
    
    @field_validator("selector")
    @classmethod
    def validate_selector(cls, v: str) -> str:
        if not v:
            raise ValueError("Selector cannot be empty")
        return v.strip()


class TypeRequest(BaseModel):
    """Request to type text into an element."""
    
    selector: str = Field(description="Element selector")
    text: str = Field(description="Text to type")
    clear_first: bool = Field(
        default=True,
        description="Clear field before typing"
    )
    
    @field_validator("selector")
    @classmethod
    def validate_selector(cls, v: str) -> str:
        if not v:
            raise ValueError("Selector cannot be empty")
        return v.strip()


class PressRequest(BaseModel):
    """Request to press a keyboard key."""
    
    key: str = Field(
        description="Key to press (e.g., 'Enter', 'Tab', 'ArrowDown')"
    )
    
    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        if not v:
            raise ValueError("Key cannot be empty")
        return v.strip()


class ScrollRequest(BaseModel):
    """Request to scroll the page."""
    
    amount: int = Field(
        default=800,
        description="Scroll amount in pixels (positive=down, negative=up)"
    )


class WaitForRequest(BaseModel):
    """Request to wait for an element or timeout."""
    
    selector: Optional[str] = Field(
        default=None,
        description="Element selector to wait for (optional)"
    )
    timeout_ms: int = Field(
        default=10000,
        ge=100,
        le=60000,
        description="Timeout in milliseconds"
    )


class ExtractRequest(BaseModel):
    """Request to extract data from an element."""
    
    selector: str = Field(description="Element selector")
    attribute: str = Field(
        default="innerText",
        description="Attribute to extract (innerText, href, value, etc.)"
    )


class ExtractVisibleTextRequest(BaseModel):
    """Request to extract all visible text from the page."""
    
    max_chars: int = Field(
        default=8000,
        ge=100,
        le=50000,
        description="Maximum characters to return"
    )


class ScreenshotRequest(BaseModel):
    """Request to take a screenshot."""
    
    label: Optional[str] = Field(
        default=None,
        description="Optional label for the screenshot"
    )


class DownloadFileRequest(BaseModel):
    """Request to download a file from the browser."""
    
    url: str = Field(description="URL to download")
    save_path: str = Field(description="Local path to save the file")
    
    @field_validator("save_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Save path cannot be empty")
        return str(Path(v).expanduser())


class DownloadImageRequest(BaseModel):
    """Request to download an image from the browser."""

    selector: Optional[str] = Field(
        default=None,
        description="Optional selector to find an image element",
    )
    url: Optional[str] = Field(
        default=None,
        description="Optional direct image URL",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename for the downloaded image",
    )


# =============================================================================
# Memory Tool Schemas (Phase 3 - placeholder)
# =============================================================================

class MemoryGetSiteRequest(BaseModel):
    """Request to get known site information."""
    domain: str = Field(description="Domain to look up")


class MemorySaveSiteRequest(BaseModel):
    """Request to save site information."""
    domain: str = Field(description="Domain to save")
    data: dict = Field(description="Site data to store")


class MemoryGetDirectoryRequest(BaseModel):
    """Request to get known directory information."""
    name: str = Field(description="Directory name to look up")


# =============================================================================
# Action Request Wrapper
# =============================================================================

# Type alias for all OS requests
OSRequest = Union[
    ListDirRequest,
    ReadFileRequest,
    WriteFileRequest,
    RunCommandRequest,
    MoveFileRequest,
    CopyFileRequest,
    DeleteFileRequest,
]

# Type alias for all browser requests
BrowserRequest = Union[
    GotoRequest,
    ClickRequest,
    TypeRequest,
    PressRequest,
    ScrollRequest,
    WaitForRequest,
    ExtractRequest,
    ExtractVisibleTextRequest,
    ScreenshotRequest,
    DownloadFileRequest,
    DownloadImageRequest,
]

# Type alias for all memory requests
MemoryRequest = Union[
    MemoryGetSiteRequest,
    MemorySaveSiteRequest,
    MemoryGetDirectoryRequest,
]


class ActionRequest(BaseModel):
    """Base wrapper for all action requests.
    
    This is the ONLY way actions should be requested.
    No freeform dicts or shell strings allowed.
    """
    
    action: str = Field(description="Action name")
    domain: ActionDomain = Field(description="Execution domain")
    request: Union[OSRequest, BrowserRequest, MemoryRequest, dict] = Field(
        description="Typed request object"
    )
    rationale: str = Field(
        default="",
        description="Why this action is being requested"
    )
    
    def to_legacy_args(self) -> dict[str, Any]:
        """Convert to legacy args dict for backward compatibility.
        
        DEPRECATED: Use typed request objects directly.
        """
        if isinstance(self.request, dict):
            return self.request
        return self.request.model_dump()


# =============================================================================
# Action Response
# =============================================================================

class ActionResult(BaseModel):
    """Result of an action execution."""
    
    success: bool
    message: str
    data: Optional[dict] = None
    risk_level: RiskLevel = RiskLevel.LOW
    required_approval: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "success": self.success,
            "message": self.message,
            "risk_level": self.risk_level.value,
        }
        if self.data is not None:
            result["data"] = self.data
        return result


# =============================================================================
# Schema Registry
# =============================================================================

ACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    # OS actions
    "os_list_dir": ListDirRequest,
    "os_read_file": ReadFileRequest,
    "os_write_file": WriteFileRequest,
    "os_exec": RunCommandRequest,
    "os_move_file": MoveFileRequest,
    "os_copy_file": CopyFileRequest,
    "os_delete_file": DeleteFileRequest,
    # Browser actions
    "goto": GotoRequest,
    "click": ClickRequest,
    "type": TypeRequest,
    "press": PressRequest,
    "scroll": ScrollRequest,
    "wait_for": WaitForRequest,
    "extract": ExtractRequest,
    "extract_visible_text": ExtractVisibleTextRequest,
    "screenshot": ScreenshotRequest,
    "download_file": DownloadFileRequest,
    "download_image": DownloadImageRequest,
    # Memory actions (Phase 3)
    "memory_get_site": MemoryGetSiteRequest,
    "memory_save_site": MemorySaveSiteRequest,
    "memory_get_directory": MemoryGetDirectoryRequest,
}


def get_schema_for_action(action: str) -> Optional[type[BaseModel]]:
    """Get the Pydantic schema for an action.
    
    Args:
        action: Action name
        
    Returns:
        Schema class or None if not found
    """
    return ACTION_SCHEMAS.get(action)


def validate_action_args(action: str, args: dict) -> tuple[bool, Optional[BaseModel], Optional[str]]:
    """Validate action arguments against schema.
    
    Args:
        action: Action name
        args: Arguments to validate
        
    Returns:
        Tuple of (is_valid, validated_model, error_message)
    """
    schema = get_schema_for_action(action)
    if schema is None:
        return False, None, f"Unknown action: {action}"
    
    try:
        validated = schema(**args)
        return True, validated, None
    except Exception as e:
        return False, None, str(e)


def create_typed_request(action: str, args: dict) -> Optional[BaseModel]:
    """Create a typed request from action and args.
    
    Args:
        action: Action name
        args: Arguments dict
        
    Returns:
        Validated Pydantic model or None if validation fails
    """
    is_valid, model, error = validate_action_args(action, args)
    if is_valid:
        return model
    return None
