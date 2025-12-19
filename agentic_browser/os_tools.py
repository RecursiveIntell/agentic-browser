"""
OS tools for Agentic Browser.

Provides OS action execution with safety controls for file operations and command execution.
"""

import os
import re
import shlex
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .config import AgentConfig
from .tool_schemas import (
    RunCommandRequest,
    ListDirRequest,
    ReadFileRequest,
    WriteFileRequest,
    MoveFileRequest,
    CopyFileRequest,
    DeleteFileRequest,
    validate_action_args,
)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    
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


class OSTools:
    """Executes OS actions with safety controls.
    
    Provides controlled access to filesystem and subprocess operations
    with timeout enforcement, path restrictions, and output truncation.
    """
    
    # Default limits
    DEFAULT_TIMEOUT_S = 30
    MAX_TIMEOUT_S = 120
    MAX_OUTPUT_CHARS = 8000
    MAX_READ_BYTES = 1024 * 1024  # 1MB
    
    # Dangerous command patterns that require HIGH risk classification
    HIGH_RISK_PATTERNS = [
        r"\brm\s+(-[rf]+\s+)*[^\s]+",  # rm with flags
        r"\bdd\b",                      # disk destroyer
        r"\bmkfs\b",                    # make filesystem
        r"\bsudo\b",                    # privilege escalation
        r"\bpkexec\b",                  # polkit escalation  
        r"\bsu\s",                      # switch user
        r"\bchmod\s+-R\b",              # recursive permissions
        r"\bchown\s+-R\b",              # recursive ownership
        r"\b(shutdown|reboot|poweroff)\b",  # system power
        r"\bsystemctl\s+(stop|disable|mask)\b",  # dangerous systemctl
        r">>\s*/etc/",                  # append to /etc
        r">\s*/etc/",                   # overwrite /etc
    ]
    
    # Paths that are HIGH risk to modify
    HIGH_RISK_PATHS = [
        "/etc", "/usr", "/bin", "/sbin", "/boot", 
        "/var", "/lib", "/lib64", "/opt", "/root",
    ]
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None,
        sandbox_dir: Optional[Path] = None,
        allow_outside_home: bool = False,
    ):
        """Initialize OS tools.
        
        Args:
            config: Agent configuration (optional)
            sandbox_dir: Sandbox directory for safe operations
            allow_outside_home: Whether to allow writes outside home
        """
        self.config = config
        self.sandbox_dir = sandbox_dir or Path.home() / "agentic_sandbox"
        self.allow_outside_home = allow_outside_home
        self.home_dir = Path.home()
        
        # Ensure sandbox exists
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, action: str, args: dict[str, Any]) -> ToolResult:
        """Execute an OS action (legacy interface).
        
        DEPRECATED: Use execute_typed() with Pydantic models instead.
        
        Args:
            action: Action name (os_exec, os_list_dir, os_read_file, os_write_file)
            args: Action arguments (raw dict)
            
        Returns:
            ToolResult with success status and message
        """
        action_map = {
            "os_exec": self._exec,
            "os_list_dir": self._list_dir,
            "os_read_file": self._read_file,
            "os_write_file": self._write_file,
            "os_move_file": self._move_file,
            "os_copy_file": self._copy_file,
            "os_delete_file": self._delete_file,
        }
        
        handler = action_map.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                message=f"Unknown OS action: {action}. Valid actions: {list(action_map.keys())}",
            )
        
        try:
            return handler(args)
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error executing {action}: {str(e)}",
            )
    
    def execute_typed(
        self,
        request: Union[
            RunCommandRequest,
            ListDirRequest,
            ReadFileRequest,
            WriteFileRequest,
            MoveFileRequest,
            CopyFileRequest,
            DeleteFileRequest,
        ],
    ) -> ToolResult:
        """Execute an OS action with typed request (PREFERRED).
        
        This is the preferred method for executing OS actions.
        Uses Pydantic models for type safety and validation.
        
        Args:
            request: Typed request object
            
        Returns:
            ToolResult with success status and message
        """
        if isinstance(request, RunCommandRequest):
            return self._exec_typed(request)
        elif isinstance(request, ListDirRequest):
            return self._list_dir({"path": request.path})
        elif isinstance(request, ReadFileRequest):
            return self._read_file({"path": request.path, "max_bytes": request.max_bytes})
        elif isinstance(request, WriteFileRequest):
            return self._write_file({
                "path": request.path,
                "content": request.content,
                "mode": request.mode,
            })
        elif isinstance(request, MoveFileRequest):
            return self._move_file({
                "source": request.source,
                "destination": request.destination,
            })
        elif isinstance(request, CopyFileRequest):
            return self._copy_file({
                "source": request.source,
                "destination": request.destination,
            })
        elif isinstance(request, DeleteFileRequest):
            return self._delete_file({"path": request.path})
        else:
            return ToolResult(
                success=False,
                message=f"Unknown request type: {type(request).__name__}",
            )
    
    def _exec_typed(self, request: RunCommandRequest) -> ToolResult:
        """Execute a command from typed request (argv list).
        
        Args:
            request: Typed RunCommandRequest with argv list
            
        Returns:
            ToolResult with stdout/stderr
        """
        timeout_s = min(request.timeout_s, self.MAX_TIMEOUT_S)
        
        cwd = request.cwd
        if cwd:
            cwd_path = Path(cwd).resolve()
            if not cwd_path.exists():
                return ToolResult(
                    success=False,
                    message=f"Working directory does not exist: {cwd}",
                )
        else:
            cwd_path = self.sandbox_dir
        
        try:
            result = subprocess.run(
                request.argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(cwd_path),
                env={**os.environ, "LC_ALL": "C"},
            )
            
            stdout = self._truncate_output(result.stdout)
            stderr = self._truncate_output(result.stderr)
            
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            
            output = "\n\n".join(output_parts) if output_parts else "(no output)"
            
            return ToolResult(
                success=result.returncode == 0,
                message=f"Command {'succeeded' if result.returncode == 0 else 'failed'} (exit code {result.returncode})",
                data={
                    "returncode": result.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "output": output,
                    "argv": request.argv,
                },
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Command timed out after {timeout_s} seconds",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                message=f"Command not found: {request.argv[0]}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                message=f"Permission denied executing: {request.argv[0]}",
            )
    
    def _exec(self, args: dict[str, Any]) -> ToolResult:
        """Execute a shell command (legacy method).
        
        DEPRECATED: Use execute_typed() with RunCommandRequest instead.
        
        Supports both legacy "cmd" string and new "argv" list formats.
        If both are provided, "argv" takes precedence.
        
        Args:
            args: {
                "argv": ["ls", "-la"],  # PREFERRED: list of args
                "cmd": "command string",  # DEPRECATED: shell string
                "timeout_s": 30,
                "cwd": "/path"
            }
            
        Returns:
            ToolResult with stdout/stderr
        """
        # Prefer argv list if provided
        argv = args.get("argv")
        if argv and isinstance(argv, list):
            request = RunCommandRequest(
                argv=argv,
                cwd=args.get("cwd"),
                timeout_s=args.get("timeout_s", self.DEFAULT_TIMEOUT_S),
            )
            return self._exec_typed(request)
        
        # Fall back to legacy cmd string (deprecated)
        cmd = args.get("cmd")
        if not cmd:
            return ToolResult(
                success=False,
                message="Missing required argument: cmd or argv",
            )
        
        # Emit deprecation warning for shell strings
        warnings.warn(
            "Using 'cmd' shell string is deprecated. Use 'argv' list instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        timeout_s = min(
            args.get("timeout_s", self.DEFAULT_TIMEOUT_S),
            self.MAX_TIMEOUT_S,
        )
        
        cwd = args.get("cwd")
        if cwd:
            cwd = Path(cwd).expanduser().resolve()
            if not cwd.exists():
                return ToolResult(
                    success=False,
                    message=f"Working directory does not exist: {cwd}",
                )
        else:
            cwd = self.sandbox_dir
        
        # Split command safely (no shell=True!)
        try:
            cmd_parts = shlex.split(cmd)
        except ValueError as e:
            return ToolResult(
                success=False,
                message=f"Invalid command syntax: {e}",
            )
        
        if not cmd_parts:
            return ToolResult(
                success=False,
                message="Empty command",
            )
        
        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(cwd),
                env={**os.environ, "LC_ALL": "C"},  # Ensure consistent output
            )
            
            stdout = self._truncate_output(result.stdout)
            stderr = self._truncate_output(result.stderr)
            
            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            
            output = "\n\n".join(output_parts) if output_parts else "(no output)"
            
            return ToolResult(
                success=result.returncode == 0,
                message=f"Command {'succeeded' if result.returncode == 0 else 'failed'} (exit code {result.returncode})",
                data={
                    "returncode": result.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "output": output,
                },
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Command timed out after {timeout_s} seconds",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                message=f"Command not found: {cmd_parts[0]}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                message=f"Permission denied executing: {cmd_parts[0]}",
            )
    
    def _list_dir(self, args: dict[str, Any]) -> ToolResult:
        """List directory contents.
        
        Args:
            args: {"path": "/path/to/dir"}
            
        Returns:
            ToolResult with file listing
        """
        path_str = args.get("path")
        if not path_str:
            return ToolResult(
                success=False,
                message="Missing required argument: path",
            )
        
        path = Path(path_str).expanduser().resolve()
        
        if not path.exists():
            return ToolResult(
                success=False,
                message=f"Path does not exist: {path}",
            )
        
        if not path.is_dir():
            return ToolResult(
                success=False,
                message=f"Path is not a directory: {path}",
            )
        
        try:
            entries = []
            for entry in sorted(path.iterdir()):
                try:
                    stat = entry.stat()
                    entry_info = {
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": stat.st_size if entry.is_file() else None,
                    }
                    entries.append(entry_info)
                except (PermissionError, OSError):
                    entries.append({
                        "name": entry.name,
                        "type": "unknown",
                        "error": "permission denied",
                    })
            
            # Format for display
            lines = [f"Contents of {path}:"]
            for e in entries[:100]:  # Limit to 100 entries
                type_icon = "📁" if e["type"] == "dir" else "📄"
                size_str = f" ({e['size']} bytes)" if e.get("size") else ""
                lines.append(f"  {type_icon} {e['name']}{size_str}")
            
            if len(entries) > 100:
                lines.append(f"  ... and {len(entries) - 100} more entries")
            
            return ToolResult(
                success=True,
                message="\n".join(lines),
                data={"entries": entries, "count": len(entries)},
            )
            
        except PermissionError:
            return ToolResult(
                success=False,
                message=f"Permission denied reading directory: {path}",
            )
    
    def _read_file(self, args: dict[str, Any]) -> ToolResult:
        """Read file contents.
        
        Args:
            args: {
                "path": "/path/to/file",
                "max_bytes": 10000  # optional
            }
            
        Returns:
            ToolResult with file contents
        """
        path_str = args.get("path")
        if not path_str:
            return ToolResult(
                success=False,
                message="Missing required argument: path",
            )
        
        path = Path(path_str).expanduser().resolve()
        
        if not path.exists():
            return ToolResult(
                success=False,
                message=f"File does not exist: {path}",
            )
        
        if not path.is_file():
            return ToolResult(
                success=False,
                message=f"Path is not a file: {path}",
            )
        
        max_bytes = min(
            args.get("max_bytes", self.MAX_READ_BYTES),
            self.MAX_READ_BYTES,
        )
        
        try:
            file_size = path.stat().st_size
            truncated = file_size > max_bytes
            
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_bytes)
            
            message = f"Contents of {path}"
            if truncated:
                message += f" (truncated from {file_size} to {max_bytes} bytes)"
            message += f":\n\n{content}"
            
            return ToolResult(
                success=True,
                message=message,
                data={
                    "content": content,
                    "size": file_size,
                    "truncated": truncated,
                },
            )
            
        except PermissionError:
            return ToolResult(
                success=False,
                message=f"Permission denied reading file: {path}",
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                message=f"File appears to be binary, cannot read as text: {path}",
            )
    
    def _write_file(self, args: dict[str, Any]) -> ToolResult:
        """Write content to a file.
        
        Args:
            args: {
                "path": "/path/to/file",
                "content": "file content",
                "mode": "overwrite|append"  # default: overwrite
            }
            
        Returns:
            ToolResult indicating success/failure
        """
        path_str = args.get("path")
        if not path_str:
            return ToolResult(
                success=False,
                message="Missing required argument: path",
            )
        
        content = args.get("content")
        if content is None:
            return ToolResult(
                success=False,
                message="Missing required argument: content",
            )
        
        mode = args.get("mode", "overwrite")
        if mode not in ("overwrite", "append"):
            return ToolResult(
                success=False,
                message=f"Invalid mode: {mode}. Must be 'overwrite' or 'append'",
            )
        
        path = Path(path_str).expanduser().resolve()
        blocked_reason = self._validate_write_path(path)
        if blocked_reason:
            return ToolResult(success=False, message=blocked_reason)
        
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            write_mode = "w" if mode == "overwrite" else "a"
            with open(path, write_mode, encoding="utf-8") as f:
                f.write(content)
            
            action = "Wrote" if mode == "overwrite" else "Appended"
            return ToolResult(
                success=True,
                message=f"{action} {len(content)} bytes to {path}",
                data={
                    "path": str(path),
                    "bytes_written": len(content),
                    "mode": mode,
                },
            )
            
        except PermissionError:
            return ToolResult(
                success=False,
                message=f"Permission denied writing to: {path}",
            )
        except OSError as e:
            return ToolResult(
                success=False,
                message=f"Error writing file: {e}",
            )

    def _move_file(self, args: dict[str, Any]) -> ToolResult:
        """Move or rename a file or directory.

        Args:
            args: {
                "source": "/path/to/source",
                "destination": "/path/to/destination"
            }

        Returns:
            ToolResult indicating success/failure
        """
        source = args.get("source")
        destination = args.get("destination")
        if not source or not destination:
            return ToolResult(
                success=False,
                message="Missing required arguments: source and destination",
            )

        source_path = Path(source).expanduser().resolve()
        dest_path = Path(destination).expanduser().resolve()

        blocked_reason = self._validate_write_path(source_path)
        if blocked_reason:
            return ToolResult(success=False, message=blocked_reason)

        blocked_reason = self._validate_write_path(dest_path)
        if blocked_reason:
            return ToolResult(success=False, message=blocked_reason)

        if not source_path.exists():
            return ToolResult(
                success=False,
                message=f"Source does not exist: {source_path}",
            )

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            result_path = shutil.move(str(source_path), str(dest_path))
            return ToolResult(
                success=True,
                message=f"Moved {source_path} to {result_path}",
                data={"source": str(source_path), "destination": str(result_path)},
            )
        except OSError as e:
            return ToolResult(
                success=False,
                message=f"Error moving file: {e}",
            )

    def _copy_file(self, args: dict[str, Any]) -> ToolResult:
        """Copy a file or directory.

        Args:
            args: {
                "source": "/path/to/source",
                "destination": "/path/to/destination"
            }

        Returns:
            ToolResult indicating success/failure
        """
        source = args.get("source")
        destination = args.get("destination")
        if not source or not destination:
            return ToolResult(
                success=False,
                message="Missing required arguments: source and destination",
            )

        source_path = Path(source).expanduser().resolve()
        dest_path = Path(destination).expanduser().resolve()

        blocked_reason = self._validate_write_path(source_path)
        if blocked_reason:
            return ToolResult(success=False, message=blocked_reason)

        blocked_reason = self._validate_write_path(dest_path)
        if blocked_reason:
            return ToolResult(success=False, message=blocked_reason)

        if not source_path.exists():
            return ToolResult(
                success=False,
                message=f"Source does not exist: {source_path}",
            )

        try:
            import shutil

            if source_path.is_dir():
                shutil.copytree(str(source_path), str(dest_path), dirs_exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source_path), str(dest_path))

            return ToolResult(
                success=True,
                message=f"Copied {source_path} to {dest_path}",
                data={"source": str(source_path), "destination": str(dest_path)},
            )
        except OSError as e:
            return ToolResult(
                success=False,
                message=f"Error copying file: {e}",
            )

    def _delete_file(self, args: dict[str, Any]) -> ToolResult:
        """Delete a file.

        Args:
            args: {
                "path": "/path/to/file"
            }

        Returns:
            ToolResult indicating success/failure
        """
        path_str = args.get("path")
        if not path_str:
            return ToolResult(
                success=False,
                message="Missing required argument: path",
            )

        path = Path(path_str).expanduser().resolve()
        blocked_reason = self._validate_write_path(path)
        if blocked_reason:
            return ToolResult(success=False, message=blocked_reason)

        if not path.exists():
            return ToolResult(
                success=False,
                message=f"Path does not exist: {path}",
            )

        if path.is_dir():
            return ToolResult(
                success=False,
                message=f"Refusing to delete directory: {path}",
            )

        try:
            path.unlink()
            return ToolResult(
                success=True,
                message=f"Deleted file {path}",
                data={"path": str(path)},
            )
        except OSError as e:
            return ToolResult(
                success=False,
                message=f"Error deleting file: {e}",
            )

    def _validate_write_path(self, path: Path) -> Optional[str]:
        """Validate a path for write/delete/move/copy operations."""
        if not self.allow_outside_home:
            if not str(path).startswith(str(self.home_dir)):
                return (
                    f"Write blocked: {path} is outside home directory. "
                    f"Writes are restricted to {self.home_dir}"
                )

        path_str_lower = str(path).lower()
        for risk_path in self.HIGH_RISK_PATHS:
            if path_str_lower.startswith(risk_path):
                return f"Write blocked: {path} is in protected system path {risk_path}"

        return None
    
    def _truncate_output(self, text: str) -> str:
        """Truncate output to max chars."""
        if len(text) <= self.MAX_OUTPUT_CHARS:
            return text
        
        half = self.MAX_OUTPUT_CHARS // 2
        return (
            text[:half] + 
            f"\n\n... [truncated {len(text) - self.MAX_OUTPUT_CHARS} chars] ...\n\n" +
            text[-half:]
        )
    
    def classify_risk(self, action: str, args: dict[str, Any]) -> str:
        """Classify the risk level of an OS action.
        
        Args:
            action: The action type
            args: Action arguments
            
        Returns:
            Risk level: "low", "medium", or "high"
        """
        if action == "os_exec":
            cmd = args.get("cmd", "")
            
            # Check high-risk patterns
            for pattern in self.HIGH_RISK_PATTERNS:
                if re.search(pattern, cmd, re.IGNORECASE):
                    return "high"
            
            # Writing/modifying actions are medium risk
            if any(op in cmd.lower() for op in ["mv ", "cp ", "rm ", ">", "tee ", "sed -i", "chmod ", "chown "]):
                return "medium"
            
            return "low"
        
        elif action == "os_write_file":
            path_str = args.get("path", "")
            path = Path(path_str).expanduser().resolve()
            
            # Check high-risk paths
            for risk_path in self.HIGH_RISK_PATHS:
                if str(path).startswith(risk_path):
                    return "high"
            
            # Writing outside home is high risk
            if not str(path).startswith(str(self.home_dir)):
                return "high"
            
            return "medium"
        
        elif action == "os_read_file":
            # Reading sensitive files
            path_str = args.get("path", "")
            if any(sensitive in path_str.lower() for sensitive in [
                "/etc/shadow", "/etc/passwd", ".ssh/", "id_rsa", ".gnupg/"
            ]):
                return "medium"
            return "low"
        
        elif action == "os_list_dir":
            return "low"
        
        return "low"
    
    def is_high_risk_command(self, cmd: str) -> tuple[bool, str]:
        """Check if a command matches high-risk patterns.
        
        Args:
            cmd: Command string
            
        Returns:
            Tuple of (is_high_risk, reason)
        """
        for pattern in self.HIGH_RISK_PATTERNS:
            match = re.search(pattern, cmd, re.IGNORECASE)
            if match:
                return True, f"Matches dangerous pattern: {pattern}"
        
        return False, ""
