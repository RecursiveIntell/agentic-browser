"""
Approval system abstraction for Agentic Browser.

Provides a unified interface for action approval that works in both CLI and GUI modes.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from .safety import RiskLevel


class Approver(ABC):
    """Abstract base class for action approval handlers."""
    
    @abstractmethod
    def request_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        rationale: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Request user approval for an action.
        
        Args:
            action: The action type (click, type, etc.)
            args: Action arguments
            risk_level: Classified risk level
            rationale: The model's rationale
            
        Returns:
            Tuple of (approved, optional_modified_action)
            If approved is False and modified_action is provided,
            use the modified action instead.
        """
        pass
    
    @abstractmethod
    def notify_denial(self) -> str:
        """Notify user that action was denied and get guidance.
        
        Returns:
            User's guidance for the next action, or empty string
        """
        pass


class ConsoleApprover(Approver):
    """CLI approval via Rich prompts.
    
    Used when running the agent from the command line.
    """
    
    def __init__(self):
        from rich.console import Console
        from rich.prompt import Prompt
        
        self.console = Console()
        self.Prompt = Prompt
    
    def request_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        rationale: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Request approval via Rich interactive prompt."""
        import json
        
        risk_color = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow", 
            RiskLevel.HIGH: "red"
        }[risk_level]
        
        self.console.print()
        self.console.print(f"[bold]Action:[/bold] {action}")
        self.console.print(f"[bold]Arguments:[/bold] {args}")
        self.console.print(f"[bold]Risk:[/bold] [{risk_color}]{risk_level.value}[/{risk_color}]")
        self.console.print(f"[bold]Rationale:[/bold] {rationale}")
        self.console.print()
        
        response = self.Prompt.ask(
            "[yellow]Approve action?[/yellow]",
            choices=["y", "n", "e"],
            default="n",
        )
        
        if response == "y":
            return True, None
        
        if response == "e":
            modified = self._edit_action(action, args)
            if modified:
                return True, modified
            return False, None
        
        return False, None
    
    def _edit_action(
        self, 
        action: str, 
        args: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Allow user to edit the action."""
        import json
        
        self.console.print()
        self.console.print("[dim]Edit the action JSON. Press Enter to submit, or type 'cancel' to abort.[/dim]")
        
        current_json = json.dumps({"action": action, "args": args}, indent=2)
        self.console.print(f"Current: {current_json}")
        
        user_input = self.Prompt.ask("New action JSON (or 'cancel')")
        
        if user_input.lower() == "cancel":
            return None
        
        try:
            modified = json.loads(user_input)
            return modified
        except json.JSONDecodeError:
            self.console.print("[red]Invalid JSON, action cancelled[/red]")
            return None
    
    def notify_denial(self) -> str:
        """Get guidance after denial via Rich prompt."""
        self.console.print()
        guidance = self.Prompt.ask(
            "[yellow]Action denied. Provide guidance for next action (or press Enter to let agent retry)[/yellow]",
            default="",
        )
        return guidance


class AutoApprover(Approver):
    """Automatically approve all actions.
    
    Used when:
    - auto-approve mode is enabled
    - Running as subprocess from GUI (approvals handled by parent)
    - Testing
    """
    
    def request_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        rationale: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Auto-approve without prompting."""
        return True, None
    
    def notify_denial(self) -> str:
        """Return empty guidance since no denial is possible."""
        return ""


class QtApprover(Approver):
    """Qt-based approval via dialog.
    
    Used when running the agent in-process within the GUI.
    Shows the ApprovalDialog from the gui module.
    """
    
    def __init__(self, parent_widget=None):
        """Initialize with optional parent widget for dialog."""
        self.parent_widget = parent_widget
    
    def request_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        rationale: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Show Qt dialog for approval."""
        from .gui.approval_dialog import ApprovalDialog
        
        dialog = ApprovalDialog(
            action=action,
            args=args,
            risk_level=risk_level,
            rationale=rationale,
            parent=self.parent_widget,
        )
        
        dialog.exec()
        result_code, modified = dialog.get_result()
        
        if result_code == ApprovalDialog.APPROVED:
            return True, None
        elif result_code == ApprovalDialog.EDITED:
            return True, modified
        else:
            return False, None
    
    def notify_denial(self) -> str:
        """Show input dialog for guidance after denial."""
        from PySide6.QtWidgets import QInputDialog
        
        text, ok = QInputDialog.getText(
            self.parent_widget,
            "Action Denied",
            "Provide guidance for next action (or leave empty):",
        )
        
        return text if ok else ""


def get_approver(
    mode: str = "cli",
    auto_approve: bool = False,
    parent_widget=None,
) -> Approver:
    """Get the appropriate approver for the given mode.
    
    Args:
        mode: "cli", "gui", "ipc", or "auto"
        auto_approve: If True, always return AutoApprover
        parent_widget: Parent widget for Qt dialogs
        
    Returns:
        Appropriate Approver instance
    """
    if auto_approve:
        return AutoApprover()
    
    if mode == "gui":
        return QtApprover(parent_widget)
    
    if mode == "ipc":
        return IPCApprover()
    
    if mode == "auto":
        return AutoApprover()
    
    # Default: CLI mode
    return ConsoleApprover()


class IPCApprover(Approver):
    """IPC-based approval for GUI subprocess mode.
    
    Communicates with parent process (GUI) via JSON over stdout/stdin.
    Uses a special protocol:
    - Output: APPROVAL_REQUEST:<json>
    - Input: APPROVAL_RESPONSE:<json>
    """
    
    # Protocol markers
    REQUEST_PREFIX = "APPROVAL_REQUEST:"
    RESPONSE_PREFIX = "APPROVAL_RESPONSE:"
    
    def request_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        rationale: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Request approval via IPC with parent process."""
        import json
        import sys
        
        # Build request JSON
        request = {
            "action": action,
            "args": args,
            "risk_level": risk_level.value,
            "rationale": rationale,
        }
        
        # Send request to parent
        request_line = f"{self.REQUEST_PREFIX}{json.dumps(request)}"
        print(request_line, flush=True)
        
        # Wait for response from parent
        try:
            response_line = sys.stdin.readline().strip()
            if response_line.startswith(self.RESPONSE_PREFIX):
                response_json = response_line[len(self.RESPONSE_PREFIX):]
                response = json.loads(response_json)
                
                approved = response.get("approved", False)
                modified = response.get("modified_action")
                
                return approved, modified
            else:
                # Unexpected response, deny
                return False, None
        except Exception:
            # On any error, deny
            return False, None
    
    def notify_denial(self) -> str:
        """Get guidance after denial via IPC."""
        import json
        import sys
        
        # Request guidance
        request = {"type": "guidance"}
        request_line = f"{self.REQUEST_PREFIX}{json.dumps(request)}"
        print(request_line, flush=True)
        
        # Wait for response
        try:
            response_line = sys.stdin.readline().strip()
            if response_line.startswith(self.RESPONSE_PREFIX):
                response_json = response_line[len(self.RESPONSE_PREFIX):]
                response = json.loads(response_json)
                return response.get("guidance", "")
        except Exception:
            pass
        
        return ""
