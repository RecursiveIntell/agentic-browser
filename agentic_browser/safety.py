"""
Safety classification and approval system for Agentic Browser.

Provides risk classification and user approval prompting.
"""

from enum import Enum
from typing import Any, Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm

from .utils import (
    contains_high_risk_keywords,
    contains_medium_risk_keywords,
    is_payment_domain,
    is_password_field,
)


class RiskLevel(str, Enum):
    """Risk level for an action."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyClassifier:
    """Classifies the risk level of browser actions."""
    
    # High-risk action patterns
    HIGH_RISK_BUTTON_TEXTS = {
        "buy", "purchase", "checkout", "pay", "order", "subscribe",
        "delete", "remove", "cancel", "terminate", "deactivate",
        "send", "post", "submit order", "place order", "confirm purchase",
        "confirm payment", "complete order", "close account",
    }
    
    # Form submission patterns
    SUBMIT_SELECTORS = {
        'input[type="submit"]',
        'button[type="submit"]',
        'input[type="button"]',
        ".submit",
        "#submit",
    }
    
    # Account/security page patterns
    SECURITY_PATHS = {
        "/account", "/settings", "/security", "/password", "/profile",
        "/preferences", "/billing", "/payment", "/subscription",
        "/delete", "/deactivate", "/close-account",
    }
    
    # Message sending patterns
    MESSAGE_SELECTORS = {
        "send", "compose", "reply", "message", "email", "tweet",
        "post", "comment", "publish",
    }
    
    def classify_action(
        self,
        action: str,
        args: dict[str, Any],
        current_url: str = "",
        page_content: str = "",
    ) -> RiskLevel:
        """Classify the risk level of an action.
        
        Args:
            action: The action type (click, type, etc.)
            args: Action arguments
            current_url: Current page URL
            page_content: Current page visible text (optional)
            
        Returns:
            Risk level classification
        """
        # Check URL-based risks first
        if is_payment_domain(current_url):
            return RiskLevel.HIGH
        
        if self._is_security_page(current_url):
            return RiskLevel.HIGH
        
        # Action-specific classification
        if action == "click":
            return self._classify_click(args, page_content)
        
        if action == "type":
            return self._classify_type(args)
        
        if action == "press":
            return self._classify_press(args, current_url)
        
        if action == "goto":
            return self._classify_goto(args)
        
        # Default: low risk for navigation/extraction actions
        return RiskLevel.LOW
    
    def _classify_click(
        self, 
        args: dict[str, Any], 
        page_content: str
    ) -> RiskLevel:
        """Classify a click action."""
        selector = args.get("selector", "").lower()
        
        # Check for high-risk button text
        for keyword in self.HIGH_RISK_BUTTON_TEXTS:
            if keyword in selector:
                return RiskLevel.HIGH
        
        # Check for submit button patterns
        for pattern in self.SUBMIT_SELECTORS:
            if pattern in selector:
                # Check page content for purchase context
                if contains_high_risk_keywords(page_content):
                    return RiskLevel.HIGH
                return RiskLevel.MEDIUM
        
        # Check for message sending
        for keyword in self.MESSAGE_SELECTORS:
            if keyword in selector:
                return RiskLevel.HIGH
        
        return RiskLevel.LOW
    
    def _classify_type(self, args: dict[str, Any]) -> RiskLevel:
        """Classify a type action."""
        selector = args.get("selector", "").lower()
        
        # Password field is medium risk (login)
        if is_password_field(selector):
            return RiskLevel.MEDIUM
        
        # Check for message composition
        for keyword in self.MESSAGE_SELECTORS:
            if keyword in selector:
                return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _classify_press(
        self, 
        args: dict[str, Any], 
        current_url: str
    ) -> RiskLevel:
        """Classify a key press action."""
        key = args.get("key", "").lower()
        
        # Enter key can submit forms
        if key == "enter":
            # On security/payment pages, this is higher risk
            if is_payment_domain(current_url) or self._is_security_page(current_url):
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _classify_goto(self, args: dict[str, Any]) -> RiskLevel:
        """Classify a navigation action."""
        url = args.get("url", "").lower()
        
        if is_payment_domain(url):
            return RiskLevel.MEDIUM  # Just navigating, not transacting
        
        if self._is_security_page(url):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _is_security_page(self, url: str) -> bool:
        """Check if URL is a security/account settings page."""
        url_lower = url.lower()
        return any(path in url_lower for path in self.SECURITY_PATHS)
    
    def should_require_approval(
        self,
        risk_level: RiskLevel,
        model_says_approval: bool,
        auto_approve: bool,
    ) -> bool:
        """Determine if approval is required for an action.
        
        Args:
            risk_level: Classified risk level
            model_says_approval: What the model indicated for requires_approval
            auto_approve: Whether auto-approve mode is enabled
            
        Returns:
            True if user approval is required
        """
        # High risk always requires approval
        if risk_level == RiskLevel.HIGH:
            return True
        
        # Medium risk requires approval unless auto-approve is on
        if risk_level == RiskLevel.MEDIUM:
            return not auto_approve
        
        # Model can also flag actions for approval
        if model_says_approval and risk_level != RiskLevel.LOW:
            return not auto_approve
        
        return False


class ApprovalPrompt:
    """Handles user approval prompting for risky actions."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the approval prompt.
        
        Args:
            console: Rich console for output (creates new if None)
        """
        self.console = console or Console()
    
    def request_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
        rationale: str,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Request user approval for an action.
        
        Args:
            action: The action type
            args: Action arguments
            risk_level: Classified risk level
            rationale: The model's rationale
            
        Returns:
            Tuple of (approved, optional_modified_action)
            If approved is False and modified_action is provided,
            use the modified action instead.
        """
        # Display the action details
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
        
        # Ask for approval
        response = Prompt.ask(
            "[yellow]Approve action?[/yellow]",
            choices=["y", "n", "e"],
            default="n",
        )
        
        if response == "y":
            return True, None
        
        if response == "e":
            # Allow editing the action
            modified = self._edit_action(action, args)
            if modified:
                return True, modified
            return False, None
        
        # Denied
        return False, None
    
    def _edit_action(
        self, 
        action: str, 
        args: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Allow user to edit the action.
        
        Args:
            action: Current action type
            args: Current arguments
            
        Returns:
            Modified action dict, or None if editing cancelled
        """
        import json
        
        self.console.print()
        self.console.print("[dim]Edit the action JSON. Press Enter to submit, or type 'cancel' to abort.[/dim]")
        
        current_json = json.dumps({"action": action, "args": args}, indent=2)
        self.console.print(f"Current: {current_json}")
        
        user_input = Prompt.ask("New action JSON (or 'cancel')")
        
        if user_input.lower() == "cancel":
            return None
        
        try:
            modified = json.loads(user_input)
            return modified
        except json.JSONDecodeError:
            self.console.print("[red]Invalid JSON, action cancelled[/red]")
            return None
    
    def notify_denial(self) -> str:
        """Notify user that action was denied and get guidance.
        
        Returns:
            User's guidance for the next action, or empty string
        """
        self.console.print()
        guidance = Prompt.ask(
            "[yellow]Action denied. Provide guidance for next action (or press Enter to let agent retry)[/yellow]",
            default="",
        )
        return guidance


def classify_risk(
    action: str,
    args: dict[str, Any],
    current_url: str = "",
    page_content: str = "",
) -> RiskLevel:
    """Convenience function to classify action risk.
    
    Args:
        action: The action type
        args: Action arguments  
        current_url: Current page URL
        page_content: Current visible page text
        
    Returns:
        Risk level classification
    """
    classifier = SafetyClassifier()
    return classifier.classify_action(action, args, current_url, page_content)
