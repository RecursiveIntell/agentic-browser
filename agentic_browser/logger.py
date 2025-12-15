"""
Logging and artifact management for Agentic Browser.

Handles JSONL step logging, screenshot saving, and rich console output.
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .config import get_runs_dir


def slugify(text: str, max_length: int = 30) -> str:
    """Convert text to a filesystem-safe slug."""
    # Convert to lowercase and replace spaces/special chars with underscores
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '_', slug).strip('_')
    return slug[:max_length]


def redact_password(text: str, field_type: str = "") -> str:
    """Redact password contents from logs."""
    if "password" in field_type.lower():
        return "[REDACTED]"
    return text


class RunLogger:
    """Manages logging and artifacts for a single agent run."""
    
    def __init__(self, goal: str, enable_console: bool = True):
        """Initialize the run logger.
        
        Args:
            goal: The goal being executed (used for directory naming)
            enable_console: Whether to print to console
        """
        self.goal = goal
        self.console = Console() if enable_console else None
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        goal_slug = slugify(goal)
        self.run_dir = get_runs_dir() / f"{timestamp}_{goal_slug}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.screenshots_dir = self.run_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)
        
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        # Initialize JSONL log file
        self.steps_file = self.run_dir / "steps.jsonl"
        self.steps_file.touch()
        
        self.step_count = 0
        
    @property
    def run_path(self) -> Path:
        """Get the path to the run directory."""
        return self.run_dir
    
    def log_step(
        self,
        state: dict[str, Any],
        model_output: dict[str, Any],
        result: dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """Log a single step to the JSONL file.
        
        Args:
            state: The page state sent to the model
            model_output: The model's response
            result: The execution result
            error: Optional error message
        """
        self.step_count += 1
        
        # Redact sensitive information
        sanitized_output = self._sanitize_model_output(model_output)
        
        step_data = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "state_summary": {
                "url": state.get("current_url", ""),
                "title": state.get("page_title", ""),
            },
            "model_output": sanitized_output,
            "result": result,
            "error": error,
        }
        
        with open(self.steps_file, "a") as f:
            f.write(json.dumps(step_data) + "\n")
    
    def _sanitize_model_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive data from model output before logging."""
        sanitized = output.copy()
        
        # Redact password typing
        if sanitized.get("action") == "type":
            args = sanitized.get("args", {})
            selector = args.get("selector", "")
            if "password" in selector.lower():
                sanitized["args"] = args.copy()
                sanitized["args"]["text"] = "[REDACTED]"
        
        return sanitized
    
    def save_screenshot(
        self, 
        screenshot_bytes: bytes, 
        label: Optional[str] = None
    ) -> Path:
        """Save a screenshot to the screenshots directory.
        
        Args:
            screenshot_bytes: PNG screenshot data
            label: Optional label for the screenshot
            
        Returns:
            Path to the saved screenshot
        """
        label_part = f"_{slugify(label)}" if label else ""
        filename = f"step_{self.step_count:03d}{label_part}.png"
        path = self.screenshots_dir / filename
        path.write_bytes(screenshot_bytes)
        return path
    
    def save_snapshot(self, content: str, extension: str = "txt") -> Path:
        """Save a page snapshot (HTML or text) to the snapshots directory.
        
        Args:
            content: The content to save
            extension: File extension (txt or html)
            
        Returns:
            Path to the saved snapshot
        """
        filename = f"step_{self.step_count:03d}.{extension}"
        path = self.snapshots_dir / filename
        path.write_text(content, encoding="utf-8")
        return path
    
    def save_trace(self, trace_path: Path) -> Optional[Path]:
        """Save Playwright trace file to the run directory.
        
        Args:
            trace_path: Path to the trace.zip file
            
        Returns:
            Path to the saved trace, or None if source doesn't exist
        """
        if trace_path.exists():
            dest = self.run_dir / "trace.zip"
            shutil.copy2(trace_path, dest)
            return dest
        return None
    
    def print_header(self) -> None:
        """Print the run header to console."""
        if not self.console:
            return
            
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]Goal:[/bold cyan] {self.goal}",
            title="🤖 Agentic Browser",
            border_style="cyan",
        ))
        self.console.print()
    
    def print_step(
        self,
        action: str,
        args: dict[str, Any],
        rationale: str,
        risk: str,
        requires_approval: bool,
    ) -> None:
        """Print a step to the console.
        
        Args:
            action: The action being taken
            args: Action arguments
            rationale: Why this action was chosen
            risk: Risk level (low/medium/high)
            requires_approval: Whether approval is required
        """
        if not self.console:
            return
        
        # Risk color coding
        risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
        risk_color = risk_colors.get(risk.lower(), "white")
        
        # Build step display
        step_text = Text()
        step_text.append(f"Step {self.step_count}: ", style="bold")
        step_text.append(f"{action}", style="bold cyan")
        
        # Format args nicely
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        if args_str:
            step_text.append(f"({args_str})", style="dim")
        
        self.console.print(step_text)
        self.console.print(f"  [dim]Rationale:[/dim] {rationale}")
        self.console.print(f"  [dim]Risk:[/dim] [{risk_color}]{risk}[/{risk_color}]", end="")
        
        if requires_approval:
            self.console.print(" [bold yellow]⚠ APPROVAL REQUIRED[/bold yellow]")
        else:
            self.console.print()
        
        self.console.print()
    
    def print_result(self, success: bool, message: str) -> None:
        """Print an action result to console.
        
        Args:
            success: Whether the action succeeded
            message: Result message
        """
        if not self.console:
            return
        
        if success:
            self.console.print(f"  [green]✓[/green] {message}")
        else:
            self.console.print(f"  [red]✗[/red] {message}")
    
    def print_error(self, error: str) -> None:
        """Print an error message to console.
        
        Args:
            error: Error message
        """
        if not self.console:
            return
        self.console.print(f"  [bold red]Error:[/bold red] {error}")
    
    def print_final_answer(self, answer: str) -> None:
        """Print the final answer to console.
        
        Args:
            answer: The final answer/summary
        """
        if not self.console:
            return
        
        self.console.print()
        self.console.print(Panel(
            answer,
            title="📋 Final Answer",
            border_style="green",
        ))
    
    def print_summary(self) -> None:
        """Print the run summary to console."""
        if not self.console:
            return
        
        table = Table(title="Run Summary", show_header=False)
        table.add_column("Property", style="dim")
        table.add_column("Value")
        
        table.add_row("Steps Executed", str(self.step_count))
        table.add_row("Logs Directory", str(self.run_dir))
        table.add_row("Steps Log", str(self.steps_file))
        table.add_row("Screenshots", str(len(list(self.screenshots_dir.glob("*.png")))))
        
        self.console.print()
        self.console.print(table)
    
    def print_approval_prompt(
        self, 
        action: str, 
        args: dict[str, Any], 
        risk: str,
        rationale: str
    ) -> None:
        """Print an approval prompt to console.
        
        Args:
            action: The action requiring approval
            args: Action arguments
            risk: Risk level
            rationale: Why this action was chosen
        """
        if not self.console:
            return
        
        risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
        risk_color = risk_colors.get(risk.lower(), "white")
        
        self.console.print()
        self.console.print(Panel(
            f"[bold]Action:[/bold] {action}\n"
            f"[bold]Args:[/bold] {args}\n"
            f"[bold]Risk:[/bold] [{risk_color}]{risk}[/{risk_color}]\n"
            f"[bold]Rationale:[/bold] {rationale}",
            title="⚠️  Approval Required",
            border_style="yellow",
        ))
