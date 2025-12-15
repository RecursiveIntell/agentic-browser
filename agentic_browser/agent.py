"""
Agent core for Agentic Browser.

Provides the main agent loop that orchestrates LLM, tools, and safety.
"""

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from .config import AgentConfig
from .llm_client import LLMClient, ActionResponse
from .logger import RunLogger
from .safety import SafetyClassifier, ApprovalPrompt, RiskLevel
from .tools import BrowserTools, ToolResult
from .utils import format_action_for_history


@dataclass
class PageState:
    """Current state of the browser page."""
    goal: str
    current_url: str
    page_title: str
    visible_text: str
    top_links: list[dict[str, str]]
    recent_history: list[dict[str, Any]]
    extracted_data: dict[str, Any]
    additional_context: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM."""
        return {
            "goal": self.goal,
            "current_url": self.current_url,
            "page_title": self.page_title,
            "visible_text": self.visible_text,
            "top_links": self.top_links,
            "recent_history": self.recent_history,
            "extracted_data": self.extracted_data,
            "additional_context": self.additional_context,
        }


@dataclass
class AgentResult:
    """Result of running the agent."""
    success: bool
    final_answer: Optional[str]
    steps_taken: int
    run_dir: Path
    error: Optional[str] = None


class BrowserAgent:
    """Main browser agent that orchestrates the automation."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the browser agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger: Optional[RunLogger] = None
        self.llm_client: Optional[LLMClient] = None
        self.safety = SafetyClassifier()
        self.approval = ApprovalPrompt()
        
        # Browser instances (set during run)
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._tools: Optional[BrowserTools] = None
        
        # State tracking
        self._history: list[dict[str, Any]] = []
        self._extracted_data: dict[str, Any] = {}
        self._action_counts: dict[str, int] = {}  # For loop detection
        
    def run(self) -> AgentResult:
        """Run the agent to accomplish the goal.
        
        Returns:
            AgentResult with outcome details
        """
        # Ensure directories exist
        self.config.ensure_directories()
        
        # Initialize logger
        self.logger = RunLogger(self.config.goal)
        self.logger.print_header()
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.config)
        
        try:
            return self._run_with_browser()
        finally:
            # Clean up
            if self.llm_client:
                self.llm_client.close()
    
    def _run_with_browser(self) -> AgentResult:
        """Run the agent with browser context."""
        with sync_playwright() as playwright:
            self._playwright = playwright
            
            # Launch browser
            self._browser = playwright.chromium.launch(
                headless=self.config.headless,
            )
            
            # Create persistent context
            context_options = {
                "viewport": {"width": 1280, "height": 800},
                "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
            
            if self.config.no_persist:
                # Use a temporary context
                self._context = self._browser.new_context(**context_options)
            else:
                # Use persistent context
                profile_dir = self.config.profile_dir
                
                # Handle fresh profile
                if self.config.fresh_profile and profile_dir.exists():
                    shutil.rmtree(profile_dir)
                    profile_dir.mkdir(parents=True, exist_ok=True)
                
                self._context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    headless=self.config.headless,
                    **context_options,
                )
            
            # Start tracing if enabled
            if self.config.enable_tracing:
                self._context.tracing.start(screenshots=True, snapshots=True)
            
            # Create page
            if hasattr(self._context, 'pages') and self._context.pages:
                self._page = self._context.pages[0]
            else:
                self._page = self._context.new_page()
            
            # Initialize tools
            self._tools = BrowserTools(
                self._page,
                default_timeout=self.config.action_timeout,
                screenshots_dir=self.logger.screenshots_dir,
            )
            
            try:
                return self._main_loop()
            finally:
                # Stop tracing
                if self.config.enable_tracing:
                    trace_path = Path(tempfile.mktemp(suffix=".zip"))
                    self._context.tracing.stop(path=str(trace_path))
                    self.logger.save_trace(trace_path)
                
                # Close context and browser
                if self._context:
                    self._context.close()
                if self._browser and not self.config.no_persist:
                    self._browser.close()
    
    def _main_loop(self) -> AgentResult:
        """Main agent loop.
        
        Returns:
            AgentResult
        """
        step = 0
        final_answer = None
        error = None
        
        while step < self.config.max_steps:
            step += 1
            self._tools.set_step(step)
            
            try:
                # Get current page state
                state = self._get_page_state()
                
                # Get next action from LLM
                try:
                    action_response = self.llm_client.get_next_action(state.to_dict())
                except ValueError as e:
                    # LLM failed to provide valid response
                    self.logger.print_error(str(e))
                    error = str(e)
                    break
                
                # Log the step
                self.logger.print_step(
                    action_response.action,
                    action_response.args,
                    action_response.rationale,
                    action_response.risk,
                    action_response.requires_approval,
                )
                
                # Check for loop detection
                if self._is_stuck(action_response):
                    self.logger.print_error("Detected loop - same action repeated 3 times")
                    state.additional_context = (
                        "WARNING: You are stuck in a loop repeating the same action. "
                        "Try a completely different approach."
                    )
                    # Take a recovery screenshot
                    self._tools.screenshot("stuck_recovery")
                    continue
                
                # Safety check
                risk_level = self.safety.classify_action(
                    action_response.action,
                    action_response.args,
                    state.current_url,
                    state.visible_text,
                )
                
                requires_approval = self.safety.should_require_approval(
                    risk_level,
                    action_response.requires_approval,
                    self.config.auto_approve,
                )
                
                # Handle approval if needed
                if requires_approval:
                    self.logger.print_approval_prompt(
                        action_response.action,
                        action_response.args,
                        risk_level.value,
                        action_response.rationale,
                    )
                    
                    approved, modified = self.approval.request_approval(
                        action_response.action,
                        action_response.args,
                        risk_level,
                        action_response.rationale,
                    )
                    
                    if not approved:
                        if modified:
                            # Use modified action
                            action_response = ActionResponse(
                                action=modified.get("action", action_response.action),
                                args=modified.get("args", action_response.args),
                                rationale="User-modified action",
                                risk=action_response.risk,
                                requires_approval=False,
                            )
                        else:
                            # Denial - get guidance and continue
                            guidance = self.approval.notify_denial()
                            state.additional_context = (
                                f"Previous action was DENIED by user. "
                                f"User guidance: {guidance or 'None provided'}"
                            )
                            continue
                
                # Execute the action
                if action_response.action == "done":
                    final_answer = action_response.final_answer or "Task completed"
                    self.logger.print_final_answer(final_answer)
                    
                    # Log the step
                    self.logger.log_step(
                        state.to_dict(),
                        action_response.model_dump(),
                        {"success": True, "message": "Task completed"},
                    )
                    break
                
                result = self._tools.execute(
                    action_response.action,
                    action_response.args,
                )
                
                # Log result
                self.logger.print_result(result.success, result.message)
                
                # Log the step
                self.logger.log_step(
                    state.to_dict(),
                    action_response.model_dump(),
                    result.to_dict(),
                    error=result.message if not result.success else None,
                )
                
                # Update history
                self._add_to_history(
                    action_response.action,
                    action_response.args,
                    result.message,
                )
                
                # Store extracted data
                if result.data and action_response.action in ("extract", "extract_visible_text"):
                    key = f"extract_{len(self._extracted_data)}"
                    self._extracted_data[key] = result.data
                
                # Handle failures
                if not result.success:
                    # Try recovery
                    state.additional_context = f"Previous action failed: {result.message}"
                    # Take a screenshot for debugging
                    self._tools.screenshot("error_recovery")
                    
            except Exception as e:
                error = f"Unexpected error: {type(e).__name__}: {str(e)}"
                self.logger.print_error(error)
                break
        
        # Print summary
        self.logger.print_summary()
        
        if step >= self.config.max_steps and not final_answer:
            error = "Reached maximum step limit"
        
        return AgentResult(
            success=final_answer is not None,
            final_answer=final_answer,
            steps_taken=step,
            run_dir=self.logger.run_path,
            error=error,
        )
    
    def _get_page_state(self) -> PageState:
        """Get the current page state."""
        page_state = self._tools.get_page_state(
            max_text_chars=self.config.visible_text_max_chars,
            max_links=self.config.max_links,
        )
        
        return PageState(
            goal=self.config.goal,
            current_url=page_state["current_url"],
            page_title=page_state["page_title"],
            visible_text=page_state["visible_text"],
            top_links=page_state["top_links"],
            recent_history=self._get_recent_history(),
            extracted_data=self._extracted_data,
        )
    
    def _get_recent_history(self) -> list[dict[str, Any]]:
        """Get the recent action history."""
        return self._history[-self.config.history_length:]
    
    def _add_to_history(
        self,
        action: str,
        args: dict[str, Any],
        result: str,
    ) -> None:
        """Add an action to the history."""
        entry = format_action_for_history(action, args, result)
        self._history.append(entry)
    
    def _is_stuck(self, action_response: ActionResponse) -> bool:
        """Check if the agent is stuck in a loop.
        
        Args:
            action_response: Current action response
            
        Returns:
            True if stuck (same action repeated too many times)
        """
        # Create a key for this action
        key = f"{action_response.action}:{action_response.args}"
        
        # Increment count
        self._action_counts[key] = self._action_counts.get(key, 0) + 1
        
        # Check if stuck
        if self._action_counts[key] >= self.config.max_repeat_actions:
            # Reset to allow recovery
            self._action_counts[key] = 0
            return True
        
        return False


def run_agent(config: AgentConfig) -> AgentResult:
    """Convenience function to run the agent.
    
    Args:
        config: Agent configuration
        
    Returns:
        AgentResult
    """
    agent = BrowserAgent(config)
    return agent.run()
