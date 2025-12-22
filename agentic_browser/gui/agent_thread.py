"""
Agent Thread for QThread-based IPC.

Provides thread-safe communication between the GUI and the Agent Graph
using Qt Signals/Slots instead of stdout-based IPC.
"""

import queue
import time
import threading
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QThread, Signal


class AgentThread(QThread):
    """Background thread for agent graph execution.
    
    Uses typed signals for GUI updates instead of stdout-based IPC.
    Supports live steering via input_queue.
    """
    
    # Typed signals for GUI updates
    signal_log = Signal(dict)        # {type, content, timestamp, agent}
    signal_state = Signal(dict)      # Full state updates
    signal_usage = Signal(dict)      # {input_tokens, output_tokens, total_cost}
    signal_node = Signal(str)        # Current node name for graph highlighting
    signal_screenshot = Signal(bytes)  # Browser screenshot bytes
    signal_complete = Signal(bool, str)  # (success, final_answer)
    signal_error = Signal(str)       # Error message
    signal_terminal = Signal(str)    # Raw console output for Terminal tab
    signal_approval_needed = Signal(dict)  # {action, args, risk_level, reason} - triggers approval dialog
    
    def __init__(
        self,
        config,
        goal: str,
        max_steps: int = 30,
        parent=None,
    ):
        """Initialize agent thread.
        
        Args:
            config: AgentConfig instance
            goal: User's goal/request
            max_steps: Maximum steps allowed
            parent: Parent QObject
        """
        super().__init__(parent)
        self.config = config
        self.goal = goal
        self.max_steps = max_steps
        self.running = True
        
        # Thread-safe steering queue
        self.input_queue: queue.Queue = queue.Queue()
        
        # Runner instance (created in run thread)
        self._runner = None
        self._browser_manager = None
        
        # Screenshot capture control
        self._screenshot_enabled = True
        self._last_screenshot_time = 0
        self._screenshot_interval = 1.0  # seconds
    
    def inject_steering(self, message: str):
        """Inject user steering message into agent execution.
        
        This message will be picked up before the next node execution
        and injected as a high-priority SystemMessage.
        
        Args:
            message: User's steering/intervention message
        """
        self.input_queue.put({
            "type": "steering",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Emit log for UI feedback
        self.signal_log.emit({
            "type": "system",
            "content": f"⚡ STEERING INJECTED: '{message}'",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
    
    def abort(self):
        """Request graceful abort of agent execution."""
        self.running = False
        self._screenshot_enabled = False
        self.input_queue.put({
            "type": "abort",
            "content": "User requested abort",
            "timestamp": datetime.now().isoformat(),
        })
    
    def run(self):
        """Execute agent graph in background thread.
        
        This method runs in a separate thread. All GUI updates
        are done via signals to maintain thread safety.
        """
        import sys
        import io
        
        # Capture stdout to emit to terminal
        class StdoutCapture(io.StringIO):
            def __init__(self, signal_emit, original_stdout):
                super().__init__()
                self.signal_emit = signal_emit
                self.original = original_stdout
            
            def write(self, text):
                if text.strip():  # Skip empty lines
                    self.signal_emit(text)
                if self.original:
                    self.original.write(text)
                return len(text)
            
            def flush(self):
                if self.original:
                    self.original.flush()
        
        # Install stdout capture
        original_stdout = sys.stdout
        sys.stdout = StdoutCapture(self.signal_terminal.emit, original_stdout)
        
        try:
            # Import here to avoid circular imports
            from ..graph.main_graph import MultiAgentRunner
            from ..graph.browser_manager import LazyBrowserManager
            from ..os_tools import OSTools
            
            # Create runner with tools
            self._browser_manager = LazyBrowserManager(self.config)
            os_tools = OSTools(config=self.config)
            
            self._runner = MultiAgentRunner(
                config=self.config,
                browser_manager=self._browser_manager,
                os_tools=os_tools,
            )
            
            # Share input queue with runner
            self._runner.input_queue = self.input_queue
            
            self.signal_log.emit({
                "type": "system",
                "content": f"🚀 Starting agent: {self.goal}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            
            # Stream execution
            final_state = None
            for state_update in self._runner.stream(self.goal, self.max_steps):
                if not self.running:
                    self._handle_abort()
                    break
                
                final_state = state_update
                self._process_state_update(state_update)
                
                # Capture screenshot from within this thread (where Playwright lives)
                self._try_capture_screenshot()
            
            # Handle completion
            if self.running and final_state:
                final_answer = self._runner.get_result(final_state)
                self.signal_complete.emit(True, final_answer)
            
            # Stop screenshot capture before closing browser
            self._screenshot_enabled = False
            
            # Cleanup
            self._runner.cleanup()
            self._browser_manager.close()
            self._browser_manager = None
            
        except Exception as e:
            self._screenshot_enabled = False
            self.signal_error.emit(str(e))
            self.signal_complete.emit(False, str(e))
        finally:
            # Restore stdout
            sys.stdout = original_stdout
    
    def _try_capture_screenshot(self):
        """Capture screenshot if enough time has passed.
        
        This must be called from the same thread that created Playwright
        (the agent thread), not from the GUI thread.
        """
        if not self._screenshot_enabled:
            return
        
        current_time = time.time()
        if current_time - self._last_screenshot_time < self._screenshot_interval:
            return
        
        self._last_screenshot_time = current_time
        
        try:
            if self._browser_manager and self._browser_manager.is_browser_open():
                page = self._browser_manager._page
                if page:
                    screenshot = page.screenshot(type="png")
                    self.signal_screenshot.emit(screenshot)
        except Exception:
            # Browser may be closing or not ready, silently ignore
            pass
    
    def _process_state_update(self, state: dict):
        """Process state update and emit appropriate signals.
        
        Args:
            state: Current agent state dict
        """
        # Emit full state for StatePanel
        self.signal_state.emit(state)
        
        # Emit current node for graph highlighting
        current_node = state.get("current_node", "supervisor")
        self.signal_node.emit(current_node)
        
        # Emit token usage
        token_usage = state.get("token_usage", {})
        if token_usage:
            self.signal_usage.emit({
                "input_tokens": token_usage.get("input_tokens", 0),
                "output_tokens": token_usage.get("output_tokens", 0),
                "total_cost": token_usage.get("total_cost", 0.0),
            })
        
        # Emit step log - agents provide status/outcome/notes, not agent/action/rationale
        step_update = state.get("step_update", {})
        active_agent = state.get("active_agent", "unknown")
        
        if step_update:
            # Use active_agent from state, not step_update
            status = step_update.get("status", "")
            outcome = step_update.get("outcome", "")
            notes = step_update.get("notes", "")
            
            # Emit outcome as action
            if outcome:
                self.signal_log.emit({
                    "type": "action",
                    "content": f"[{status}] {outcome}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": active_agent,
                })
            
            # Emit notes as thought
            if notes:
                self.signal_log.emit({
                    "type": "thought",
                    "content": notes,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": active_agent,
                })
        
        # Emit error if present
        error = state.get("error")
        if error:
            self.signal_log.emit({
                "type": "error",
                "content": error,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
        
        # Check for pending approval (high-risk action)
        pending = state.get("pending_approval")
        if pending:
            # Emit signal to GUI and log
            self.signal_approval_needed.emit(pending)
            self.signal_log.emit({
                "type": "system",
                "content": f"⚠️ APPROVAL REQUIRED: {pending.get('action')} (Risk: {pending.get('risk_level', 'HIGH')})",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
        
        # Check for task completion
        if state.get("task_complete"):
            final_answer = state.get("final_answer", "")
            self.signal_log.emit({
                "type": "success",
                "content": final_answer or "Task completed",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
    
    def _handle_abort(self):
        """Handle graceful abort."""
        self._screenshot_enabled = False
        self.signal_log.emit({
            "type": "system",
            "content": "⏹️ Agent execution aborted by user",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        self.signal_complete.emit(False, "Aborted by user")

    def capture_browser_screenshot(self) -> Optional[bytes]:
        """DEPRECATED: Screenshot capture now happens internally.
        
        This method is kept for compatibility but does nothing.
        Screenshots are captured from within the agent thread to avoid
        cross-thread Playwright/greenlet errors.
        """
        # Do nothing - screenshots are captured internally in _try_capture_screenshot()
        return None
