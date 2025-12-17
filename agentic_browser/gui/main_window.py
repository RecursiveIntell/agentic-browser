"""
Main window for Agentic Browser GUI.

Provides the primary application interface with dark theme.
"""

import sys
import re
import subprocess
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QStatusBar, QSplitter, QFrame,
    QMessageBox, QScrollArea,
)
from PySide6.QtCore import Qt, QProcess, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QTextCursor

from ..settings_store import SettingsStore, get_settings
from ..providers import Provider, PROVIDER_DISPLAY_NAMES

from .settings_dialog import SettingsDialog


# Dark theme colors
DARK_BG = "#1e1e1e"
DARK_SURFACE = "#252526"
DARK_SURFACE_LIGHT = "#2d2d30"
DARK_BORDER = "#3e3e42"
DARK_TEXT = "#cccccc"
DARK_TEXT_DIM = "#808080"
DARK_ACCENT = "#0e639c"
DARK_ACCENT_HOVER = "#1177bb"
DARK_SUCCESS = "#4ec9b0"
DARK_WARNING = "#dcdcaa"
DARK_ERROR = "#f14c4c"
DARK_INFO = "#569cd6"


DARK_STYLESHEET = f"""
    QMainWindow {{
        background: {DARK_BG};
    }}
    QWidget {{
        background: {DARK_BG};
        color: {DARK_TEXT};
        font-family: 'Segoe UI', 'Ubuntu', sans-serif;
    }}
    QLabel {{
        background: transparent;
        color: {DARK_TEXT};
    }}
    QLineEdit {{
        background: {DARK_SURFACE};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        padding: 8px;
        color: {DARK_TEXT};
    }}
    QLineEdit:focus {{
        border-color: {DARK_ACCENT};
    }}
    QPushButton {{
        background: {DARK_SURFACE_LIGHT};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        padding: 8px 16px;
        color: {DARK_TEXT};
    }}
    QPushButton:hover {{
        background: {DARK_BORDER};
    }}
    QPushButton:disabled {{
        background: {DARK_SURFACE};
        color: {DARK_TEXT_DIM};
    }}
    QTextEdit {{
        background: {DARK_SURFACE};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        color: {DARK_TEXT};
        font-family: 'Consolas', 'Ubuntu Mono', monospace;
        font-size: 12px;
    }}
    QListWidget {{
        background: {DARK_SURFACE};
        border: none;
        color: {DARK_TEXT};
    }}
    QListWidget::item {{
        padding: 4px;
    }}
    QListWidget::item:selected {{
        background: {DARK_ACCENT};
    }}
    QFrame {{
        background: {DARK_SURFACE};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
    }}
    QStatusBar {{
        background: {DARK_SURFACE_LIGHT};
        color: {DARK_TEXT_DIM};
    }}
    QSplitter::handle {{
        background: {DARK_BORDER};
    }}
    QGroupBox {{
        font-weight: bold;
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 12px;
        color: {DARK_TEXT};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }}
    QComboBox {{
        background: {DARK_SURFACE};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        padding: 6px;
        color: {DARK_TEXT};
    }}
    QComboBox:hover {{
        border-color: {DARK_ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
    }}
    QComboBox QAbstractItemView {{
        background: {DARK_SURFACE};
        border: 1px solid {DARK_BORDER};
        color: {DARK_TEXT};
        selection-background-color: {DARK_ACCENT};
    }}
    QSpinBox {{
        background: {DARK_SURFACE};
        border: 1px solid {DARK_BORDER};
        border-radius: 4px;
        padding: 6px;
        color: {DARK_TEXT};
    }}
    QCheckBox {{
        color: {DARK_TEXT};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
    }}
    QScrollBar:vertical {{
        background: {DARK_SURFACE};
        width: 12px;
        border-radius: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: {DARK_BORDER};
        border-radius: 6px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QDialog {{
        background: {DARK_BG};
    }}
"""


class MainWindow(QMainWindow):
    """Main application window with dark theme."""
    
    def __init__(self):
        super().__init__()
        self.store = SettingsStore()
        self._process: Optional[QProcess] = None
        self._current_step = 0
        self._output_buffer = ""
        
        self._setup_ui()
        self._connect_signals()
        self._update_status_bar()
    
    def _setup_ui(self):
        """Set up the main window UI."""
        self.setWindowTitle("Agentic Browser")
        self.setMinimumSize(1000, 750)
        
        # Load window size from settings
        settings = self.store.settings
        self.resize(settings.window_width, settings.window_height)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header with goal input
        header_layout = QHBoxLayout()
        
        goal_label = QLabel("Goal:")
        goal_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(goal_label)
        
        self.goal_edit = QLineEdit()
        self.goal_edit.setPlaceholderText("Enter your goal, e.g., 'Search Google for Python tutorials'")
        self.goal_edit.setStyleSheet("font-size: 14px; padding: 10px;")
        header_layout.addWidget(self.goal_edit, 1)
        
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {DARK_SUCCESS};
                color: #1e1e1e;
                font-weight: bold;
                font-size: 14px;
                padding: 10px 24px;
                border-radius: 4px;
                border: none;
            }}
            QPushButton:hover {{
                background: #5fd9c0;
            }}
            QPushButton:disabled {{
                background: {DARK_SURFACE_LIGHT};
                color: {DARK_TEXT_DIM};
            }}
        """)
        header_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⬛ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {DARK_ERROR};
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px 24px;
                border-radius: 4px;
                border: none;
            }}
            QPushButton:hover {{
                background: #f77;
            }}
            QPushButton:disabled {{
                background: {DARK_SURFACE_LIGHT};
                color: {DARK_TEXT_DIM};
            }}
        """)
        header_layout.addWidget(self.stop_btn)
        
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px;
                padding: 10px 20px;
                background: {DARK_SURFACE_LIGHT};
            }}
        """)
        header_layout.addWidget(self.settings_btn)
        
        layout.addLayout(header_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Output log (main area)
        output_frame = QFrame()
        output_layout = QVBoxLayout(output_frame)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(0)
        
        output_header = QLabel("  📋 Agent Output")
        output_header.setStyleSheet(f"""
            font-weight: bold; 
            padding: 10px; 
            background: {DARK_SURFACE_LIGHT}; 
            border-bottom: 1px solid {DARK_BORDER};
            border-radius: 4px 4px 0 0;
        """)
        output_layout.addWidget(output_header)
        
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setStyleSheet(f"""
            border: none;
            border-radius: 0 0 4px 4px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.5;
        """)
        self.output_log.setPlaceholderText("Agent output will appear here...\n\nTip: The agent will control a browser window. Keep it visible to see what's happening.")
        output_layout.addWidget(self.output_log)
        
        splitter.addWidget(output_frame)
        
        # Status/Summary area
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(0)
        
        status_header = QLabel("  📊 Status")
        status_header.setStyleSheet(f"""
            font-weight: bold; 
            padding: 10px; 
            background: {DARK_SURFACE_LIGHT}; 
            border-bottom: 1px solid {DARK_BORDER};
            border-radius: 4px 4px 0 0;
        """)
        status_layout.addWidget(status_header)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet(f"""
            border: none;
            border-radius: 0 0 4px 4px;
            padding: 12px;
            font-size: 13px;
        """)
        self.status_text.setPlaceholderText("Status and results will appear here...")
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)
        
        splitter.addWidget(status_frame)
        splitter.setSizes([500, 150])
        
        layout.addWidget(splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.provider_label = QLabel()
        self.status_bar.addPermanentWidget(self.provider_label)
        
        self.step_count_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.step_count_label)
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.settings_btn.clicked.connect(self._on_settings)
        self.goal_edit.returnPressed.connect(self._on_run)
    
    def _update_status_bar(self):
        """Update the status bar with current settings."""
        settings = self.store.settings
        try:
            provider = Provider(settings.provider)
            provider_name = PROVIDER_DISPLAY_NAMES.get(provider, settings.provider)
        except ValueError:
            provider_name = settings.provider
        
        model = settings.model or "default"
        self.provider_label.setText(f"Provider: {provider_name} | Model: {model}")
    
    def _log(self, message: str, level: str = "info"):
        """Add a message to the output log with color coding."""
        colors = {
            "info": DARK_TEXT,
            "success": DARK_SUCCESS,
            "warning": DARK_WARNING,
            "error": DARK_ERROR,
            "action": DARK_INFO,
            "dim": DARK_TEXT_DIM,
        }
        color = colors.get(level, DARK_TEXT)
        
        cursor = self.output_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Insert colored text
        format = cursor.charFormat()
        format.setForeground(QColor(color))
        cursor.setCharFormat(format)
        cursor.insertText(message + "\n")
        
        self.output_log.setTextCursor(cursor)
        self.output_log.ensureCursorVisible()
    
    def _on_run(self):
        """Start the agent as a subprocess."""
        goal = self.goal_edit.text().strip()
        if not goal:
            QMessageBox.warning(self, "No Goal", "Please enter a goal first.")
            self.goal_edit.setFocus()
            return
        
        # Validate settings
        settings = self.store.settings
        provider_config = settings.get_provider_config()
        is_valid, error = provider_config.validate()
        
        if not is_valid:
            result = QMessageBox.question(
                self,
                "Configuration Required",
                f"{error}\n\nWould you like to open settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._on_settings()
            return
        
        # Clear previous run
        self.output_log.clear()
        self.status_text.clear()
        self._current_step = 0
        self._output_buffer = ""
        
        # Log start
        self._log(f"🎯 Goal: {goal}", "info")
        self._log(f"🔌 Provider: {provider_config.display_name}", "dim")
        self._log(f"🤖 Model: {provider_config.effective_model}", "dim")
        self._log(f"📁 Profile: {settings.profile_name}", "dim")
        if not settings.auto_approve:
            self._log("⚠️  Auto-approve OFF: Approval prompts will appear in TERMINAL", "warning")
        self._log("─" * 50, "dim")
        self._log("🚀 Starting agent...", "info")
        self._log("", "dim")
        
        # Update UI state
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.goal_edit.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.step_count_label.setText("Running...")
        
        # Build command
        cmd = [
            sys.executable, "-m", "agentic_browser", "run",
            goal,
            "--profile", settings.profile_name,
            "--max-steps", str(settings.max_steps),
            "--model-endpoint", provider_config.endpoint,
            "--model", provider_config.effective_model,
        ]
        
        # Only add auto-approve if explicitly enabled in settings
        # Otherwise use GUI IPC for approval dialogs
        if settings.auto_approve:
            cmd.append("--auto-approve")
        else:
            cmd.append("--gui-ipc")
        
        if settings.headless:
            cmd.append("--headless")
        
        # Add API key as environment variable
        env = dict(subprocess.os.environ)
        if provider_config.api_key:
            env["AGENTIC_BROWSER_API_KEY"] = provider_config.api_key
        
        # Log the command for debugging
        self._log(f"[DEBUG] Command: agentic-browser run \"{goal}\"", "dim")
        self._log(f"[DEBUG] Endpoint: {provider_config.endpoint}", "dim")
        self._log("", "dim")
        
        # Start process
        self._process = QProcess(self)
        self._process.setProcessEnvironment(self._create_env(env))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        
        # Start the command
        self._process.start(cmd[0], cmd[1:])
        
        self.status_bar.showMessage("Agent running... Keep the browser window visible!")
    
    def _create_env(self, env_dict: dict) -> "QProcessEnvironment":
        """Create QProcessEnvironment from dict."""
        from PySide6.QtCore import QProcessEnvironment
        qenv = QProcessEnvironment.systemEnvironment()
        for key, value in env_dict.items():
            qenv.insert(key, value)
        return qenv
    
    def _on_stop(self):
        """Stop the agent process."""
        if self._process:
            self._log("", "dim")
            self._log("⏹️ Stopping agent...", "warning")
            self._process.terminate()
            # Give it a moment to terminate gracefully
            QTimer.singleShot(2000, self._force_kill)
            self.status_bar.showMessage("Stopping agent...")
    
    def _force_kill(self):
        """Force kill the process if still running."""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._log("⚠️ Force killing agent process", "error")
            self._process.kill()
    
    def _on_stdout(self):
        """Handle stdout from the process."""
        if not self._process:
            return
        
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._output_buffer += data
        
        # Parse lines
        lines = self._output_buffer.split("\n")
        self._output_buffer = lines[-1]  # Keep incomplete line
        
        for line in lines[:-1]:
            self._parse_output_line(line)
    
    def _on_stderr(self):
        """Handle stderr from the process."""
        if not self._process:
            return
        
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        for line in data.strip().split("\n"):
            if line.strip():
                self._log(f"[stderr] {line}", "error")
    
    def _parse_output_line(self, line: str):
        """Parse a line of output to extract step info."""
        # Check for approval request (IPC protocol)
        if line.startswith("APPROVAL_REQUEST:"):
            self._handle_approval_request(line[17:])  # Skip prefix
            return
        
        # Strip ANSI codes for parsing
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_line = ansi_escape.sub('', line).strip()
        
        if not clean_line:
            return
        
        # Determine log level based on content
        level = "info"
        
        # Skip rich formatting lines
        if clean_line.startswith("─") or clean_line.startswith("━"):
            return
        if clean_line.startswith("│") or clean_line.startswith("┃"):
            clean_line = clean_line[1:].strip()
        
        # Detect patterns
        lower = clean_line.lower()
        
        if "error" in lower or "failed" in lower or "fatal" in lower:
            level = "error"
        elif "success" in lower or "completed" in lower or "✓" in clean_line:
            level = "success"
        elif "warning" in lower or "denied" in lower:
            level = "warning"
        elif "step" in lower or "action:" in lower:
            level = "action"
            # Update step count
            match = re.search(r"step\s*(\d+)", lower)
            if match:
                self._current_step = int(match.group(1))
                self.step_count_label.setText(f"Step {self._current_step}")
        elif "goal" in lower:
            level = "info"
        elif clean_line.startswith("[") and "]" in clean_line:
            level = "dim"
        
        # Log the line
        self._log(clean_line, level)
        
        # Check for final answer
        if "final answer" in lower or "goal accomplished" in lower:
            self.status_text.append(f"✅ {clean_line}")
    
    def _handle_approval_request(self, json_str: str):
        """Handle approval request from subprocess via IPC protocol."""
        import json as json_module
        from .approval_dialog import ApprovalDialog
        from ..safety import RiskLevel
        
        try:
            request = json_module.loads(json_str)
            
            # Check if this is a guidance request
            if request.get("type") == "guidance":
                # Show simple input dialog
                from PySide6.QtWidgets import QInputDialog
                text, ok = QInputDialog.getText(
                    self,
                    "Action Denied",
                    "Provide guidance for next action (or leave empty):",
                )
                guidance = text if ok else ""
                response = {"guidance": guidance}
                self._send_approval_response(response)
                return
            
            # Parse approval request
            action = request.get("action", "unknown")
            args = request.get("args", {})
            risk_str = request.get("risk_level", "low")
            rationale = request.get("rationale", "")
            
            # Convert risk level
            try:
                risk_level = RiskLevel(risk_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM
            
            self._log(f"⚠️ Approval requested for: {action}", "warning")
            
            # Show approval dialog
            dialog = ApprovalDialog(
                action=action,
                args=args,
                risk_level=risk_level,
                rationale=rationale,
                parent=self,
            )
            
            dialog.exec()
            result_code, modified = dialog.get_result()
            
            # Build response
            if result_code == ApprovalDialog.APPROVED:
                response = {"approved": True}
                self._log("✓ Action approved", "success")
            elif result_code == ApprovalDialog.EDITED:
                response = {"approved": True, "modified_action": modified}
                self._log("✓ Action approved (edited)", "success")
            else:
                response = {"approved": False}
                self._log("✗ Action denied", "warning")
            
            self._send_approval_response(response)
            
        except Exception as e:
            self._log(f"Error handling approval: {e}", "error")
            # Deny on error
            self._send_approval_response({"approved": False})
    
    def _send_approval_response(self, response: dict):
        """Send approval response to subprocess via stdin."""
        import json as json_module
        
        if not self._process:
            return
        
        response_line = f"APPROVAL_RESPONSE:{json_module.dumps(response)}\n"
        self._process.write(response_line.encode("utf-8"))
    
    
    def _on_finished(self, exit_code: int, exit_status):
        """Handle process finished."""
        # Reset UI state
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.goal_edit.setEnabled(True)
        self.settings_btn.setEnabled(True)
        
        self._log("", "dim")
        if exit_code == 0:
            self._log("✅ Agent completed successfully!", "success")
            self.status_bar.showMessage("Agent completed!", 5000)
            self.step_count_label.setText("Done")
        else:
            self._log(f"❌ Agent exited with code {exit_code}", "error")
            self.status_bar.showMessage(f"Agent exited with code {exit_code}", 5000)
            self.step_count_label.setText("Failed")
        
        self._process = None
    
    def _on_error(self, error):
        """Handle process error."""
        error_msgs = {
            QProcess.ProcessError.FailedToStart: "Failed to start - is agentic-browser installed?",
            QProcess.ProcessError.Crashed: "Process crashed",
            QProcess.ProcessError.Timedout: "Process timed out",
            QProcess.ProcessError.WriteError: "Write error",
            QProcess.ProcessError.ReadError: "Read error",
        }
        msg = error_msgs.get(error, f"Unknown error: {error}")
        self._log(f"❌ Process error: {msg}", "error")
        self.status_bar.showMessage("Agent error occurred", 5000)
    
    def _on_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self._update_status_bar()
    
    def closeEvent(self, event):
        """Handle window close."""
        # Save window size
        self.store.update(
            window_width=self.width(),
            window_height=self.height(),
        )
        
        # Stop process if running
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(3000)
            if self._process.state() != QProcess.ProcessState.NotRunning:
                self._process.kill()
        
        event.accept()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Agentic Browser")
    app.setStyle("Fusion")
    
    # Apply dark theme
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Set dark palette for native widgets
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(DARK_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(DARK_TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(DARK_SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_SURFACE_LIGHT))
    palette.setColor(QPalette.ColorRole.Text, QColor(DARK_TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(DARK_SURFACE_LIGHT))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(DARK_TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(DARK_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    return app.exec()
