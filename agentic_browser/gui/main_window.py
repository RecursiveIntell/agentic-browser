"""
Main window for Agentic Browser GUI.

Provides the primary application interface.
"""

import sys
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QStatusBar, QToolBar, QSplitter, QFrame,
    QMessageBox, QProgressBar,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QAction, QFont, QColor, QIcon

from ..config import AgentConfig
from ..agent import BrowserAgent, AgentResult
from ..safety import RiskLevel
from ..settings_store import SettingsStore, get_settings
from ..providers import Provider

from .settings_dialog import SettingsDialog
from .approval_dialog import ApprovalDialog


class AgentWorker(QThread):
    """Background worker thread for running the agent."""
    
    step_started = Signal(int, str, dict, str, str)  # step, action, args, rationale, risk
    step_completed = Signal(int, bool, str)  # step, success, message
    approval_needed = Signal(int, str, dict, str, str)  # step, action, args, risk, rationale
    agent_finished = Signal(bool, str, str)  # success, final_answer, error
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self._agent: Optional[BrowserAgent] = None
        self._stop_requested = False
        self._approval_response = None
        self._approval_event = None
    
    def run(self):
        """Run the agent in background."""
        from threading import Event
        self._approval_event = Event()
        
        try:
            # Create agent with GUI callbacks
            self._agent = BrowserAgent(self.config)
            
            # Override the agent's approval method to use GUI
            original_run = self._agent._main_loop
            
            # We'll use a simpler approach - just run the agent
            result = self._agent.run()
            
            self.agent_finished.emit(
                result.success,
                result.final_answer or "",
                result.error or "",
            )
            
        except Exception as e:
            self.agent_finished.emit(False, "", str(e))
    
    def request_stop(self):
        """Request the agent to stop."""
        self._stop_requested = True


class StepItem(QFrame):
    """Custom widget for displaying a step in the log."""
    
    def __init__(self, step: int, action: str, args: dict, risk: str, parent=None):
        super().__init__(parent)
        self.step = step
        self.action = action
        self.args = args
        self.risk = risk
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Step number
        step_label = QLabel(f"#{self.step}")
        step_label.setFixedWidth(35)
        step_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(step_label)
        
        # Action
        action_label = QLabel(self.action)
        action_label.setFixedWidth(100)
        action_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        layout.addWidget(action_label)
        
        # Args summary
        args_text = str(self.args)[:60] + "..." if len(str(self.args)) > 60 else str(self.args)
        args_label = QLabel(args_text)
        args_label.setStyleSheet("color: #444;")
        layout.addWidget(args_label, 1)
        
        # Risk badge
        risk_colors = {"low": "#28a745", "medium": "#ffc107", "high": "#dc3545"}
        color = risk_colors.get(self.risk.lower(), "#888")
        risk_label = QLabel(self.risk.upper())
        risk_label.setFixedWidth(60)
        risk_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        risk_label.setStyleSheet(f"background: {color}; color: white; border-radius: 3px; padding: 2px;")
        layout.addWidget(risk_label)
        
        # Status indicator
        self.status_label = QLabel("⏳")
        self.status_label.setFixedWidth(25)
        layout.addWidget(self.status_label)
    
    def set_result(self, success: bool, message: str = ""):
        """Update the step result."""
        if success:
            self.status_label.setText("✅")
            self.status_label.setToolTip(message)
        else:
            self.status_label.setText("❌")
            self.status_label.setToolTip(message)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.store = SettingsStore()
        self._worker: Optional[AgentWorker] = None
        self._step_items: dict[int, StepItem] = {}
        
        self._setup_ui()
        self._connect_signals()
        self._update_status_bar()
    
    def _setup_ui(self):
        """Set up the main window UI."""
        self.setWindowTitle("Agentic Browser")
        self.setMinimumSize(900, 700)
        
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
        self.goal_edit.setPlaceholderText("Enter your goal, e.g., 'Open example.com and tell me the title'")
        self.goal_edit.setStyleSheet("font-size: 14px; padding: 8px;")
        header_layout.addWidget(self.goal_edit, 1)
        
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #28a745;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #218838;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """)
        header_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⬛ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #dc3545;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #c82333;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """)
        header_layout.addWidget(self.stop_btn)
        
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px 16px;
            }
        """)
        header_layout.addWidget(self.settings_btn)
        
        layout.addLayout(header_layout)
        
        # Splitter for log and result
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Step log
        log_frame = QFrame()
        log_frame.setStyleSheet("background: white; border: 1px solid #ddd; border-radius: 4px;")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_header = QLabel("Step Log")
        log_header.setStyleSheet("font-weight: bold; padding: 8px; background: #f8f9fa; border-bottom: 1px solid #ddd;")
        log_layout.addWidget(log_header)
        
        self.step_list = QListWidget()
        self.step_list.setStyleSheet("border: none;")
        log_layout.addWidget(self.step_list)
        
        splitter.addWidget(log_frame)
        
        # Result area
        result_frame = QFrame()
        result_frame.setStyleSheet("background: white; border: 1px solid #ddd; border-radius: 4px;")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(0, 0, 0, 0)
        
        result_header = QLabel("Result")
        result_header.setStyleSheet("font-weight: bold; padding: 8px; background: #f8f9fa; border-bottom: 1px solid #ddd;")
        result_layout.addWidget(result_header)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("border: none; font-size: 13px; padding: 8px;")
        self.result_text.setPlaceholderText("Results will appear here after the agent completes...")
        result_layout.addWidget(self.result_text)
        
        splitter.addWidget(result_frame)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.provider_label = QLabel()
        self.status_bar.addPermanentWidget(self.provider_label)
        
        self.step_count_label = QLabel("Steps: 0")
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
            from ..providers import PROVIDER_DISPLAY_NAMES
            provider_name = PROVIDER_DISPLAY_NAMES.get(provider, settings.provider)
        except ValueError:
            provider_name = settings.provider
        
        model = settings.model or "default"
        self.provider_label.setText(f"Provider: {provider_name} | Model: {model}")
    
    def _on_run(self):
        """Start the agent."""
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
        self.step_list.clear()
        self._step_items.clear()
        self.result_text.clear()
        self.step_count_label.setText("Steps: 0")
        
        # Build config
        config = AgentConfig(
            goal=goal,
            profile_name=settings.profile_name,
            headless=settings.headless,
            max_steps=settings.max_steps,
            auto_approve=settings.auto_approve,
            model_endpoint=provider_config.endpoint,
            model=provider_config.effective_model,
            api_key=provider_config.api_key,
        )
        
        # Update UI state
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.goal_edit.setEnabled(False)
        self.settings_btn.setEnabled(False)
        
        # Start worker
        self._worker = AgentWorker(config)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_completed.connect(self._on_step_completed)
        self._worker.agent_finished.connect(self._on_agent_finished)
        self._worker.start()
        
        self.status_bar.showMessage("Agent running...")
    
    def _on_stop(self):
        """Stop the agent."""
        if self._worker:
            self._worker.request_stop()
            self.status_bar.showMessage("Stopping agent...")
    
    def _on_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self._update_status_bar()
    
    @Slot(int, str, dict, str, str)
    def _on_step_started(self, step: int, action: str, args: dict, rationale: str, risk: str):
        """Handle step started."""
        item = StepItem(step, action, args, risk)
        list_item = QListWidgetItem()
        list_item.setSizeHint(item.sizeHint())
        
        self.step_list.addItem(list_item)
        self.step_list.setItemWidget(list_item, item)
        self.step_list.scrollToBottom()
        
        self._step_items[step] = item
        self.step_count_label.setText(f"Steps: {step}")
    
    @Slot(int, bool, str)
    def _on_step_completed(self, step: int, success: bool, message: str):
        """Handle step completed."""
        if step in self._step_items:
            self._step_items[step].set_result(success, message)
    
    @Slot(bool, str, str)
    def _on_agent_finished(self, success: bool, final_answer: str, error: str):
        """Handle agent finished."""
        # Reset UI state
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.goal_edit.setEnabled(True)
        self.settings_btn.setEnabled(True)
        
        if success:
            self.result_text.setStyleSheet("border: none; font-size: 13px; padding: 8px; color: #155724; background: #d4edda;")
            self.result_text.setText(final_answer)
            self.status_bar.showMessage("Agent completed successfully!", 5000)
        else:
            self.result_text.setStyleSheet("border: none; font-size: 13px; padding: 8px; color: #721c24; background: #f8d7da;")
            self.result_text.setText(error or "Agent did not complete the goal.")
            self.status_bar.showMessage("Agent finished with errors", 5000)
        
        self._worker = None
    
    def closeEvent(self, event):
        """Handle window close."""
        # Save window size
        self.store.update(
            window_width=self.width(),
            window_height=self.height(),
        )
        
        # Stop worker if running
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)
        
        event.accept()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Agentic Browser")
    app.setStyle("Fusion")
    
    # Apply dark-friendly styling
    app.setStyleSheet("""
        QMainWindow {
            background: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 12px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    return app.exec()
