"""
Approval dialog for Agentic Browser GUI.

Modal dialog for approving risky actions.
"""

import json

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QGroupBox, QFormLayout,
    QMessageBox, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ..safety import RiskLevel


class ApprovalDialog(QDialog):
    """Dialog for approving risky browser actions."""
    
    # Result codes
    APPROVED = 1
    DENIED = 2
    EDITED = 3
    
    def __init__(
        self, 
        action: str, 
        args: dict, 
        risk_level: RiskLevel,
        rationale: str,
        parent=None
    ):
        super().__init__(parent)
        self.action = action
        self.args = args
        self.risk_level = risk_level
        self.rationale = rationale
        self.modified_action = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("⚠️ Action Requires Approval")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Warning header
        header = QLabel("This action requires your approval before executing.")
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)
        
        # Action details
        details_group = QGroupBox("Action Details")
        details_layout = QFormLayout(details_group)
        details_layout.setSpacing(8)
        
        # Action type
        action_label = QLabel(self.action)
        action_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        details_layout.addRow("Action:", action_label)
        
        # Arguments
        args_text = json.dumps(self.args, indent=2)
        args_label = QLabel(args_text)
        args_label.setWordWrap(True)
        args_label.setStyleSheet("font-family: monospace; background: #f5f5f5; padding: 8px; border-radius: 4px;")
        details_layout.addRow("Arguments:", args_label)
        
        # Risk level with color
        risk_colors = {
            RiskLevel.LOW: "#28a745",
            RiskLevel.MEDIUM: "#ffc107", 
            RiskLevel.HIGH: "#dc3545",
        }
        risk_color = risk_colors.get(self.risk_level, "#888")
        risk_label = QLabel(self.risk_level.value.upper())
        risk_label.setStyleSheet(f"color: {risk_color}; font-weight: bold; font-size: 13px;")
        details_layout.addRow("Risk Level:", risk_label)
        
        # Rationale
        rationale_label = QLabel(self.rationale)
        rationale_label.setWordWrap(True)
        details_layout.addRow("Rationale:", rationale_label)
        
        layout.addWidget(details_group)
        
        # Edit area (hidden by default)
        self.edit_group = QGroupBox("Edit Action JSON")
        self.edit_group.setVisible(False)
        edit_layout = QVBoxLayout(self.edit_group)
        
        self.edit_area = QTextEdit()
        self.edit_area.setFont(QFont("monospace"))
        self.edit_area.setMinimumHeight(100)
        self.edit_area.setPlainText(json.dumps({
            "action": self.action,
            "args": self.args,
        }, indent=2))
        edit_layout.addWidget(self.edit_area)
        
        layout.addWidget(self.edit_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.edit_btn = QPushButton("Edit Action")
        button_layout.addWidget(self.edit_btn)
        
        button_layout.addStretch()
        
        self.deny_btn = QPushButton("Deny")
        self.deny_btn.setStyleSheet("background: #dc3545; color: white;")
        button_layout.addWidget(self.deny_btn)
        
        self.approve_btn = QPushButton("Approve")
        self.approve_btn.setStyleSheet("background: #28a745; color: white;")
        self.approve_btn.setDefault(True)
        button_layout.addWidget(self.approve_btn)
        
        layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.approve_btn.clicked.connect(self._on_approve)
        self.deny_btn.clicked.connect(self._on_deny)
        self.edit_btn.clicked.connect(self._on_edit)
    
    def _on_approve(self):
        """Approve the action."""
        if self.edit_group.isVisible():
            # Parse edited JSON
            try:
                edited = json.loads(self.edit_area.toPlainText())
                self.modified_action = edited
                self.done(self.EDITED)
            except json.JSONDecodeError as e:
                QMessageBox.warning(
                    self,
                    "Invalid JSON",
                    f"The edited action is not valid JSON:\n{e}"
                )
                return
        else:
            self.done(self.APPROVED)
    
    def _on_deny(self):
        """Deny the action."""
        self.done(self.DENIED)
    
    def _on_edit(self):
        """Toggle edit mode."""
        is_visible = self.edit_group.isVisible()
        self.edit_group.setVisible(not is_visible)
        
        if not is_visible:
            self.edit_btn.setText("Cancel Edit")
            self.approve_btn.setText("Approve Edited")
        else:
            self.edit_btn.setText("Edit Action")
            self.approve_btn.setText("Approve")
    
    def get_result(self) -> tuple[int, dict | None]:
        """Get the dialog result.
        
        Returns:
            Tuple of (result_code, modified_action_or_none)
        """
        return self.result(), self.modified_action
