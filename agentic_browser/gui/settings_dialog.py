"""
Settings dialog for Agentic Browser GUI.

Provides provider configuration and agent settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QPushButton, QGroupBox, QWidget, QCompleter,
    QMessageBox, QApplication,
)
from PySide6.QtCore import Qt, QThread, Signal

from ..providers import (
    Provider, ProviderConfig, 
    PROVIDER_DISPLAY_NAMES, PROVIDER_REQUIRES_API_KEY,
    PROVIDER_DEFAULT_MODELS, PROVIDER_MODEL_SUGGESTIONS,
    PROVIDER_ENDPOINTS,
)
from ..settings_store import SettingsStore, Settings
from ..model_fetcher import fetch_models


class ModelFetchWorker(QThread):
    """Background worker for fetching models."""
    
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, provider: Provider, api_key: str, custom_endpoint: str = None):
        super().__init__()
        self.provider = provider
        self.api_key = api_key
        self.custom_endpoint = custom_endpoint
    
    def run(self):
        try:
            models = fetch_models(
                self.provider,
                self.api_key,
                self.custom_endpoint,
            )
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))


class SettingsDialog(QDialog):
    """Settings dialog for configuring the agent."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.store = SettingsStore()
        self._fetch_worker = None
        self._setup_ui()
        self._load_settings()
        self._connect_signals()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Provider settings group
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout(provider_group)
        provider_layout.setSpacing(12)
        
        # Provider dropdown
        self.provider_combo = QComboBox()
        for provider in Provider:
            self.provider_combo.addItem(
                PROVIDER_DISPLAY_NAMES[provider],
                provider.value
            )
        provider_layout.addRow("Provider:", self.provider_combo)
        
        # API Key
        api_key_layout = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Not required for LM Studio")
        api_key_layout.addWidget(self.api_key_edit)
        
        self.show_key_btn = QPushButton("Show")
        self.show_key_btn.setFixedWidth(60)
        self.show_key_btn.setCheckable(True)
        api_key_layout.addWidget(self.show_key_btn)
        provider_layout.addRow("API Key:", api_key_layout)
        
        # Model with refresh button
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        model_layout.addWidget(self.model_combo, 1)
        
        self.refresh_models_btn = QPushButton("🔄 Refresh")
        self.refresh_models_btn.setToolTip("Fetch available models from the provider")
        self.refresh_models_btn.setFixedWidth(90)
        model_layout.addWidget(self.refresh_models_btn)
        provider_layout.addRow("Model:", model_layout)
        
        # Model status label
        self.model_status_label = QLabel()
        self.model_status_label.setStyleSheet("color: #888; font-size: 11px;")
        provider_layout.addRow("", self.model_status_label)
        
        # Custom endpoint
        self.endpoint_edit = QLineEdit()
        self.endpoint_edit.setPlaceholderText("Leave empty to use default")
        provider_layout.addRow("Custom Endpoint:", self.endpoint_edit)
        
        # Default endpoint label
        self.default_endpoint_label = QLabel()
        self.default_endpoint_label.setStyleSheet("color: #888; font-size: 11px;")
        provider_layout.addRow("", self.default_endpoint_label)
        
        layout.addWidget(provider_group)
        
        # Agent settings group
        agent_group = QGroupBox("Agent Settings")
        agent_layout = QFormLayout(agent_group)
        agent_layout.setSpacing(12)
        
        # Profile name
        self.profile_edit = QLineEdit()
        self.profile_edit.setPlaceholderText("default")
        agent_layout.addRow("Profile Name:", self.profile_edit)
        
        # Max steps
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 100)
        self.max_steps_spin.setValue(30)
        agent_layout.addRow("Max Steps:", self.max_steps_spin)
        
        # Checkboxes
        self.headless_check = QCheckBox("Run browser in headless mode")
        agent_layout.addRow("", self.headless_check)
        
        self.auto_approve_check = QCheckBox("Auto-approve medium-risk actions")
        agent_layout.addRow("", self.auto_approve_check)
        
        layout.addWidget(agent_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.reset_btn = QPushButton("Reset to Defaults")
        button_layout.addWidget(self.reset_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setDefault(True)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.show_key_btn.toggled.connect(self._on_show_key_toggled)
        self.refresh_models_btn.clicked.connect(self._on_refresh_models)
        self.save_btn.clicked.connect(self._on_save)
        self.cancel_btn.clicked.connect(self.reject)
        self.reset_btn.clicked.connect(self._on_reset)
    
    def _load_settings(self):
        """Load current settings into the UI."""
        settings = self.store.settings
        
        # Find provider index
        provider_index = 0
        for i in range(self.provider_combo.count()):
            if self.provider_combo.itemData(i) == settings.provider:
                provider_index = i
                break
        self.provider_combo.setCurrentIndex(provider_index)
        
        # Load other fields
        self.api_key_edit.setText(settings.api_key or "")
        self.endpoint_edit.setText(settings.custom_endpoint or "")
        self.profile_edit.setText(settings.profile_name)
        self.max_steps_spin.setValue(settings.max_steps)
        self.headless_check.setChecked(settings.headless)
        self.auto_approve_check.setChecked(settings.auto_approve)
        
        # Update provider-specific UI
        self._on_provider_changed()
        
        # Set model after updating suggestions
        if settings.model:
            self.model_combo.setCurrentText(settings.model)
    
    def _on_provider_changed(self):
        """Handle provider selection change."""
        provider_value = self.provider_combo.currentData()
        try:
            provider = Provider(provider_value)
        except ValueError:
            provider = Provider.LM_STUDIO
        
        # Update API key placeholder and requirement
        requires_key = PROVIDER_REQUIRES_API_KEY.get(provider, True)
        if requires_key:
            self.api_key_edit.setPlaceholderText("Required")
        else:
            self.api_key_edit.setPlaceholderText("Not required for LM Studio")
        
        # Update model suggestions
        self.model_combo.clear()
        suggestions = PROVIDER_MODEL_SUGGESTIONS.get(provider, [])
        self.model_combo.addItems(suggestions)
        
        # Set default model
        default_model = PROVIDER_DEFAULT_MODELS.get(provider, "")
        if default_model in suggestions:
            self.model_combo.setCurrentText(default_model)
        
        # Update default endpoint label
        default_endpoint = PROVIDER_ENDPOINTS.get(provider, "")
        self.default_endpoint_label.setText(f"Default: {default_endpoint}")
        
        # Clear model status
        self.model_status_label.setText("")
    
    def _on_show_key_toggled(self, checked: bool):
        """Toggle API key visibility."""
        if checked:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("Show")
    
    def _on_refresh_models(self):
        """Fetch models from the provider."""
        provider_value = self.provider_combo.currentData()
        try:
            provider = Provider(provider_value)
        except ValueError:
            provider = Provider.LM_STUDIO
        
        api_key = self.api_key_edit.text().strip() or None
        custom_endpoint = self.endpoint_edit.text().strip() or None
        
        # Check if API key is required
        requires_key = PROVIDER_REQUIRES_API_KEY.get(provider, True)
        if requires_key and not api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                f"Please enter an API key for {PROVIDER_DISPLAY_NAMES[provider]} first."
            )
            self.api_key_edit.setFocus()
            return
        
        # Disable button and show loading
        self.refresh_models_btn.setEnabled(False)
        self.refresh_models_btn.setText("Loading...")
        self.model_status_label.setText("Fetching models...")
        self.model_status_label.setStyleSheet("color: #0066cc; font-size: 11px;")
        
        # Start background fetch
        self._fetch_worker = ModelFetchWorker(provider, api_key, custom_endpoint)
        self._fetch_worker.finished.connect(self._on_models_fetched)
        self._fetch_worker.error.connect(self._on_models_error)
        self._fetch_worker.start()
    
    def _on_models_fetched(self, models: list):
        """Handle successful model fetch."""
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("🔄 Refresh")
        
        if models:
            current_model = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(models)
            
            # Restore previous selection if still available
            if current_model in models:
                self.model_combo.setCurrentText(current_model)
            
            self.model_status_label.setText(f"Found {len(models)} models")
            self.model_status_label.setStyleSheet("color: #28a745; font-size: 11px;")
        else:
            self.model_status_label.setText("No models found")
            self.model_status_label.setStyleSheet("color: #ffc107; font-size: 11px;")
    
    def _on_models_error(self, error: str):
        """Handle model fetch error."""
        self.refresh_models_btn.setEnabled(True)
        self.refresh_models_btn.setText("🔄 Refresh")
        self.model_status_label.setText(f"Error: {error[:50]}...")
        self.model_status_label.setStyleSheet("color: #dc3545; font-size: 11px;")
    
    def _on_save(self):
        """Save settings and close."""
        provider_value = self.provider_combo.currentData()
        try:
            provider = Provider(provider_value)
        except ValueError:
            provider = Provider.LM_STUDIO
        
        # Validate
        api_key = self.api_key_edit.text().strip() or None
        requires_key = PROVIDER_REQUIRES_API_KEY.get(provider, True)
        
        if requires_key and not api_key:
            QMessageBox.warning(
                self,
                "Validation Error",
                f"{PROVIDER_DISPLAY_NAMES[provider]} requires an API key."
            )
            self.api_key_edit.setFocus()
            return
        
        # Save settings
        self.store.update(
            provider=provider.value,
            api_key=api_key,
            model=self.model_combo.currentText() or None,
            custom_endpoint=self.endpoint_edit.text().strip() or None,
            profile_name=self.profile_edit.text().strip() or "default",
            max_steps=self.max_steps_spin.value(),
            headless=self.headless_check.isChecked(),
            auto_approve=self.auto_approve_check.isChecked(),
        )
        
        self.accept()
    
    def _on_reset(self):
        """Reset to default settings."""
        result = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        
        if result == QMessageBox.StandardButton.Yes:
            self.store.reset()
            self._load_settings()
