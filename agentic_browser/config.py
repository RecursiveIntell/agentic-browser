"""
Configuration management for Agentic Browser.

Provides configuration dataclass and environment variable loading.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


def get_base_dir() -> Path:
    """Get the base directory for agentic browser data."""
    return Path.home() / ".agentic_browser"


def get_profiles_dir() -> Path:
    """Get the directory for browser profiles."""
    return get_base_dir() / "profiles"


def get_runs_dir() -> Path:
    """Get the directory for run logs."""
    return get_base_dir() / "runs"


@dataclass
class AgentConfig:
    """Configuration for the browser agent."""
    
    # Goal to accomplish
    goal: str
    
    # Profile settings
    profile_name: str = "default"
    fresh_profile: bool = False
    no_persist: bool = False
    
    # Browser settings
    headless: bool = False
    
    # Agent settings
    max_steps: int = 30
    auto_approve: bool = False
    
    # LLM settings
    model_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "AGENTIC_BROWSER_ENDPOINT", 
            "http://127.0.0.1:1234/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "AGENTIC_BROWSER_MODEL", 
            "qwen2.5:7b"
        )
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AGENTIC_BROWSER_API_KEY")
    )
    
    # Timeouts (ms)
    navigation_timeout: int = 30000
    action_timeout: int = 10000
    
    # Content limits
    visible_text_max_chars: int = 8000
    max_links: int = 15
    history_length: int = 5
    
    # Loop detection
    max_repeat_actions: int = 3
    
    # Enable Playwright tracing
    enable_tracing: bool = False
    
    @property
    def profile_dir(self) -> Path:
        """Get the path to the browser profile directory."""
        if self.no_persist:
            return Path(tempfile.mkdtemp(prefix="agentic_browser_"))
        return get_profiles_dir() / self.profile_name
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        get_base_dir().mkdir(parents=True, exist_ok=True)
        get_profiles_dir().mkdir(parents=True, exist_ok=True)
        get_runs_dir().mkdir(parents=True, exist_ok=True)
        
        if not self.no_persist:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_cli_args(
        cls,
        goal: str,
        profile: str = "default",
        headless: bool = False,
        max_steps: int = 30,
        model_endpoint: Optional[str] = None,
        model: Optional[str] = None,
        auto_approve: bool = False,
        fresh_profile: bool = False,
        no_persist: bool = False,
        enable_tracing: bool = False,
    ) -> "AgentConfig":
        """Create configuration from CLI arguments."""
        config = cls(
            goal=goal,
            profile_name=profile,
            headless=headless,
            max_steps=max_steps,
            auto_approve=auto_approve,
            fresh_profile=fresh_profile,
            no_persist=no_persist,
            enable_tracing=enable_tracing,
        )
        
        # Override with CLI args if provided
        if model_endpoint:
            config.model_endpoint = model_endpoint
        if model:
            config.model = model
            
        return config


# Default configuration values for documentation
DEFAULTS = {
    "profile": "default",
    "headless": False,
    "max_steps": 30,
    "model_endpoint": "http://127.0.0.1:1234/v1",
    "model": "qwen2.5:7b",
    "auto_approve": False,
    "navigation_timeout_ms": 30000,
    "action_timeout_ms": 10000,
    "visible_text_max_chars": 8000,
    "max_links": 15,
    "history_length": 5,
}
