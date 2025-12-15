"""
LLM Provider configuration for Agentic Browser.

Provides provider-specific endpoints and default models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    """Supported LLM providers."""
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# Default endpoints for each provider
PROVIDER_ENDPOINTS = {
    Provider.LM_STUDIO: "http://127.0.0.1:1234/v1",
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.ANTHROPIC: "https://api.anthropic.com/v1",
    Provider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
}

# Default models for each provider
PROVIDER_DEFAULT_MODELS = {
    Provider.LM_STUDIO: "qwen2.5:7b",
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-haiku-20240307",
    Provider.GOOGLE: "gemini-1.5-flash",
}

# Suggested models for each provider
PROVIDER_MODEL_SUGGESTIONS = {
    Provider.LM_STUDIO: [
        "qwen2.5:7b",
        "llama3.1:8b",
        "mistral:7b",
        "codellama:7b",
    ],
    Provider.OPENAI: [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    Provider.ANTHROPIC: [
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-sonnet-20241022",
    ],
    Provider.GOOGLE: [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ],
}

# Provider display names
PROVIDER_DISPLAY_NAMES = {
    Provider.LM_STUDIO: "LM Studio (Local)",
    Provider.OPENAI: "OpenAI",
    Provider.ANTHROPIC: "Anthropic",
    Provider.GOOGLE: "Google AI",
}

# Whether provider requires API key
PROVIDER_REQUIRES_API_KEY = {
    Provider.LM_STUDIO: False,
    Provider.OPENAI: True,
    Provider.ANTHROPIC: True,
    Provider.GOOGLE: True,
}


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    
    provider: Provider = Provider.LM_STUDIO
    api_key: Optional[str] = None
    model: Optional[str] = None
    custom_endpoint: Optional[str] = None
    
    @property
    def endpoint(self) -> str:
        """Get the endpoint URL for this provider."""
        if self.custom_endpoint:
            return self.custom_endpoint.rstrip("/")
        return PROVIDER_ENDPOINTS.get(self.provider, PROVIDER_ENDPOINTS[Provider.LM_STUDIO])
    
    @property
    def effective_model(self) -> str:
        """Get the effective model name."""
        if self.model:
            return self.model
        return PROVIDER_DEFAULT_MODELS.get(self.provider, "qwen2.5:7b")
    
    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key."""
        return PROVIDER_REQUIRES_API_KEY.get(self.provider, True)
    
    @property
    def display_name(self) -> str:
        """Get the display name for this provider."""
        return PROVIDER_DISPLAY_NAMES.get(self.provider, self.provider.value)
    
    def get_model_suggestions(self) -> list[str]:
        """Get model suggestions for this provider."""
        return PROVIDER_MODEL_SUGGESTIONS.get(self.provider, [])
    
    def validate(self) -> tuple[bool, str]:
        """Validate the configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.requires_api_key and not self.api_key:
            return False, f"{self.display_name} requires an API key"
        return True, ""
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfig":
        """Create from dictionary."""
        provider_str = data.get("provider", "lm_studio")
        try:
            provider = Provider(provider_str)
        except ValueError:
            provider = Provider.LM_STUDIO
            
        return cls(
            provider=provider,
            api_key=data.get("api_key"),
            model=data.get("model"),
            custom_endpoint=data.get("custom_endpoint"),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "provider": self.provider.value,
            "api_key": self.api_key,
            "model": self.model,
            "custom_endpoint": self.custom_endpoint,
        }
