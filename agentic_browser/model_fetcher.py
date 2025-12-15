"""
Model fetching for LLM providers.

Provides API calls to fetch available models from each provider.
"""

from typing import Optional
import httpx

from .providers import Provider, PROVIDER_ENDPOINTS


def fetch_models(
    provider: Provider,
    api_key: Optional[str] = None,
    custom_endpoint: Optional[str] = None,
    timeout: float = 10.0,
) -> list[str]:
    """Fetch available models from a provider's API.
    
    Args:
        provider: The LLM provider
        api_key: API key (required for most providers)
        custom_endpoint: Custom endpoint override
        timeout: Request timeout in seconds
        
    Returns:
        List of model names/IDs
        
    Raises:
        Exception: On network or API errors
    """
    endpoint = custom_endpoint or PROVIDER_ENDPOINTS.get(provider)
    if not endpoint:
        return []
    
    endpoint = endpoint.rstrip("/")
    
    if provider == Provider.LM_STUDIO:
        return _fetch_lm_studio_models(endpoint, timeout)
    elif provider == Provider.OPENAI:
        return _fetch_openai_models(endpoint, api_key, timeout)
    elif provider == Provider.ANTHROPIC:
        return _fetch_anthropic_models(endpoint, api_key, timeout)
    elif provider == Provider.GOOGLE:
        return _fetch_google_models(endpoint, api_key, timeout)
    
    return []


def _fetch_lm_studio_models(endpoint: str, timeout: float) -> list[str]:
    """Fetch models from LM Studio (OpenAI-compatible API)."""
    url = f"{endpoint}/models"
    
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            model_id = model.get("id")
            if model_id:
                models.append(model_id)
        
        return sorted(models)


def _fetch_openai_models(endpoint: str, api_key: Optional[str], timeout: float) -> list[str]:
    """Fetch models from OpenAI API."""
    if not api_key:
        raise ValueError("OpenAI requires an API key")
    
    url = f"{endpoint}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            # Filter to chat models
            if any(x in model_id for x in ["gpt-4", "gpt-3.5", "o1", "o3"]):
                models.append(model_id)
        
        return sorted(models)


def _fetch_anthropic_models(endpoint: str, api_key: Optional[str], timeout: float) -> list[str]:
    """Fetch models from Anthropic API.
    
    Note: Anthropic doesn't have a models endpoint, so we return a static list.
    """
    if not api_key:
        raise ValueError("Anthropic requires an API key")
    
    # Anthropic doesn't expose a models list API, return known models
    return [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]


def _fetch_google_models(endpoint: str, api_key: Optional[str], timeout: float) -> list[str]:
    """Fetch models from Google AI API."""
    if not api_key:
        raise ValueError("Google AI requires an API key")
    
    url = f"{endpoint}/models?key={api_key}"
    
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("models", []):
            name = model.get("name", "")
            # Extract model ID from "models/gemini-1.5-pro" format
            if name.startswith("models/"):
                model_id = name[7:]
                # Filter to generative models 
                if "gemini" in model_id:
                    models.append(model_id)
        
        return sorted(models)
