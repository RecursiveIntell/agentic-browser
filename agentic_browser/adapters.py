"""
Provider adapters for Agentic Browser.

Provides unified LLM interface with adapters for different providers:
- OpenAI (and OpenAI-compatible like LM Studio)
- Anthropic (Claude)
- Google GenAI (Gemini)
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import httpx

from .providers import Provider, ProviderConfig


class Message:
    """Simple message container for LLM conversations."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMAdapter(ABC):
    """Abstract base class for LLM provider adapters."""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: list[Message],
        max_retries: int = 3,
    ) -> str:
        """Send a chat completion request.
        
        Args:
            messages: List of conversation messages
            max_retries: Maximum retries on rate limit
            
        Returns:
            The assistant's response content
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any resources."""
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI Chat Completions API.
    
    Works with OpenAI, LM Studio, and other OpenAI-compatible APIs.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.endpoint = config.endpoint.rstrip("/")
        self.model = config.effective_model
        
        self.headers = {"Content-Type": "application/json"}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"
        
        self.client = httpx.Client(timeout=60.0)
    
    def chat_completion(
        self,
        messages: list[Message],
        max_retries: int = 3,
    ) -> str:
        url = f"{self.endpoint}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": 0.1,
            "max_tokens": 1000,
        }
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                
                response = self.client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < max_retries:
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise
        
        raise last_error
    
    def close(self) -> None:
        self.client.close()


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Messages API (Claude).
    
    Uses the native Anthropic API format instead of OpenAI compatibility.
    """
    
    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.endpoint = config.custom_endpoint or self.ANTHROPIC_API_URL
        self.model = config.effective_model
        self.api_key = config.api_key
        
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        
        self.client = httpx.Client(timeout=60.0)
    
    def chat_completion(
        self,
        messages: list[Message],
        max_retries: int = 3,
    ) -> str:
        # Convert messages to Anthropic format
        # Anthropic uses "system" parameter separately
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                # Anthropic expects "user" and "assistant" roles
                role = "user" if msg.role == "user" else "assistant"
                user_messages.append({
                    "role": role,
                    "content": msg.content,
                })
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": user_messages,
        }
        
        if system_message:
            payload["system"] = system_message
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                
                response = self.client.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                )
                response.raise_for_status()
                
                data = response.json()
                # Anthropic returns content as a list of blocks
                content_blocks = data.get("content", [])
                if content_blocks and content_blocks[0].get("type") == "text":
                    return content_blocks[0]["text"]
                return ""
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < max_retries:
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise
        
        raise last_error
    
    def close(self) -> None:
        self.client.close()


class GoogleAdapter(LLMAdapter):
    """Adapter for Google GenAI API (Gemini).
    
    Uses the native Google Generative AI API format.
    """
    
    GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.custom_endpoint or self.GOOGLE_API_URL
        self.model = config.effective_model
        self.api_key = config.api_key
        
        self.client = httpx.Client(timeout=60.0)
    
    def chat_completion(
        self,
        messages: list[Message],
        max_retries: int = 3,
    ) -> str:
        # Google Gemini uses a different format
        # Convert to Gemini "contents" format
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}],
                })
        
        # Build URL with API key
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000,
            },
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                
                response = self.client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                
                data = response.json()
                # Google returns candidates with content.parts
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429 and attempt < max_retries:
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise
        
        raise last_error
    
    def close(self) -> None:
        self.client.close()


def create_adapter(provider: Provider, config: ProviderConfig) -> LLMAdapter:
    """Create an LLM adapter for the specified provider.
    
    Args:
        provider: The LLM provider type
        config: Provider configuration
        
    Returns:
        Configured LLM adapter
    """
    adapters = {
        Provider.LM_STUDIO: OpenAIAdapter,
        Provider.OPENAI: OpenAIAdapter,
        Provider.ANTHROPIC: AnthropicAdapter,
        Provider.GOOGLE: GoogleAdapter,
    }
    
    adapter_class = adapters.get(provider, OpenAIAdapter)
    return adapter_class(config)
