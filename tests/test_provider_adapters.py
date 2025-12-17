"""
Tests for provider adapters (Section 1).

Tests the LLM adapters for OpenAI, Anthropic, and Google with mocked HTTP responses.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import httpx

from agentic_browser.adapters import (
    Message,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    create_adapter,
)
from agentic_browser.providers import Provider, ProviderConfig


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=MagicMock(),
                response=self,
            )


class TestOpenAIAdapter:
    """Tests for OpenAI adapter."""
    
    def test_chat_completion_success(self):
        """Test successful chat completion."""
        config = ProviderConfig(
            provider=Provider.OPENAI,
            api_key="test-key",
            model="gpt-4",
        )
        adapter = OpenAIAdapter(config)
        
        mock_response = MockResponse({
            "choices": [{"message": {"content": "Hello, I'm GPT-4!"}}]
        })
        
        with patch.object(adapter.client, 'post', return_value=mock_response):
            result = adapter.chat_completion([
                Message("system", "You are a helpful assistant."),
                Message("user", "Hello!"),
            ])
        
        assert result == "Hello, I'm GPT-4!"
        adapter.close()
    
    def test_chat_completion_builds_correct_payload(self):
        """Test that the correct payload is sent."""
        config = ProviderConfig(
            provider=Provider.OPENAI,
            api_key="test-key",
            model="gpt-4o-mini",
        )
        adapter = OpenAIAdapter(config)
        
        mock_response = MockResponse({
            "choices": [{"message": {"content": "Response"}}]
        })
        
        with patch.object(adapter.client, 'post', return_value=mock_response) as mock_post:
            adapter.chat_completion([
                Message("user", "Test message"),
            ])
            
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs["json"]
            
            assert payload["model"] == "gpt-4o-mini"
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][0]["content"] == "Test message"
        
        adapter.close()


class TestAnthropicAdapter:
    """Tests for Anthropic adapter."""
    
    def test_chat_completion_success(self):
        """Test successful Anthropic chat completion."""
        config = ProviderConfig(
            provider=Provider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-sonnet",
        )
        adapter = AnthropicAdapter(config)
        
        mock_response = MockResponse({
            "content": [{"type": "text", "text": "Hello from Claude!"}]
        })
        
        with patch.object(adapter.client, 'post', return_value=mock_response):
            result = adapter.chat_completion([
                Message("system", "You are Claude."),
                Message("user", "Hello!"),
            ])
        
        assert result == "Hello from Claude!"
        adapter.close()
    
    def test_system_message_separated(self):
        """Test that system message is passed as separate parameter."""
        config = ProviderConfig(
            provider=Provider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-haiku",
        )
        adapter = AnthropicAdapter(config)
        
        mock_response = MockResponse({
            "content": [{"type": "text", "text": "Response"}]
        })
        
        with patch.object(adapter.client, 'post', return_value=mock_response) as mock_post:
            adapter.chat_completion([
                Message("system", "System instruction"),
                Message("user", "User message"),
            ])
            
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs["json"]
            
            # System should be separate parameter
            assert payload["system"] == "System instruction"
            # Messages should only contain user/assistant
            assert len(payload["messages"]) == 1
            assert payload["messages"][0]["role"] == "user"
        
        adapter.close()


class TestGoogleAdapter:
    """Tests for Google GenAI adapter."""
    
    def test_chat_completion_success(self):
        """Test successful Google chat completion."""
        config = ProviderConfig(
            provider=Provider.GOOGLE,
            api_key="test-key",
            model="gemini-1.5-flash",
        )
        adapter = GoogleAdapter(config)
        
        mock_response = MockResponse({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}]
                }
            }]
        })
        
        with patch.object(adapter.client, 'post', return_value=mock_response):
            result = adapter.chat_completion([
                Message("user", "Hello!"),
            ])
        
        assert result == "Hello from Gemini!"
        adapter.close()
    
    def test_conversation_format(self):
        """Test that messages are converted to Google format."""
        config = ProviderConfig(
            provider=Provider.GOOGLE,
            api_key="test-key",
            model="gemini-1.5-flash",
        )
        adapter = GoogleAdapter(config)
        
        mock_response = MockResponse({
            "candidates": [{
                "content": {"parts": [{"text": "Response"}]}
            }]
        })
        
        with patch.object(adapter.client, 'post', return_value=mock_response) as mock_post:
            adapter.chat_completion([
                Message("system", "System instruction"),
                Message("user", "User message"),
                Message("assistant", "Previous response"),
                Message("user", "Follow up"),
            ])
            
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs["json"]
            
            # System should be in systemInstruction
            assert "systemInstruction" in payload
            
            # Check contents format
            assert len(payload["contents"]) == 3  # user, model, user
            assert payload["contents"][0]["role"] == "user"
            assert payload["contents"][1]["role"] == "model"
            assert payload["contents"][2]["role"] == "user"
        
        adapter.close()


class TestAdapterFactory:
    """Tests for the create_adapter factory function."""
    
    def test_create_openai_adapter(self):
        """Test creating OpenAI adapter."""
        config = ProviderConfig(provider=Provider.OPENAI)
        adapter = create_adapter(Provider.OPENAI, config)
        
        assert isinstance(adapter, OpenAIAdapter)
        adapter.close()
    
    def test_create_lm_studio_adapter(self):
        """Test creating LM Studio adapter (OpenAI-compatible)."""
        config = ProviderConfig(provider=Provider.LM_STUDIO)
        adapter = create_adapter(Provider.LM_STUDIO, config)
        
        assert isinstance(adapter, OpenAIAdapter)
        adapter.close()
    
    def test_create_anthropic_adapter(self):
        """Test creating Anthropic adapter."""
        config = ProviderConfig(provider=Provider.ANTHROPIC, api_key="key")
        adapter = create_adapter(Provider.ANTHROPIC, config)
        
        assert isinstance(adapter, AnthropicAdapter)
        adapter.close()
    
    def test_create_google_adapter(self):
        """Test creating Google adapter."""
        config = ProviderConfig(provider=Provider.GOOGLE, api_key="key")
        adapter = create_adapter(Provider.GOOGLE, config)
        
        assert isinstance(adapter, GoogleAdapter)
        adapter.close()
