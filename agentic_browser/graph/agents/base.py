"""
Base agent class for specialized agent nodes.

Provides common functionality for all agent types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

# Lazy import for optional providers to avoid import errors if not installed
def _get_anthropic_client():
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic
    except ImportError:
        return None

def _get_google_client():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI
    except ImportError:
        return None

from ..state import AgentState
from ...config import AgentConfig

# Configure debug logging
logger = logging.getLogger("agentic_browser.agents")
DEBUG_MODE = True  # Set to False in production


def create_llm_client(config: AgentConfig, max_tokens: int = 1000):
    """Create the appropriate LangChain LLM client based on provider detection.
    
    Detects provider from endpoint URL or model name and returns the
    appropriate client (ChatOpenAI, ChatAnthropic, or ChatGoogleGenerativeAI).
    """
    model_lower = (config.model or "").lower()
    endpoint_lower = (config.model_endpoint or "").lower()
    
    # Detect provider from endpoint or model name
    is_anthropic = "anthropic" in endpoint_lower or "claude" in model_lower
    is_google = "googleapis" in endpoint_lower or "gemini" in model_lower
    is_gpt5 = any(x in model_lower for x in ["gpt-5", "gpt5"])
    is_reasoning_model = any(x in model_lower for x in ["o1", "o3", "o4"])
    
    # Set appropriate max_tokens for GPT-5
    if is_gpt5:
        max_tokens = max(max_tokens, 4000)
    
    if is_anthropic:
        # Use ChatAnthropic for Anthropic models
        ChatAnthropic = _get_anthropic_client()
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
        
        return ChatAnthropic(
            api_key=config.api_key or "not-required",
            model=config.model,
            max_tokens=max_tokens,
        )
    
    elif is_google:
        # Use ChatGoogleGenerativeAI for Google models
        ChatGoogle = _get_google_client()
        if ChatGoogle is None:
            raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
        
        return ChatGoogle(
            google_api_key=config.api_key or "not-required",
            model=config.model,
            max_output_tokens=max_tokens,
        )
    
    else:
        # Default: Use ChatOpenAI (works for OpenAI and OpenAI-compatible endpoints)
        llm_kwargs = {
            "base_url": config.model_endpoint,
            "api_key": config.api_key or "not-required",
            "model": config.model,
            "max_tokens": max_tokens,
        }
        
        # Only add temperature for models that support it
        if not is_reasoning_model:
            llm_kwargs["temperature"] = 0.1
        
        return ChatOpenAI(**llm_kwargs)


class BaseAgent(ABC):
    """Abstract base class for specialized agents.
    
    Each agent type (Browser, OS, Research, Code) inherits from this
    and implements its own execution logic and system prompt.
    """
    
    # Override in subclasses
    AGENT_NAME: str = "base"
    MAX_STEPS_PER_INVOCATION: int = 5
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        
        # Use factory function to create provider-appropriate LLM client
        self.llm = create_llm_client(config, max_tokens=1000)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent type."""
        pass
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's task.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state
        """
        pass
    
    def _build_messages(
        self, 
        state: AgentState,
        task_context: str = "",
    ) -> list:
        """Build messages for LLM invocation.
        
        Args:
            state: Current state
            task_context: Additional context for this invocation
            
        Returns:
            List of messages for LLM
        """
        messages = [
            SystemMessage(content=self.system_prompt),
        ]
        
        # Add recent message history (last 5 for context)
        for msg in state["messages"][-5:]:
            messages.append(msg)
        
        # Add current task context
        context = f"""
Current Task: {state['goal']}
Step: {state['step_count']} / {state['max_steps']}
Previous Domain: {state['current_domain']}

{task_context}

Respond with your action in JSON format.
"""
        messages.append(HumanMessage(content=context))
        
        return messages
    
    def _update_state(
        self,
        state: AgentState,
        messages: list[BaseMessage] | None = None,  # Changed from single message
        extracted_data: dict | None = None,
        visited_url: str | None = None,
        file_accessed: str | None = None,
        error: str | None = None,
        task_complete: bool = False,
        final_answer: str | None = None,
    ) -> AgentState:
        """Create updated state with new information."""
        updates: dict[str, Any] = {
            "step_count": state["step_count"] + 1,
            "active_agent": self.AGENT_NAME,
        }
        
        if messages:
            updates["messages"] = messages
        
        if extracted_data:
            merged = {**state["extracted_data"], **extracted_data}
            updates["extracted_data"] = merged
        
        if visited_url:
            updates["visited_urls"] = [visited_url]
        
        if file_accessed:
            updates["files_accessed"] = [file_accessed]
        
        if error:
            updates["error"] = error
            updates["retry_count"] = state["retry_count"] + 1
        else:
            updates["error"] = None
        
        if task_complete:
            updates["task_complete"] = True
            updates["final_answer"] = final_answer
        
        return {**state, **updates}
