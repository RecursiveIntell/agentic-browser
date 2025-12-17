"""
Base agent class for specialized agent nodes.

Provides common functionality for all agent types.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

from ..state import AgentState
from ...config import AgentConfig


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
        
        # Build LLM kwargs - some models (o1, o3) don't support temperature
        llm_kwargs = {
            "base_url": config.model_endpoint,
            "api_key": config.api_key or "not-required",
            "model": config.model,
            "max_tokens": 1000,
        }
        
        # Only add temperature for models that support it
        model_lower = (config.model or "").lower()
        is_reasoning_model = any(x in model_lower for x in ["o1", "o3", "o4"])
        if not is_reasoning_model:
            llm_kwargs["temperature"] = 0.1
            
        self.llm = ChatOpenAI(**llm_kwargs)
    
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
