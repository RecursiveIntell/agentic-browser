"""
Base agent class for specialized agent nodes.

Provides common functionality for all agent types.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
        self.llm = ChatOpenAI(
            base_url=config.model_endpoint,
            api_key=config.api_key or "not-required",
            model=config.model,
            temperature=0.1,
            max_tokens=1000,
        )
    
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
        message: AIMessage | None = None,
        extracted_data: dict | None = None,
        visited_url: str | None = None,
        file_accessed: str | None = None,
        error: str | None = None,
        task_complete: bool = False,
        final_answer: str | None = None,
    ) -> AgentState:
        """Create updated state with new information.
        
        Args:
            state: Current state
            message: New AI message to add
            extracted_data: Data to merge into extracted_data
            visited_url: URL to add to visited list
            file_accessed: File path to add to accessed list
            error: Error message if any
            task_complete: Whether task is complete
            final_answer: Final answer if complete
            
        Returns:
            Updated state dict
        """
        updates: dict[str, Any] = {
            "step_count": state["step_count"] + 1,
            "active_agent": self.AGENT_NAME,
        }
        
        if message:
            updates["messages"] = [message]
        
        if extracted_data:
            merged = {**state["extracted_data"], **extracted_data}
            updates["extracted_data"] = merged
        
        if visited_url and visited_url not in state["visited_urls"]:
            updates["visited_urls"] = state["visited_urls"] + [visited_url]
        
        if file_accessed and file_accessed not in state["files_accessed"]:
            updates["files_accessed"] = state["files_accessed"] + [file_accessed]
        
        if error:
            updates["error"] = error
            updates["retry_count"] = state["retry_count"] + 1
        else:
            updates["error"] = None
        
        if task_complete:
            updates["task_complete"] = True
            updates["final_answer"] = final_answer
        
        return {**state, **updates}
