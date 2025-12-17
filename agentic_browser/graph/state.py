"""
State schema for LangGraph multi-agent system.

Defines the shared state passed between all agent nodes.
"""

from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Shared state for multi-agent graph.
    
    This state is passed between all nodes in the graph and accumulates
    information as agents collaborate.
    """
    
    # Message history - accumulates with operator.add
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Original user goal
    goal: str
    
    # Current active domain/agent
    current_domain: str  # "browser", "os", "research", "code", "supervisor"
    active_agent: str
    
    # Task completion status
    task_complete: bool
    final_answer: str | None
    
    # Collected data
    extracted_data: dict[str, Any]
    visited_urls: list[str]
    files_accessed: list[str]
    
    # Error tracking
    error: str | None
    retry_count: int
    
    # Step tracking
    step_count: int
    max_steps: int
    
    # Safety
    pending_approval: dict | None  # Action awaiting human approval
    approved_actions: list[str]  # Actions user has pre-approved
    
    # Runtime config and tools (injected by MultiAgentRunner)
    _config: Any  # AgentConfig
    _browser_tools: Any  # BrowserTools or None
    _os_tools: Any  # OSTools or None


def create_initial_state(
    goal: str, 
    max_steps: int = 30,
    config: Any = None,
    browser_tools: Any = None,
    os_tools: Any = None,
) -> AgentState:
    """Create initial state for a new task.
    
    Args:
        goal: User's goal/request
        max_steps: Maximum steps allowed
        config: AgentConfig instance
        browser_tools: BrowserTools instance
        os_tools: OSTools instance
        
    Returns:
        Initial AgentState
    """
    return AgentState(
        messages=[],
        goal=goal,
        current_domain="supervisor",
        active_agent="supervisor",
        task_complete=False,
        final_answer=None,
        extracted_data={},
        visited_urls=[],
        files_accessed=[],
        error=None,
        retry_count=0,
        step_count=0,
        max_steps=max_steps,
        pending_approval=None,
        approved_actions=[],
        _config=config,
        _browser_tools=browser_tools,
        _os_tools=os_tools,
    )

