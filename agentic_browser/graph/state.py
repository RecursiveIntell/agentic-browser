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
    visited_urls: Annotated[list[str], operator.add]
    files_accessed: Annotated[list[str], operator.add]
    
    # Error tracking
    error: str | None
    retry_count: int
    consecutive_errors: int
    
    # Step tracking
    step_count: int
    max_steps: int
    
    # Safety
    pending_approval: dict | None  # Action awaiting human approval
    approved_actions: list[str]  # Actions user has pre-approved
    
    # Session ID for tool registry lookup (serializable string)
    session_id: str
    
    # Research agent: track clicked links to avoid re-clicking
    # NOTE: Not using operator.add here - we manually manage this list to avoid exponential growth
    clicked_selectors: list[str]
    
    # Track if last action was scroll - forces extract on next step to update context
    last_action_was_scroll: bool


def create_initial_state(
    goal: str, 
    max_steps: int = 30,
    session_id: str = "",
) -> AgentState:
    """Create initial state for a new task.
    
    Args:
        goal: User's goal/request
        max_steps: Maximum steps allowed
        session_id: Session ID for looking up tools from registry
        
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
        consecutive_errors=0,
        step_count=0,
        max_steps=max_steps,
        pending_approval=None,
        approved_actions=[],
        session_id=session_id,
        clicked_selectors=[],
        last_action_was_scroll=False,
    )

