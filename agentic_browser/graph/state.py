"""
State schema for LangGraph multi-agent system.

Defines the shared state passed between all agent nodes.
"""

import logging
import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage

logger = logging.getLogger("agentic_browser.state")


# =============================================================================
# BOUNDED REDUCERS - Prevent unbounded state growth (Performance Optimization)
# =============================================================================

def bounded_message_reducer(existing: Sequence[BaseMessage], new: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Combine messages with bounded size - keep last MAX_MESSAGES.
    
    This prevents O(n²) context building and SQLite blob overflow.
    16GB RAM optimized: 40 message window.
    """
    MAX_MESSAGES = 40
    combined = list(existing) + list(new)
    if len(combined) > MAX_MESSAGES:
        logger.debug(f"Bounded messages: {len(combined)} → {MAX_MESSAGES}")
        return combined[-MAX_MESSAGES:]
    return combined


def bounded_url_reducer(existing: list[str], new: list[str]) -> list[str]:
    """Combine URLs with bounded size and deduplication.
    
    Keeps last 50 unique URLs to prevent unbounded growth.
    """
    MAX_URLS = 50
    # Dedupe while preserving order
    combined = list(dict.fromkeys(existing + new))
    if len(combined) > MAX_URLS:
        return combined[-MAX_URLS:]
    return combined


class AgentState(TypedDict):
    """Shared state for multi-agent graph.
    
    This state is passed between all nodes in the graph and accumulates
    information as agents collaborate.
    """

    # Message history - bounded to 40 messages (was: operator.add)
    messages: Annotated[Sequence[BaseMessage], bounded_message_reducer]

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
    visited_urls: Annotated[list[str], bounded_url_reducer]  # Bounded to 50 URLs
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

    # Track URLs that have been scrolled (shared between agents to avoid redundant scrolling)
    scrolled_urls: list[str]  # URLs scrolled this session (no accumulation needed)

    # Retrospective Agent Tracking
    retrospective_ran: bool

    # Planning-First Architecture: Implementation plan created by planner agent
    implementation_plan: dict | None  # Structured plan with steps, agents, success criteria

    plan_step_index: int  # Current step being executed (0-indexed)

    # Cost & Usage Tracking
    # Dictionary with keys: input_tokens, output_tokens, total_tokens, total_cost
    # Using operator.add doesn't work well for dicts, so we'll need a custom reducer or manual updates
    # Actually, we can just treat it as a dict that gets overwritten by the latest state update,
    # BUT we want to *accumulate*.
    # Better: We'll store it as a dict and agents will READ current, ADD to it, and WRITE back.
    token_usage: dict[str, float]

    # ========================================================================
    # Runtime Tracking (Browser Agent Loop Detection)
    # These fields support internal loop detection and URL-change tracking
    # ========================================================================
    browser_failed_download_count: int    # Track download failures for fallback logic
    browser_downloaded_image_count: int   # Track successful downloads for multi-image goals
    browser_recent_actions: list[str]     # Track recent actions for loop detection
    browser_failed_nav_clicks: int        # Track clicks that didn't navigate
    browser_same_page_actions: int        # Count actions on same page
    browser_last_page_base: str           # Last page URL base for comparison


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
        current_domain="planner",  # Start with planner (Planning-First)
        active_agent="planner",
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
        scrolled_urls=[],
        implementation_plan=None,
        plan_step_index=0,
        retrospective_ran=False,

        # Initial usage stats
        token_usage={
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "total_cost": 0.0,
        },

        # Browser runtime tracking (loop detection)
        browser_failed_download_count=0,
        browser_downloaded_image_count=0,
        browser_recent_actions=[],
        browser_failed_nav_clicks=0,
        browser_same_page_actions=0,
        browser_last_page_base="",
    )

