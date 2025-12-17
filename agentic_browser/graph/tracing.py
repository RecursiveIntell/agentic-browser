"""
LangSmith tracing integration for debugging and observability.

Provides automatic tracing of all LangGraph operations when enabled.
"""

import os
from typing import Optional


def configure_tracing(
    api_key: Optional[str] = None,
    project_name: str = "agentic-browser",
    enabled: bool = True,
) -> bool:
    """Configure LangSmith tracing for the application.
    
    LangSmith provides:
    - Trace visualization for agent execution
    - Token usage tracking per agent
    - Latency breakdown for each step
    - Error debugging with full context
    - Prompt inspection
    
    Args:
        api_key: LangSmith API key (or use LANGCHAIN_API_KEY env var)
        project_name: Project name for grouping traces
        enabled: Whether to enable tracing
        
    Returns:
        True if tracing was enabled successfully
    """
    if not enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    
    # Check for API key
    key = api_key or os.environ.get("LANGCHAIN_API_KEY")
    if not key:
        # Try alternative env var names
        key = os.environ.get("LANGSMITH_API_KEY")
    
    if not key:
        print("⚠️ LangSmith API key not found. Set LANGCHAIN_API_KEY to enable tracing.")
        return False
    
    # Configure environment for LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    # Optional: Set endpoint if using self-hosted
    if not os.environ.get("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    print(f"✓ LangSmith tracing enabled for project: {project_name}")
    return True


def get_tracing_url(run_id: str) -> str:
    """Get the LangSmith URL for a specific run.
    
    Args:
        run_id: The run ID from graph execution
        
    Returns:
        URL to view the trace in LangSmith
    """
    project = os.environ.get("LANGCHAIN_PROJECT", "agentic-browser")
    return f"https://smith.langchain.com/o/default/projects/{project}/traces/{run_id}"


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
