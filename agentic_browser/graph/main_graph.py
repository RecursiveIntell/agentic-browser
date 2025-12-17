"""
Main LangGraph construction for multi-agent system.

Wires together supervisor and specialized agents into a StateGraph.
"""

from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, create_initial_state
from .supervisor import supervisor_node, route_to_agent
from .agents.browser_agent import browser_agent_node
from .agents.os_agent import os_agent_node
from .agents.research_agent import research_agent_node
from .agents.code_agent import code_agent_node


def build_agent_graph(checkpointer: MemorySaver | None = None):
    """Build the multi-agent StateGraph.
    
    Creates a graph with:
    - supervisor: Routes tasks and synthesizes results
    - browser: Web navigation agent
    - os: Filesystem/shell agent
    - research: Multi-source research agent
    - code: Code analysis agent
    
    Args:
        checkpointer: Optional memory saver for persistence
        
    Returns:
        Compiled StateGraph
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("browser", browser_agent_node)
    graph.add_node("os", os_agent_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("code", code_agent_node)
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    # Add conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "browser": "browser",
            "os": "os",
            "research": "research",
            "code": "code",
            "__end__": END,
        }
    )
    
    # All worker agents return to supervisor
    graph.add_edge("browser", "supervisor")
    graph.add_edge("os", "supervisor")
    graph.add_edge("research", "supervisor")
    graph.add_edge("code", "supervisor")
    
    # Compile with optional checkpointer
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


class MultiAgentRunner:
    """High-level interface for running the multi-agent graph.
    
    Manages tool initialization and graph execution.
    """
    
    def __init__(
        self,
        config,
        browser_tools=None,
        os_tools=None,
        enable_checkpointing: bool = False,
    ):
        """Initialize the multi-agent runner.
        
        Args:
            config: AgentConfig instance
            browser_tools: Optional BrowserTools instance
            os_tools: Optional OSTools instance
            enable_checkpointing: Enable session persistence
        """
        self.config = config
        self.browser_tools = browser_tools
        self.os_tools = os_tools
        
        # Build graph
        checkpointer = MemorySaver() if enable_checkpointing else None
        self.graph = build_agent_graph(checkpointer)
        
        # Runtime config passed to all nodes
        self.runtime_config = {
            "agent_config": config,
            "browser_tools": browser_tools,
            "os_tools": os_tools,
        }
    
    def set_browser_tools(self, browser_tools) -> None:
        """Set browser tools after initialization."""
        self.browser_tools = browser_tools
        self.runtime_config["browser_tools"] = browser_tools
    
    def set_os_tools(self, os_tools) -> None:
        """Set OS tools after initialization."""
        self.os_tools = os_tools
        self.runtime_config["os_tools"] = os_tools
    
    def run(self, goal: str, max_steps: int = 30) -> dict[str, Any]:
        """Run the multi-agent system on a goal.
        
        Args:
            goal: User's goal/request
            max_steps: Maximum steps allowed
            
        Returns:
            Final state dict
        """
        # Create initial state
        initial_state = create_initial_state(goal, max_steps)
        
        # Run the graph
        final_state = self.graph.invoke(
            initial_state,
            config={"configurable": self.runtime_config},
        )
        
        return final_state
    
    def stream(self, goal: str, max_steps: int = 30):
        """Stream execution steps for real-time updates.
        
        Args:
            goal: User's goal/request
            max_steps: Maximum steps allowed
            
        Yields:
            State updates as they occur
        """
        initial_state = create_initial_state(goal, max_steps)
        
        for event in self.graph.stream(
            initial_state,
            config={"configurable": self.runtime_config},
        ):
            yield event
    
    def get_result(self, state: dict) -> str:
        """Extract final answer from state.
        
        Args:
            state: Final state dict
            
        Returns:
            Final answer string
        """
        if state.get("final_answer"):
            return state["final_answer"]
        
        if state.get("extracted_data"):
            import json
            return f"Collected data: {json.dumps(state['extracted_data'], indent=2)}"
        
        return "Task completed but no specific answer was generated."
