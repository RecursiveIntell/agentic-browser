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
from .agents.data_agent import data_agent_node
from .agents.network_agent import network_agent_node
from .agents.sysadmin_agent import sysadmin_agent_node
from .agents.media_agent import media_agent_node
from .agents.package_agent import package_agent_node
from .agents.automation_agent import automation_agent_node


def build_agent_graph(checkpointer: MemorySaver | None = None):
    """Build the multi-agent StateGraph.
    
    Creates a graph with:
    - supervisor: Routes tasks and synthesizes results
    - browser: Web navigation agent
    - os: Filesystem/shell agent
    - research: Multi-source research agent
    - code: Code analysis agent
    - data: Data transformation agent
    - network: Network diagnostics agent
    - sysadmin: System administration agent
    - media: Media processing agent
    - package: Package & dev environment agent
    
    Args:
        checkpointer: Optional memory saver for persistence
        
    Returns:
        Compiled StateGraph
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes - original agents
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("browser", browser_agent_node)
    graph.add_node("os", os_agent_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("code", code_agent_node)
    
    # Add nodes - new agents
    graph.add_node("data", data_agent_node)
    graph.add_node("network", network_agent_node)
    graph.add_node("sysadmin", sysadmin_agent_node)
    graph.add_node("media", media_agent_node)
    graph.add_node("package", package_agent_node)
    graph.add_node("automation", automation_agent_node)
    
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
            "data": "data",
            "network": "network",
            "sysadmin": "sysadmin",
            "media": "media",
            "package": "package",
            "automation": "automation",
            "__end__": END,
        }
    )
    
    # Function to check if task is complete
    def check_task_complete(state: AgentState) -> str:
        if state.get("task_complete"):
            return "__end__"
        return "supervisor"
    
    # Worker agents: return to supervisor OR end if task complete
    # Original agents
    graph.add_conditional_edges("browser", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("os", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("research", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("code", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    
    # New agents
    graph.add_conditional_edges("data", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("network", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("sysadmin", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("media", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("package", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("automation", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    
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
        browser_manager=None,
        os_tools=None,
        enable_checkpointing: bool = False,
    ):
        """Initialize the multi-agent runner.
        
        Args:
            config: AgentConfig instance
            browser_manager: Optional LazyBrowserManager for on-demand browser
            os_tools: Optional OSTools instance
            enable_checkpointing: Enable session persistence
        """
        import uuid
        from .tool_registry import ToolRegistry
        
        self.config = config
        self.browser_manager = browser_manager
        self.os_tools = os_tools
        self.enable_checkpointing = enable_checkpointing
        
        # Generate session ID and register tools
        self.session_id = str(uuid.uuid4())
        self._registry = ToolRegistry.get_instance()
        self._registry.register(
            self.session_id,
            config=config,
            browser_manager=browser_manager,
            os_tools=os_tools,
        )
        
        # Build graph
        checkpointer = MemorySaver() if enable_checkpointing else None
        self.graph = build_agent_graph(checkpointer)
    
    def set_browser_manager(self, browser_manager) -> None:
        """Set browser manager after initialization."""
        self.browser_manager = browser_manager
        self._registry.update_browser_manager(self.session_id, browser_manager)
    
    def set_os_tools(self, os_tools) -> None:
        """Set OS tools after initialization."""
        self.os_tools = os_tools
        self._registry.update_os_tools(self.session_id, os_tools)
    
    def run(self, goal: str, max_steps: int = 30) -> dict[str, Any]:
        """Run the multi-agent system on a goal.
        
        Args:
            goal: User's goal/request
            max_steps: Maximum steps allowed
            
        Returns:
            Final state dict
        """
        # Create initial state with session_id for tool lookup
        initial_state = create_initial_state(
            goal=goal,
            max_steps=max_steps,
            session_id=self.session_id,
        )
        
        # Run the graph with thread_id for checkpointer
        final_state = self.graph.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": self.session_id,
                }
            },
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
        # Create initial state with session_id for tool lookup
        initial_state = create_initial_state(
            goal=goal,
            max_steps=max_steps,
            session_id=self.session_id,
        )
        
        try:
            for event in self.graph.stream(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": self.session_id,
                    },
                    "recursion_limit": 50,  # Allow more agent interactions
                },
            ):
                yield event
        except Exception as e:
            error_msg = str(e).lower()
            # Handle empty response errors from Anthropic/OpenAI
            if "empty" in error_msg or "must contain" in error_msg:
                print(f"[WARN] Graph stream error (empty response): {e}")
                # Yield a final state with error message
                yield {
                    "supervisor": {
                        "task_complete": True,
                        "final_answer": "Model returned an empty response. Please try again.",
                        "error": str(e),
                    }
                }
            else:
                raise
    
    def cleanup(self) -> None:
        """Unregister tools from registry."""
        self._registry.unregister(self.session_id)
    
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
