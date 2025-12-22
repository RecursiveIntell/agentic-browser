"""
Main LangGraph construction for multi-agent system.

Wires together supervisor and specialized agents into a StateGraph.
Supports live steering via input_queue.
"""

import queue
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

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
from .agents.planner_agent import planner_agent_node
from .agents.workflow_agent import workflow_agent_node
from .agents.retrospective_agent import retrospective_agent_node


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
    - automation: Scheduling & reminders agent
    - workflow: n8n workflow integration agent
    
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
    graph.add_node("workflow", workflow_agent_node)
    graph.add_node("retrospective", retrospective_agent_node)
    
    # Planning-First: Planner runs first
    graph.add_node("planner", planner_agent_node)
    
    # Set entry point to PLANNER (Planning-First Architecture)
    graph.set_entry_point("planner")
    
    # Planner always goes to supervisor after creating plan
    graph.add_edge("planner", "supervisor")
    
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
            "workflow": "workflow",
            "retrospective": "retrospective",
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
    graph.add_conditional_edges("workflow", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("retrospective", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    
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
        
        # Thread-safe input queue for live steering
        self.input_queue: queue.Queue = queue.Queue()
        
        # Current node for graph visualization
        self._current_node: str = "planner"
        
        # Initialize persistence
        from .memory import SessionStore
        self.session_store = SessionStore() if enable_checkpointing else None
        
        # Initialize RecallTool
        from .run_history import RecallTool
        self.recall_tool = RecallTool(self.session_store) if self.session_store else None
        
        # Generate session ID and register tools
        self.session_id = str(uuid.uuid4())
        self._registry = ToolRegistry.get_instance()
        self._registry.register(
            self.session_id,
            config=config,
            browser_manager=browser_manager,
            os_tools=os_tools,
            recall_tool=self.recall_tool,
        )
        
        # Build graph
        # CRITICAL: MemorySaver causes hangs - checkpoints after every super-step
        # causing exponential memory growth. Disable by default.
        checkpointer = None  # Disabled - was causing step 13+ hangs
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
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": self.session_id,
                    }
                },
            )
            return final_state
        except Exception as e:
            error_msg = str(e).lower()
            # Handle empty response errors - model returned nothing
            empty_patterns = ["empty", "must contain", "output text", "tool calls", "cannot both be empty"]
            if any(p in error_msg for p in empty_patterns):
                print(f"[GRAPH] ⚠️ Model returned empty response: {e}")
                return {
                    **initial_state,
                    "task_complete": True,
                    "final_answer": "Model returned empty response - please try a different model or restart LM Studio",
                    "error": str(e),
                }
            raise
    
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
            # Create session in DB
            if self.session_store:
                self.session_store.create_session(
                    self.session_id, 
                    goal, 
                    {**initial_state, "session_id": self.session_id}
                )
            
            # We need an LLM client.
            from .agents.base import create_llm_client
            
            # Get config from registry
            for event in self.graph.stream(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": self.session_id,
                    },
                    "recursion_limit": 50,  # Allow more agent interactions
                },
            ):
                # LIVE STEERING: Check input queue before processing
                steering_messages, approval_update = self._process_steering_queue()
                
                # PERSISTENCE: Update session store (THROTTLED for performance)
                for node_name, node_state in event.items():
                    # Update current node for graph visualization
                    self._current_node = node_name
                    
                    # Apply approval update if present
                    if approval_update:
                        if approval_update.get("pending_approval") is None:
                            node_state["pending_approval"] = None
                        if "add_approved_action" in approval_update:
                            action_key = approval_update["add_approved_action"]
                            approved = node_state.get("approved_actions", [])
                            if action_key not in approved:
                                node_state["approved_actions"] = approved + [action_key]
                        if "error" in approval_update:
                            node_state["error"] = approval_update["error"]
                    
                    # Inject steering messages if present
                    if steering_messages and "messages" in node_state:
                        for msg in steering_messages:
                            node_state["messages"].append(msg)
                    
                    # Add current_node to state for GUI
                    node_state["current_node"] = node_name
                    
                    if self.session_store:
                        # THROTTLE: Only persist every 3 steps or on completion
                        step_count = node_state.get("step_count", 0)
                        is_finished = node_state.get("task_complete", False)
                        
                        if is_finished or step_count % 3 == 0 or step_count == 1:
                            self.session_store.update_session(self.session_id, node_state)
                        
                        # Log step if applicable
                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            last_msg = node_state["messages"][-1]
                            agent_name = node_name
                            step_num = node_state.get("step_count", 0)
                            
                            # VECTOR DB: Persist message to separate table
                            role = type(last_msg).__name__.replace("Message", "").lower()
                            content = str(last_msg.content) if hasattr(last_msg, 'content') else str(last_msg)
                            try:
                                self.session_store.add_message(self.session_id, role, content, step_num)
                            except Exception as e:
                                print(f"[SESSION] Message save failed: {e}")
                            
                            # Log tool calls or text
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    self.session_store.add_step(
                                        self.session_id,
                                        step_num,
                                        agent_name,
                                        tc.get("name"),
                                        tc.get("args"),
                                        None
                                    )
                            else:
                                content = str(last_msg.content)
                                self.session_store.add_step(
                                    self.session_id,
                                    step_num,
                                    agent_name,
                                    "think",
                                    {"text": content[:200]},
                                    None
                                )
                
                yield event
            
            if self.session_store:
                self.session_store.close()
                
        except Exception as e:
            error_msg = str(e).lower()
            if self.session_store:
                 # Update session with error
                 try:
                     self.session_store.update_session(self.session_id, {"error": str(e), "task_complete": True})
                 except:
                     pass
            


            # Handle empty response errors from Anthropic/OpenAI
            if "empty" in error_msg or "must contain" in error_msg:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Graph stream error (empty response): {e}")
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
    
    def _process_steering_queue(self) -> tuple[list, dict | None]:
        """Process any pending steering messages from the input queue.
        
        Returns:
            Tuple of (messages list, approval_update dict or None)
        """
        messages = []
        approval_update = None
        
        while not self.input_queue.empty():
            try:
                item = self.input_queue.get_nowait()
                if item.get("type") == "abort":
                    # Abort requested - return special message
                    return [SystemMessage(content="[ABORT] User requested task cancellation. Wrap up immediately.")], None
                elif item.get("type") == "steering":
                    content = item.get("content", "")
                    if content:
                        messages.append(SystemMessage(
                            content=f"[USER INTERVENTION - HIGH PRIORITY] {content}"
                        ))
                elif item.get("type") == "approval_response":
                    # Handle user approval/rejection of pending action
                    approved = item.get("approved", False)
                    action = item.get("action", "")
                    args = item.get("args", {})
                    action_key = f"{action}:{hash(str(args))}"
                    
                    if approved:
                        print(f"[GRAPH] ✅ User APPROVED action: {action}")
                        approval_update = {
                            "pending_approval": None,
                            "add_approved_action": action_key,
                        }
                    else:
                        print(f"[GRAPH] ❌ User REJECTED action: {action}")
                        approval_update = {
                            "pending_approval": None,
                            "error": f"User rejected action: {action}",
                        }
            except Exception:
                break
        
        return messages, approval_update
    
    @property
    def current_node(self) -> str:
        """Get current active node name for graph visualization."""
        return self._current_node
    
    def cleanup(self) -> None:
        """Unregister tools from registry."""
        self._registry.unregister(self.session_id)
    
    def get_result(self, state: dict) -> str:
        """Extract final answer from state with comprehensive fallbacks.
        
        Args:
            state: Final state dict
            
        Returns:
            Final answer string - synthesized from available data
        """
        # 1. Direct final_answer (highest priority)
        if state.get("final_answer"):
            return state["final_answer"]
        
        # 2. Check extracted_data for meaningful content
        extracted = state.get("extracted_data", {})
        if extracted:
            results = []
            
            # Downloaded images
            download_keys = [k for k in extracted.keys() if 'download' in k.lower()]
            if download_keys:
                for k in download_keys:
                    if 'filename' not in k:  # Skip filename keys
                        results.append(f"📥 Downloaded: {extracted[k]}")
            
            # Research findings
            research_keys = [k for k in extracted.keys() if 'research' in k.lower()]
            if research_keys:
                findings = [extracted[k] for k in research_keys if extracted[k]]
                if findings:
                    # Summarize if too long
                    combined = "\n\n".join(str(f)[:500] for f in findings[:3])
                    results.append(f"📚 Research findings:\n{combined}")
            
            # Browser findings
            browser_keys = [k for k in extracted.keys() if 'browser' in k.lower()]
            if browser_keys:
                for k in browser_keys:
                    if extracted[k]:
                        results.append(f"🌐 Browser: {str(extracted[k])[:300]}")
            
            # OS findings
            os_keys = [k for k in extracted.keys() if 'os_' in k.lower()]
            if os_keys:
                for k in os_keys:
                    if extracted[k]:
                        results.append(f"💻 OS: {str(extracted[k])[:300]}")
            
            # Implementation plan summary
            if "implementation_plan" in extracted:
                plan = extracted["implementation_plan"]
                if isinstance(plan, dict):
                    results.append(f"📋 Plan: {plan.get('goal_analysis', 'Created execution plan')}")
            
            # Generic fallback for any other data
            if not results:
                import json
                return f"Collected data:\n{json.dumps(extracted, indent=2)[:2000]}"
            
            return "\n\n".join(results)
        
        # 3. Check last few messages for useful content
        messages = state.get("messages", [])
        if messages:
            # Look at last few messages for AI responses with substance
            for msg in reversed(messages[-5:]):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                # Skip empty or very short messages
                if content and len(content) > 50:
                    # Skip tool output messages
                    if not content.startswith("Tool output:"):
                        return content[:2000]
        
        # 4. Check for task completion indicators
        if state.get("task_complete"):
            step_count = state.get("step_count", 0)
            return f"Task completed after {step_count} steps (no explicit result captured)"
        
        return "No result (task may not have completed)"
    

