"""
Planner Agent - First step in Planning-First Architecture.

Runs FIRST on every prompt to create an implementation plan
that all subsequent agents reference.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class PlannerAgentNode(BaseAgent):
    """Planner agent that creates implementation plans.
    
    Runs as the FIRST step for every user prompt, analyzing the goal
    and creating a structured plan for other agents to follow.
    """
    
    AGENT_NAME = "planner"
    MAX_STEPS_PER_INVOCATION = 1  # One-shot planning
    
    SYSTEM_PROMPT = """You are a PLANNER agent. You run FIRST on every task to create an implementation plan.

CORE PRINCIPLE: Decompose goals into steps, route via simple if-then rules.

Your job is to:
1. Analyze the user's goal
2. Break it down into concrete steps
3. Assign the right agent to each step
4. Define success criteria

AVAILABLE AGENTS:
- research: Search internet, visit websites, gather information
- browser: Navigate to specific URLs, interact with web pages
- code: Analyze code, read source files, understand projects
- os: File operations, run shell commands, list directories
- data: Convert formats (JSON/CSV/YAML), compress/extract files
- network: Ping, DNS lookup, HTTP requests, SSL checks
- sysadmin: Check services, processes, disk/memory usage, logs
- media: Video/audio conversion, image resize/compress
- package: Python venv, pip install, git clone, project setup
- automation: Desktop notifications, reminders, timers

OUTPUT FORMAT (JSON only):
{
  "goal_analysis": "What the user wants to achieve",
  "steps": [
    {
      "step": 1,
      "agent": "research",
      "action": "Search for information about X",
      "success_criteria": "Found at least 3 relevant sources"
    },
    {
      "step": 2,
      "agent": "code",
      "action": "Analyze the collected information",
      "success_criteria": "Summary generated"
    }
  ],
  "estimated_agents": ["research", "code"],
  "constraints": {
    "max_sources": 5,
    "time_sensitive": false
  }
}

RULES:
- CHECK MEMORY FIRST: 
  1. I will auto-inject PROVEN STRATEGIES from your encrypted bank.
  2. I will auto-inject MISTAKES TO AVOID from your Apocalypse bank.
  3. I will also show RAW INSIGHTS from recent runs for new ideas.
  4. PRIORITIZE encrypted banks (they're battle-tested), but consider raw insights for innovation!
- Create a HIGHLY GRANULAR plan (aim for 20-30 steps for complex tasks)
- Break actions down: "Research X" -> "Search for X", "Click Result 1", "Verify Content", "Search Result 2", "Synthesize"
- Apply SYSTEMS THINKING: consider dependencies, edge cases, and verification
- Each step must have exactly ONE agent
- Be specific about success criteria
- Include explicit VERIFICATION steps after major actions
- Do not be afraid to create long, detailed plans. DETAILED IS BETTER."""
    
    def __init__(self, config):
        """Initialize planner agent."""
        super().__init__(config)
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def execute(self, state: AgentState) -> AgentState:
        """Create implementation plan for the goal.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with implementation_plan
        """
        goal = state.get("goal", "")
        
        print(f"[PLANNER] 📋 Creating plan for: {goal[:50]}...")
        
        # Build context for planning (simplified for faster response)
        context = f"""
USER GOAL: {goal}

Create a step-by-step plan. Keep it focused and practical (3-8 steps for most tasks).

Output as JSON:
{{"goal_analysis": "what user wants", "steps": [{{"step": 1, "agent": "research", "action": "what to do", "success_criteria": "how to verify"}}], "estimated_agents": ["research"]}}
"""
        
        messages = self._build_messages(state, context)
        
        try:
            # We first try to see if the model WANTS to use a tool to look up history
            # But the planner's main job is generating JSON.
            # OPTION 1: Just prompt it to include "search_runs" in its thought process? 
            # OPTION 2: Two-step planning: (1) Research history (2) Create plan.
            # Let's stick to the prompt instruction "CHECK HISTORY FIRST" and let it hallucinate a tool call if it wants?
            # Actually, `safe_invoke` doesn't support tools natively for PlannerAgent (returns AIMessage).
            # We'll just run a background search for relevant history automatically and inject it into context.
            
            # TIERED RECALL INJECTION (Strategies > Apocalypse > Raw Runs)
            # Skip if --no-memory flag is set for faster startup
            history_context = ""
            if not getattr(self.config, 'no_memory', False):
                try:
                    from ..knowledge_base import get_knowledge_base
                    kb = get_knowledge_base()
                    recall_result = kb.tiered_recall_async("planner", goal)
                    history_context = recall_result.to_prompt_injection()
                    
                    if history_context:
                        context += f"\n\n{history_context}"
                        messages = self._build_messages(state, context)
                        print("[PLANNER] 🧠 Injected tiered recall context")
                except Exception as e:
                    print(f"[PLANNER] ⚠️ Tiered recall failed: {e}")
            
            response = self.safe_invoke(messages)
            
            # Check for empty response
            if not response or not response.content or not response.content.strip():
                print("[PLANNER] ⚠️ Model returned empty response - using fallback plan")
                fallback_plan = self._create_fallback_plan(state)
                return self._update_state(
                    state,
                    messages=[AIMessage(content=json.dumps(fallback_plan))],
                    extracted_data={"implementation_plan": fallback_plan},
                    implementation_plan=fallback_plan,
                    error="Model returned empty response",
                )
            
            # Update token usage
            token_usage = self.update_token_usage(state, response)
            
            plan = self._parse_plan(response.content)
            
            print(f"[PLANNER] Created plan with {len(plan.get('steps', []))} steps")
            print(f"[PLANNER] Agents needed: {plan.get('estimated_agents', [])}")
            
            # Determine first agent from plan
            first_agent = "supervisor"  # Default fallback
            if plan.get("steps") and len(plan["steps"]) > 0:
                first_agent = plan["steps"][0].get("agent", "research")
            
            # Store plan and route to supervisor
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content)],
                extracted_data={"implementation_plan": plan},
                implementation_plan=plan,
                token_usage=token_usage,
            )
            
        except Exception as e:
            print(f"[PLANNER] Error creating plan: {e}")
            # Create minimal fallback plan
            fallback_plan = self._create_fallback_plan(state)
            return self._update_state(
                state,
                messages=[AIMessage(content=json.dumps(fallback_plan))],
                extracted_data={"implementation_plan": fallback_plan},
                implementation_plan=fallback_plan,
                error=f"Plan generation error: {str(e)}",
            )
    
    def _parse_plan(self, response: str) -> dict:
        """Parse LLM response into plan dict."""
        import re
        
        try:
            content = response.strip()
            
            # Strip <think>...</think> reasoning tags from thinking models
            if "<think>" in content:
                think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
                content = think_pattern.sub('', content).strip()
                print("[PLANNER] Stripped thinking blocks from response")
            
            # If content is empty after stripping, the model only returned thinking
            if not content or content.isspace():
                print("[PLANNER] ⚠️ Model only returned thinking - no actual plan")
                raise json.JSONDecodeError("Empty content after stripping think tags", "", 0)
            
            # Handle markdown code blocks
            if "```" in content:
                lines = content.split("\n")
                in_block = False
                json_lines = []
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)
            
            # Extract JSON
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            plan = json.loads(content)
            
            # Validate structure
            if "steps" not in plan:
                plan["steps"] = []
            if "goal_analysis" not in plan:
                plan["goal_analysis"] = "Goal analysis not provided"
            if "estimated_agents" not in plan:
                plan["estimated_agents"] = list(set(
                    s.get("agent", "research") for s in plan.get("steps", [])
                ))
            
            return plan
            
        except json.JSONDecodeError as e:
            print(f"[PLANNER] JSON parse error: {e}")
            # Return a generic plan
            return {
                "goal_analysis": "Could not parse plan",
                "steps": [
                    {"step": 1, "agent": "research", "action": "Investigate the goal", "success_criteria": "Information gathered"}
                ],
                "estimated_agents": ["research"],
                "constraints": {},
            }
    
    def _create_fallback_plan(self, state: AgentState) -> dict:
        """Create a minimal fallback plan when planning fails."""
        goal = state.get("goal", "").lower()
        
        # Heuristic-based fallback
        if any(kw in goal for kw in ["search", "find", "research", "look up"]):
            agent = "research"
        elif any(kw in goal for kw in ["file", "directory", "folder", "run"]):
            agent = "os"
        elif any(kw in goal for kw in ["code", "analyze", "function"]):
            agent = "code"
        elif any(kw in goal for kw in ["ping", "network", "dns"]):
            agent = "network"
        else:
            agent = "research"  # Default
        
        return {
            "goal_analysis": f"Fallback plan for: {state.get('goal', 'Unknown goal')}",
            "steps": [
                {"step": 1, "agent": agent, "action": "Execute the goal", "success_criteria": "Task completed"}
            ],
            "estimated_agents": [agent],
            "constraints": {},
        }
    
    def _update_state(
        self,
        state: AgentState,
        messages: list = None,
        extracted_data: dict = None,
        implementation_plan: dict = None,
        error: str = None,
        token_usage: dict = None,
    ) -> AgentState:
        """Create updated state with plan."""
        updates = {
            "step_count": state["step_count"] + 1,
            "active_agent": self.AGENT_NAME,
            "current_domain": "supervisor",  # Route to supervisor after planning
        }
        
        if messages:
            updates["messages"] = messages
        
        if extracted_data:
            merged = {**state.get("extracted_data", {}), **extracted_data}
            updates["extracted_data"] = merged
        
        if implementation_plan:
            updates["implementation_plan"] = implementation_plan
        
        if error:
            updates["error"] = error
        else:
            updates["error"] = None
            
        if token_usage:
            updates["token_usage"] = token_usage
        
        return {**state, **updates}


def planner_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for planner agent."""
    from ..tool_registry import get_tools
    
    # Get config from registry
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig()
    
    agent = PlannerAgentNode(agent_config)
    
    # Initialize RecallTool
    from ..run_history import RecallTool
    
    # Check if session store exists in registry/runner context?
    # Actually, we can just create a new read-only connection or get it if we passed it in.
    # The clean way is to let RecallTool manage its own connection for reading.
    agent.recall_tool = RecallTool()
    
    return agent.execute(state)
