"""
Base agent class for specialized agent nodes.

Provides common functionality for all agent types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

from ...cost import calculate_cost

logger = logging.getLogger("agentic_browser.agents")

# Lazy import for optional providers to avoid import errors if not installed
def _get_anthropic_client():
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic
    except ImportError:
        return None

def _get_google_client():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI
    except ImportError:
        return None

from ..state import AgentState
from ...config import AgentConfig

# Configure debug logging
logger = logging.getLogger("agentic_browser.agents")
DEBUG_MODE = True  # Set to False in production


def create_llm_client(config: AgentConfig, max_tokens: int = 1000):
    """Create the appropriate LangChain LLM client based on provider detection.
    
    Detects provider from endpoint URL or model name and returns the
    appropriate client (ChatOpenAI, ChatAnthropic, or ChatGoogleGenerativeAI).
    """
    model_lower = (config.model or "").lower()
    endpoint_lower = (config.model_endpoint or "").lower()
    
    # Detect provider from endpoint or model name
    # CRITICAL FIX: If endpoint is local (LM Studio/Ollama), treating it as Anthropic Native will fail
    # because they provide OpenAI-compatible APIs. 
    is_local_endpoint = "localhost" in endpoint_lower or "127.0.0.1" in endpoint_lower or "0.0.0.0" in endpoint_lower
    
    # Only use native ChatAnthropic if:
    # 1. Endpoint explicitly says anthropic (official or proxy)
    # 2. OR Model is claude AND endpoint is NOT local (defaulting to official)
    is_anthropic = "anthropic" in endpoint_lower or ("claude" in model_lower and not is_local_endpoint)
    
    is_google = "googleapis" in endpoint_lower or "gemini" in model_lower
    is_gpt5 = any(x in model_lower for x in ["gpt-5", "gpt5"])
    is_reasoning_model = any(x in model_lower for x in ["o1", "o3", "o4"])
    
    # Detect EXPLICIT thinking/reasoning models that use <think> tags
    # NOT vision-language models like qwen3-vl which are just VL, not thinking
    # Thinking models explicitly have "/think", ":think", or "deepseek-r1" patterns
    is_thinking_model = any(x in model_lower for x in [
        "/think", ":think", "-think",  # Explicit thinking mode suffixes
        "deepseek-r1", "deepseek_r1",  # DeepSeek reasoning model
        "-cot", "cot-", "/cot"  # Chain-of-thought modes
    ])
    
    # Set appropriate max_tokens for GPT-5
    if is_gpt5:
        max_tokens = max(max_tokens, 4000)
    
    # Thinking models need more tokens for their reasoning chain
    if is_thinking_model:
        max_tokens = max(max_tokens, 4000)
    
    if is_anthropic:
        # Use ChatAnthropic for Anthropic models
        ChatAnthropic = _get_anthropic_client()
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
        
        # Ensure we respect the endpoint if it's set (e.g. for proxies)
        anthropic_kwargs = {
            "api_key": config.api_key or "not-required",
            "model": config.model.strip(),  # Strip whitespace
            "max_tokens": max_tokens,
            "default_headers": {"anthropic-version": "2023-06-01"}
        }
        
        # Only pass base_url/endpoint if it's explicitly set and different from default
        # This fixes issues where config might have a lingering default that breaks things
        if config.model_endpoint and "api.anthropic.com" not in config.model_endpoint:
             anthropic_kwargs["anthropic_api_url"] = config.model_endpoint
        
        return ChatAnthropic(**anthropic_kwargs)
    
    elif is_google:
        # Use ChatGoogleGenerativeAI for Google models
        ChatGoogle = _get_google_client()
        if ChatGoogle is None:
            raise ImportError("langchain-google-genai not installed. Run: pip install langchain-google-genai")
        
        return ChatGoogle(
            google_api_key=config.api_key or "not-required",
            model=config.model.strip(),
            max_output_tokens=max_tokens,
        )
    
    else:
        # Default: Use ChatOpenAI (works for OpenAI and OpenAI-compatible endpoints)
        # Default timeout: 2 minutes (120s) - raised to 5 min for explicit thinking models
        timeout = 300 if is_thinking_model else 120
        
        llm_kwargs = {
            "base_url": config.model_endpoint,
            "api_key": config.api_key or "not-required",
            "model": config.model.strip(),
            "max_tokens": max_tokens,
            "request_timeout": timeout,
        }
        
        # O-series reasoning models (o1, o3, o4) require special handling:
        # - They require temperature=1 (not None, not omitted)
        # - They need higher max_tokens for reasoning
        if is_reasoning_model:
            llm_kwargs["temperature"] = 1  # o-series REQUIRES temperature=1
            llm_kwargs["max_tokens"] = max(max_tokens, 16384)  # Higher limit for reasoning
        elif is_thinking_model:
            # Thinking models work with low temperature
            llm_kwargs["temperature"] = 0.1
            logger.info(f"Detected thinking model: {config.model} - using 5min timeout")
        else:
            llm_kwargs["temperature"] = 0.1
        
        return ChatOpenAI(**llm_kwargs)


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
        
        # Use factory function to create provider-appropriate LLM client
        self.llm = create_llm_client(config, max_tokens=1000)

    async def _invoke_llm_async(self, messages: list) -> Any:
        """Invoke LLM asynchronously using ainvoke (Phase 7)."""
        try:
            # All LangChain LLMs support ainvoke
            return await self.llm.ainvoke(messages)
        except Exception as e:
            print(f"[AGENT] Async LLM invocation failed: {e}")
            return self.llm.invoke(messages)
    
    def invoke_llm_with_timeout(self, messages: list, timeout_s: float = 60.0) -> Any:
        """Invoke LLM with timeout protection (Phase 7)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.llm.invoke(messages)
        return self.llm.invoke(messages)
    
    def update_token_usage(self, state: AgentState, response: AIMessage) -> dict:
        """Calculate and return updated token usage stats.
        
        Args:
            state: Current state
            response: LLM response
            
        Returns:
            Updated token_usage dict
        """
        current = state.get("token_usage", {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "total_cost": 0.0,
        })
        
        usage = getattr(response, "usage_metadata", {}) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        if not input_tokens and not output_tokens:
            return current
            
        cost = calculate_cost(self.config.model, input_tokens, output_tokens)
        
        new_usage = {
            "input_tokens": current["input_tokens"] + input_tokens,
            "output_tokens": current["output_tokens"] + output_tokens,
            "total_tokens": current["total_tokens"] + input_tokens + output_tokens,
            "total_cost": current["total_cost"] + cost,
        }
        
        # Emit structured execution log for GUI
        import json
        try:
            log_data = {
                "type": "usage_update",
                "model": self.config.model,
                "input": input_tokens,
                "output": output_tokens,
                "total_input": new_usage["input_tokens"],
                "total_output": new_usage["output_tokens"],
                "cost": cost,
                "total_cost": new_usage["total_cost"]
            }
            print(f"__GUI_EVENT__{json.dumps(log_data)}")
        except Exception:
            pass
            
        return new_usage

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent type."""
        pass
    
    def safe_invoke(self, messages: list) -> AIMessage:
        """Invoke LLM with fallback handling for 404 errors and empty responses.
        
        If the configured model is not found (404), this will:
        1. Log the error
        2. Switch to a fallback model (e.g. claude-3-haiku)
        3. Re-initialize the LLM client
        4. Retry the invocation
        
        Args:
            messages: List of messages to send
            
        Returns:
            AIMessage response
            
        Raises:
            Exception: If retry fails or unrelated error
        """
        import json
        
        def notify_gui(status_msg: str, is_error: bool = False):
            try:
                event = {
                    "type": "status_update",
                    "status": status_msg,
                    "level": "error" if is_error else "warning"
                }
                print(f"__GUI_EVENT__{json.dumps(event)}")
            except:
                pass

        try:
            # Simple retry loop for 429 rate limits
            max_retries = 3  # Increased from 2 for better recovery
            for attempt in range(max_retries + 1):
                try:
                    response = self.llm.invoke(messages)
                    
                    # Handle empty responses
                    if response is None or (hasattr(response, 'content') and not response.content):
                        notify_gui("Model returned empty response, using fallback", is_error=True)
                        print("[WARN] LLM returned empty response, providing fallback")
                        return AIMessage(content='{"action": "done", "args": {"summary": "Unable to process - model returned empty response"}}')
                    
                    return response

                except Exception as inner_e:
                    err_str = str(inner_e).lower()
                    
                    # Detect provider for specific error handling
                    endpoint_lower = (self.config.model_endpoint or "").lower()
                    model_lower = (self.config.model or "").lower()
                    is_local = "localhost" in endpoint_lower or "127.0.0.1" in endpoint_lower
                    is_anthropic = "anthropic" in endpoint_lower or ("claude" in model_lower and not is_local)

                    # Check for 429 Rate Limit
                    if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                        # ANTHROPIC SPECIFIC: Wait & Retry because of strict TPM limits
                        if is_anthropic and attempt < max_retries:
                            wait_time = 20 * (attempt + 1)  # 20s, then 40s
                            msg = f"Anthropic Rate limit hit (429). Waiting {wait_time}s..."
                            print(f"[WARN] {msg}")
                            notify_gui(msg, is_error=False)
                            import time
                            time.sleep(wait_time)
                            continue
                        
                        # OPENAI/OTHER PROVIDERS: Wait 5s minimum (OpenAI needs ~3s)
                        elif attempt < max_retries:
                            wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s (was 2s, 4s)
                            print(f"[WARN] Rate limit hit (429). Waiting {wait_time}s...")
                            notify_gui(f"Rate limit (429). Waiting {wait_time}s...", is_error=False)
                            import time
                            time.sleep(wait_time)
                            continue
                    
                    # Re-raise if not a rate limit or retries exhausted
                    raise inner_e
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle empty response errors - catch all variations
            empty_patterns = ["empty", "must contain", "output text", "tool calls", "cannot both be empty"]
            if any(p in error_msg for p in empty_patterns):
                notify_gui(f"Empty/invalid response: {str(e)[:50]}...", is_error=True)
                print(f"[WARN] Empty/invalid response error: {e}")
                return AIMessage(content='{"action": "done", "args": {"summary": "Model returned empty response - try a different model"}}')
            
            # Check for 404 / model not found errors
            if "404" in error_msg or "not_found" in error_msg or "model" in error_msg and "not found" in error_msg:
                print(f"[WARN] detailed error: {str(e)}")
                print(f"[WARN] Model {self.config.model} not found. Attempting fallback...")
                notify_gui(f"Model {self.config.model} not found. Falling back...", is_error=True)
                
                # Determine fallback model based on current provider/model
                fallback_model = None
                
                if "claude" in (self.config.model or ""):
                    # Fallback chain for Anthropic
                    if "sonnet-20241022" in self.config.model:
                        fallback_model = "claude-3-5-sonnet-20240620"
                    elif "sonnet" in self.config.model:
                        fallback_model = "claude-3-haiku-20240307"
                    else:
                        fallback_model = "claude-3-haiku-20240307"
                elif "gpt" in (self.config.model or ""):
                    # Fallback for OpenAI
                    fallback_model = "gpt-4o-mini"
                elif "gemini" in (self.config.model or ""):
                    # Fallback for Google
                    fallback_model = "gemini-1.5-flash"
                
                if fallback_model and fallback_model != self.config.model:
                    print(f"[INFO] Switching to fallback model: {fallback_model}")
                    notify_gui(f"Switching to fallback: {fallback_model}")
                    
                    # Update config and re-initialize LLM
                    self.config.model = fallback_model
                    self.llm = create_llm_client(self.config, max_tokens=1000)
                    
                    # Retry
                    try:
                        return self.llm.invoke(messages)
                    except Exception as retry_err:
                        print(f"[WARN] Retry also failed: {retry_err}")
                        notify_gui(f"Fallback failed: {str(retry_err)[:50]}", is_error=True)
                        return AIMessage(content='{"action": "done", "args": {"summary": "Model error during retry"}}')
            
            # Catch-all: return a fallback response instead of crashing
            print(f"[WARN] Unhandled LLM error, returning fallback: {e}")
            notify_gui(f"LLM Error: {str(e)[:50]}", is_error=True)
            return AIMessage(content='{"action": "done", "args": {"summary": "An error occurred with the model"}}')

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's task.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state
        """
        pass
    
    def capture_screenshot_base64(self, browser_tools) -> str | None:
        """Capture a screenshot and return as base64 string.
        
        Args:
            browser_tools: BrowserTools instance with page access
            
        Returns:
            Base64-encoded PNG string, or None if failed
        """
        import base64
        
        if not self.config.vision_mode:
            return None
        
        try:
            if browser_tools and hasattr(browser_tools, 'page') and browser_tools.page:
                screenshot_bytes = browser_tools.page.screenshot(full_page=False)
                return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            print(f"[VISION] Screenshot capture failed: {e}")
        
        return None
    
    def build_vision_message(self, text_content: str, screenshot_b64: str | None) -> HumanMessage:
        """Build a HumanMessage with optional vision content.
        
        Args:
            text_content: The text prompt
            screenshot_b64: Base64-encoded screenshot (or None)
            
        Returns:
            HumanMessage with text and optionally image
        """
        if screenshot_b64 and self.config.vision_mode:
            # Multi-modal message with image
            return HumanMessage(content=[
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": "low"  # low detail to save tokens (high uses ~5k extra)
                    }
                }
            ])
        else:
            # Text-only message
            return HumanMessage(content=text_content)
    
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
        
        # "Infinite Context" Implementation with Provider-Based Safety Limits:
        # 1. Summarize ancient history (older than last 10 steps)
        # 2. Keep recent history (last 10 steps) but enforce total size limit
        # 3. Aggressively strip content if over safe limits (avoid 413 errors)
        
        all_msgs = state["messages"]
        
        # CRITICAL: Hard cap on messages to prevent unbounded growth
        # operator.add accumulates forever without this
        # VISION MODE: Use stricter limit since screenshots add ~500 tokens each
        is_vision = self.config.vision_mode if hasattr(self.config, 'vision_mode') else False
        MAX_MESSAGES = 20 if is_vision else 40
        
        if len(all_msgs) > MAX_MESSAGES:
            all_msgs = all_msgs[-MAX_MESSAGES:]
            
        # Strip image_url content from older messages to reduce memory
        # Only keep text from messages beyond the last 5
        pruned_msgs = []
        for i, msg in enumerate(all_msgs):
            is_recent = i >= len(all_msgs) - 5
            
            if is_recent:
                 # Keep recent messages fully intact
                pruned_msgs.append(msg)
            else:
                # COMPACT OLDER MESSAGES (Phase 5 Optimization)
                # 1. Convert complex/list content to simple string
                # 2. Truncate to 500 chars
                content = msg.content
                if isinstance(content, list):
                    # Multi-modal -> Text only
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = " ".join(text_parts)
                
                # Truncate large text
                content_str = str(content)
                if len(content_str) > 500:
                    content_str = content_str[:500] + "...[truncated]"
                
                # Reconstruct as simplified text-only message
                if isinstance(msg, HumanMessage):
                    pruned_msgs.append(HumanMessage(content=content_str))
                elif isinstance(msg, AIMessage):
                    pruned_msgs.append(AIMessage(content=content_str))
                else:
                    # System/Tool messages - keep as is but with truncated content if possible
                    # Or just forcing a generic message type might be safer, but let's try to preserve type if it has content
                    if hasattr(msg, 'content'):
                        msg.content = content_str
                    pruned_msgs.append(msg)
        all_msgs = pruned_msgs

        
        recent_count = 10
        
        # Check provider type for safety limits
        endpoint_lower = (self.config.model_endpoint or "").lower()
        model_lower = (self.config.model or "").lower()
        is_local = "localhost" in endpoint_lower or "127.0.0.1" in endpoint_lower
        is_anthropic = "anthropic" in endpoint_lower or ("claude" in model_lower and not is_local)
        
        # Determine total estimated size
        total_chars = sum(len(str(m.content)) for m in all_msgs)
        
        recent_count = 10
            
        # If history is long, summarize the ancient part
        if len(all_msgs) > recent_count:
            ancient_msgs = all_msgs[:-recent_count]
            summary = self._summarize_history(ancient_msgs)
            if summary:
                messages.append(SystemMessage(content=f"PREVIOUS ACTION HISTORY (Summarized):\n{summary}"))
            
            # Process only the recent messages
            recent_msgs = all_msgs[-recent_count:]
        else:
            recent_msgs = all_msgs

        for i, msg in enumerate(recent_msgs):
            # Is this an older message (not the very last 1-2)?
            if i < len(recent_msgs) - 2:
                # If it's a HumanMessage with large page content, strip it
                content_str = str(msg.content)
                
                # Check for "Visible content" block 
                if isinstance(msg, HumanMessage) and "Visible content (truncated):" in content_str:
                    # Simple heuristic to strip likely page content block
                    start = content_str.find("Visible content (truncated):")
                    # Find end of block - usually "Data collected:" or end of string
                    end = content_str.find("Data collected:", start)
                    if end == -1:
                        end = len(content_str)
                    
                    # Replace the heavy block
                    new_content = content_str[:start] + "\n[Page content stripped from history to save context]\n\n" + content_str[end:]
                    messages.append(HumanMessage(content=new_content))
                    continue
                
                # Also strip raw tool outputs if they are huge (reduced from 5000)
                if len(content_str) > 2000:
                     new_content = content_str[:500] + f"\n... [{len(content_str)-1000} chars truncated] ...\n" + content_str[-500:]
                     messages.append(HumanMessage(content=new_content))
                     continue
            
            # Default: append full message
            messages.append(msg)
        
        # Build plan context if plan exists (Planning-First Architecture)
        plan_context = ""
        plan = state.get("implementation_plan")
        if plan and isinstance(plan, dict):
            steps = plan.get("steps", [])
            plan_step_idx = state.get("plan_step_index", 0)
            
            plan_context = "=== IMPLEMENTATION PLAN ===\n"
            
            # Format Plan with GC (Collapsing completed steps)
            for i, step in enumerate(steps):
                step_num = i + 1
                status = step.get("status", "pending")
                agent_name = step.get("agent", "unknown")
                description = step.get("description", "")[:200]  # Limit description length
                
                if i < plan_step_idx:
                    # Completed: Concise summary
                    outcome = step.get("outcome", "Completed")[:100]
                    plan_context += f"✔ Step {step_num} [{agent_name}]: {outcome}\n"
                elif i == plan_step_idx:
                    # Current: Detailed view
                    plan_context += f"\n👉 CURRENT STEP {step_num} ({agent_name}):\n"
                    plan_context += f"   Task: {description}\n"
                    plan_context += f"   Success Criteria: {step.get('success_criteria', 'Task completed')[:150]}\n"
                    plan_context += f"   Notes: {step.get('notes', '')[:150]}\n\n"
                else:
                    # Future: List view
                    plan_context += f"○ Step {step_num} [{agent_name}]: {description}\n"
            
            plan_context += "===========================\n"
            
            # CRITICAL: Guard against plan_context explosion
            MAX_PLAN_CONTEXT = 3000
            if len(plan_context) > MAX_PLAN_CONTEXT:
                plan_context = plan_context[:MAX_PLAN_CONTEXT] + "\n...[Plan truncated]\n==="
        
        # Add current task context
        context = f"""
Current Task: {state['goal']}
Step: {state['step_count']} / {state['max_steps']}
Previous Domain: {state['current_domain']}
{plan_context}
{task_context}
"""
        # GUARD: If current context itself is huge, truncate it
        # This is a safety net - the task_context guards should prevent this
        MAX_CONTEXT_LEN = 12000
        if len(context) > MAX_CONTEXT_LEN:
            context = context[:MAX_CONTEXT_LEN] + "\n...[Current context truncated]"
            print(f"[CONTEXT] Current prompt truncated to {MAX_CONTEXT_LEN} chars")
            
        messages.append(HumanMessage(content=context))
        
        return messages
    
    def _synthesize_context(self, messages: list, state: AgentState) -> str:
        """Synthesize the most important information from message history.
        
        Extracts:
        - Key findings and data collected
        - Visited URLs and their relevance
        - Important errors to avoid
        - Current progress toward goal
        
        Target: 2k-3.5k chars (max 5k if essential)
        """
        import json
        
        synthesis_parts = []
        visited_sites = []
        key_findings = []
        errors_seen = []
        actions_taken = []
        
        for msg in messages:
            content = str(msg.content) if hasattr(msg, 'content') else ""
            
            # Extract from AI messages (actions)
            if isinstance(msg, AIMessage):
                try:
                    data = json.loads(content)
                    action = data.get("action", "")
                    args = data.get("args", {})
                    
                    if action == "goto":
                        url = args.get("url", "")
                        if url and "duckduckgo" not in url.lower():
                            domain = url.split('/')[2] if '//' in url else url[:40]
                            if domain not in visited_sites:
                                visited_sites.append(domain)
                    elif action == "extract_visible_text":
                        actions_taken.append("extracted text")
                    elif action == "done":
                        summary = args.get("summary", "")
                        if summary and "error" not in summary.lower():
                            key_findings.append(summary[:200])
                except:
                    pass
                    
            # Extract from Human messages (results, errors)
            elif isinstance(msg, HumanMessage):
                # Look for extracted data markers
                if "research_source" in content.lower() or "collected" in content.lower():
                    # Extract just the key data, not verbose content
                    lines = content.split('\n')
                    for line in lines[:5]:  # First 5 lines only
                        if ':' in line and len(line) < 150:
                            key_findings.append(line.strip()[:100])
                            
                if "Error:" in content or "failed" in content.lower():
                    # Extract error summary
                    error_line = content.split('\n')[0][:100]
                    if error_line not in errors_seen:
                        errors_seen.append(error_line)
        
        # Build synthesis - prioritize brevity
        if visited_sites:
            synthesis_parts.append(f"Sites visited: {', '.join(visited_sites[:8])}")
        
        # Check extracted_data in state (gold data)
        extracted = state.get("extracted_data", {})
        if extracted:
            data_keys = list(extracted.keys())[:5]
            if data_keys:
                synthesis_parts.append(f"Data collected: {', '.join(data_keys)}")
                # Add brief summary of each (max 100 chars each)
                for key in data_keys[:3]:
                    val = str(extracted[key])[:150]
                    synthesis_parts.append(f"  • {key}: {val}...")
        
        if key_findings:
            synthesis_parts.append("Key findings:")
            for finding in key_findings[:5]:
                synthesis_parts.append(f"  • {finding}")
        
        if errors_seen:
            synthesis_parts.append(f"Errors to avoid: {len(errors_seen)} issues seen")
        
        # Calculate size and adjust
        result = "\n".join(synthesis_parts)
        
        # Strictly limit to 3.5k-5k chars as per requirements
        MAX_SYNTHESIS = 4500
        if len(result) > MAX_SYNTHESIS:
            result = result[:MAX_SYNTHESIS] + "\n[Synthesis truncated for brevity]"
            
        return result
    
    def _llm_synthesize_context(self, messages: list, state: AgentState) -> str:
        """Use LLM to intelligently compress context when it exceeds 18k chars.
        
        Slower (~2-3s) but better semantic preservation than programmatic extraction.
        Falls back to programmatic synthesis if LLM fails.
        
        Args:
            messages: All messages in history
            state: Current state
            
        Returns:
            Compressed context string (target: 3.5k-4.5k chars)
        """
        try:
            # Build a comprehensive context string from all messages
            full_context = ""
            for msg in messages:
                content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                full_context += f"{type(msg).__name__}: {content[:500]}...\n\n"
            
            # Add state info
            full_context += f"\n\nGoal: {state['goal']}\n"
            full_context += f"Visited URLs: {len(state.get('visited_urls', []))}\n"
            full_context += f"Data collected: {list(state.get('extracted_data', {}).keys())}\n"
            
            # Prompt for LLM
            compression_prompt = f"""Compress this browsing/research history to 3500-4500 characters while preserving ALL critical information.

FULL CONTEXT (may be very long):
{full_context[:8000]}

REQUIREMENTS:
1. Extract and preserve:
   - Sites/URLs visited (deduplicated domains only)
   - Key data collected or extracted
   - Important findings or summaries
   - Errors/failures to avoid
2. Target size: 3500-4500 characters (strict)
3. Be concise but don't lose critical navigation or data context
4. Format as bullet points for readability

Output the compressed context ONLY, no preamble."""

            # Use the LLM to compress
            from langchain_core.messages import SystemMessage as SysMsg, HumanMessage as HumMsg
            
            response = self.llm.invoke([
                SysMsg(content="You are a context compression expert. Preserve critical information while reducing size."),
                HumMsg(content=compression_prompt)
            ])
            
            compressed = response.content.strip()
            
            # Validate size
            if len(compressed) > 5000:
                print(f"[CONTEXT] LLM synthesis too large ({len(compressed)} chars), truncating")
                compressed = compressed[:4500] + "\n...[LLM synthesis truncated]"
            elif len(compressed) < 500:
                print(f"[CONTEXT] LLM synthesis too small ({len(compressed)} chars), falling back to programmatic")
                return self._synthesize_context(messages, state)
            
            print(f"[CONTEXT] LLM synthesis: {len(compressed)} chars (target 3.5k-4.5k)")
            return compressed
            
        except Exception as e:
            print(f"[CONTEXT] LLM synthesis failed ({e}), falling back to programmatic")
            return self._synthesize_context(messages, state)
    
    def _summarize_history(self, messages: list[BaseMessage]) -> str:
        """Condense messages into ultra-compressed summary (max 10 lines)."""
        import json
        summary_lines = []
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                try:
                    if hasattr(msg, 'content') and msg.content:
                        data = json.loads(str(msg.content))
                        action = data.get("action", "?")
                        args = data.get("args", {})
                        # Ultra-concise: only key info
                        if action == "click":
                            summary_lines.append(f"→ click: {args.get('selector', '')[:30]}")
                        elif action == "goto":
                            url = args.get('url', '')
                            domain = url.split('/')[2] if '/' in url else url[:30]
                            summary_lines.append(f"→ goto: {domain}")
                        elif action == "extract_visible_text":
                            summary_lines.append("→ extracted page")
                        elif action == "done":
                            summary_lines.append("→ DONE")
                        else:
                            summary_lines.append(f"→ {action}")
                except:
                    pass  # Skip unparseable
                    
            elif isinstance(msg, HumanMessage):
                content = str(msg.content)[:60]
                if "Visible content" in content or "Page content" in content:
                    continue  # Skip verbose page content messages
                if "Result:" in content or "Error:" in content:
                    summary_lines.append(content[:40])
        
        # Cap at 10 lines to prevent bloat
        return "\n".join(summary_lines[:10])
    
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
        token_usage: dict | None = None,
        step_update: dict | None = None,
    ) -> AgentState:
        """Create updated state with new information."""
        updates: dict[str, Any] = {
            "step_count": state["step_count"] + 1,
            "active_agent": self.AGENT_NAME,
        }
        
        # Handle plan updates
        if step_update and state.get("implementation_plan"):
            plan = state["implementation_plan"].copy()
            idx = state.get("plan_step_index", 0)
            if "steps" in plan and idx < len(plan["steps"]):
                # Update the specific step
                step = plan["steps"][idx]
                step["status"] = step_update.get("status", "completed")
                step["outcome"] = step_update.get("outcome", "")
                step["notes"] = step_update.get("notes", "")
                
                # If completed, move to next step automatically
                if step["status"] == "completed":
                     updates["plan_step_index"] = idx + 1
                
                updates["implementation_plan"] = plan
        
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
            
        if token_usage:
            updates["token_usage"] = token_usage
        
        return {**state, **updates}
