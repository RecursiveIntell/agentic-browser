"""
LangChain-based LLM client for Agentic Browser.

Provides conversation memory and advanced context management using LangChain.
"""

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError

from .config import AgentConfig
from .llm_client import ActionResponse


class LangChainLLMClient:
    """LangChain-based client with conversation memory."""
    
    SYSTEM_PROMPT = """You are an autonomous browser agent that helps users accomplish goals on the web.

You receive the current page state and must respond with a SINGLE action to take next.

CRITICAL: You MUST respond with ONLY valid JSON, no markdown, no prose, no explanation outside the JSON.

Your response must follow this exact schema:
{
  "action": "goto|click|type|press|scroll|wait_for|extract|extract_visible_text|screenshot|back|forward|done",
  "args": { ... },
  "rationale": "short reason for this action",
  "risk": "low|medium|high",
  "requires_approval": true/false,
  "final_answer": "only when action=done - put your complete answer/summary here"
}

Action argument rules:
- goto: { "url": "https://..." }
- click: { "selector": "css or text selector", "timeout_ms": 10000 }
- type: { "selector": "...", "text": "...", "clear_first": true/false }
- press: { "key": "Enter|Tab|ArrowDown|..." }
- scroll: { "amount": 800 } (positive=down, negative=up)
- wait_for: { "selector": "...", "timeout_ms": 10000 } OR { "timeout_ms": 2000 }
- extract: { "selector": "...", "attribute": "innerText|href|value|..." }
- extract_visible_text: { "max_chars": 8000 }
- screenshot: { "label": "optional description" }
- back/forward: {}
- done: { "summary_style": "bullets|paragraph" }

CRITICAL WORKFLOW PATTERNS:

🔍 SEARCH ENGINE: ALWAYS USE DUCKDUCKGO (https://duckduckgo.com)
- Google BLOCKS AI agents with CAPTCHAs - NEVER use Google!
- DuckDuckGo is AI-friendly and won't block you
- For ANY search/research task, start with: goto("https://duckduckgo.com")
- Then type your query and press Enter

⚠️ URL VALIDATION - AVOID 404 ERRORS:
- Only use URLs that start with https:// or http://
- NEVER use partial URLs like "/about" or "example.com/page"
- If a URL looks malformed or truncated, DON'T use it
- If you get a 404, DON'T try variations - move to next site

1. SEARCH: After typing in a search box, you MUST press Enter to submit
2. FORMS: After filling form fields, click the submit button  
3. NEVER repeat the same action - always progress to the next step

4. RESEARCH TASKS: When asked to "research", "compare", "find information":
   - Start with DuckDuckGo search
   - Visit 2-3 sources from results
   - Only use COMPLETE URLs starting with https://
   - Skip broken/truncated links
   - Extract text from each site, then call "done"

5. CLICKING LINKS: If click fails, use goto with URL from Top Links

Risk classification:
- HIGH: purchases, payments, sending messages, account settings, deleting
- MEDIUM: logging in, uploading files, granting permissions
- LOW: navigation, reading, extracting data

Set requires_approval=true for HIGH risk and MEDIUM risk actions."""

    def __init__(self, config: AgentConfig):
        """Initialize the LangChain client.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        
        # Initialize the LangChain ChatOpenAI model
        # Works with any OpenAI-compatible API
        self.llm = ChatOpenAI(
            base_url=config.model_endpoint,
            api_key=config.api_key or "not-required",
            model=config.model,
            temperature=0.1,
            max_tokens=1000,
            request_timeout=60,
        )
        
        # Conversation history for full context
        self.message_history: list[HumanMessage | AIMessage] = []
        
        # Track important context
        self.visited_urls: list[str] = []
        self.extracted_summaries: list[str] = []
        
        # State pruning - store condensed summaries instead of full text
        self.page_summaries: dict[str, str] = {}  # URL -> summary
        self.step_count: int = 0
        self.last_visible_text: str = ""  # Cache to avoid re-summarizing same content
        
    def close(self) -> None:
        """Close the client (no-op for LangChain)."""
        pass
    
    def add_visited_url(self, url: str) -> None:
        """Track a visited URL."""
        if url and url not in self.visited_urls:
            self.visited_urls.append(url)
    
    def add_extraction(self, summary: str) -> None:
        """Track an extraction summary."""
        if summary and len(summary) < 500:
            self.extracted_summaries.append(summary[:500])
    
    def summarize_page_content(self, url: str, title: str, visible_text: str) -> str:
        """Summarize page content to reduce context size.
        
        Args:
            url: Current page URL
            title: Page title
            visible_text: Full visible text (potentially 8000+ chars)
            
        Returns:
            Condensed summary (200-400 chars) for context
        """
        # Return cached summary if we already summarized this URL
        if url in self.page_summaries:
            return self.page_summaries[url]
        
        # Skip summarization for very short content
        if len(visible_text) < 500:
            return visible_text
        
        # Use LLM to create a brief summary
        try:
            summary_prompt = f"""Summarize this webpage content in 2-3 sentences (max 200 words).
Focus on: key information, main topic, important data points.

Page: {title}
URL: {url}
Content:
{visible_text[:3000]}

Summary:"""
            
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content.strip()
            
            # Cache the summary
            self.page_summaries[url] = summary
            
            # Keep summaries cache manageable
            if len(self.page_summaries) > 10:
                # Remove oldest entries
                oldest_urls = list(self.page_summaries.keys())[:-10]
                for old_url in oldest_urls:
                    del self.page_summaries[old_url]
            
            return summary
            
        except Exception:
            # Fallback: truncate to first 500 chars
            return visible_text[:500] + "..."
    
    def _build_state_message(self, state: dict[str, Any]) -> str:
        """Build the current state message for the LLM with state pruning.
        
        Uses summaries instead of full text after the first few steps to keep
        context size manageable.
        """
        self.step_count += 1
        
        # Format visited URLs
        visited_str = "\n".join(f"- {url}" for url in self.visited_urls[-10:]) if self.visited_urls else "(none yet)"
        
        # Format links - only top 10 to reduce size
        links = state.get('top_links', [])
        links_str = "\n".join(
            f"- [{l.get('text', 'no text')[:40]}]({l.get('href', '#')})"
            for l in links[:10]
        ) if links else "(no links found)"
        
        # Format recent history with sliding window
        # Keep detailed history for last 3 actions, summarize older ones
        history = state.get('recent_history', [])
        if len(history) > 3:
            # Summarize older history
            old_actions = [h.get('action', '?') for h in history[:-3]]
            old_summary = f"(Earlier: {', '.join(old_actions)})"
            recent_history = history[-3:]
            history_str = old_summary + "\n" + "\n".join(
                f"- {item.get('action', 'unknown')}: {item.get('result', '')[:100]}"
                for item in recent_history
            )
        else:
            history_str = "\n".join(
                f"- {item.get('action', 'unknown')}: {item.get('result', '')[:100]}"
                for item in history
            ) if history else "(no previous actions)"
        
        # STATE PRUNING: Use summaries after step 2 to reduce context
        current_url = state.get('current_url', '')
        page_title = state.get('page_title', '')
        visible_text = state.get('visible_text', '')
        
        if self.step_count <= 2:
            # First 2 steps: full context for orientation
            visible_section = f"Visible Text:\n{visible_text[:3000]}"
        else:
            # Later steps: use summarized content
            page_summary = self.summarize_page_content(current_url, page_title, visible_text)
            visible_section = f"Page Summary:\n{page_summary}"
        
        # Condensed extracted data - just show keys and truncated values
        extracted = state.get('extracted_data', {})
        if extracted:
            extracted_lines = []
            for key, value in list(extracted.items())[-5:]:  # Last 5 extractions
                val_str = str(value)[:200] if isinstance(value, str) else str(value)[:100]
                extracted_lines.append(f"- {key}: {val_str}...")
            extracted_str = "\n".join(extracted_lines)
        else:
            extracted_str = "(none)"
        
        # Build compact message
        message = f"""Current State [Step {self.step_count}]:
- Goal: {state.get('goal', '')}
- URL: {current_url}
- Title: {page_title}

{visible_section}

Top Links:
{links_str}

⚠️ VISITED (do NOT revisit):
{visited_str}

Recent Actions:
{history_str}

Extracted Data:
{extracted_str}

{state.get('additional_context', '')}

Next action? JSON only."""
        
        return message
    
    def get_next_action(
        self, 
        state: dict[str, Any],
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ) -> ActionResponse:
        """Get the next action from the LLM using conversation memory.
        
        Args:
            state: Current page state
            max_retries: Maximum number of retries for invalid JSON
            system_prompt: Optional override for system prompt (e.g., OS mode)
            
        Returns:
            Validated action response
            
        Raises:
            ValueError: If unable to get valid JSON after retries
        """
        # Track current URL
        current_url = state.get('current_url', '')
        self.add_visited_url(current_url)
        
        # Build the state message
        state_message = self._build_state_message(state)
        
        # Use provided system prompt or default
        prompt = system_prompt if system_prompt else self.SYSTEM_PROMPT
        
        # Build messages including history
        messages = [
            SystemMessage(content=prompt),
            *self.message_history[-10:],  # Keep last 10 exchanges for context
            HumanMessage(content=state_message),
        ]
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Get response from LLM
                response = self.llm.invoke(messages)
                response_content = response.content
                
                # Parse the response
                action_response = self._parse_response(response_content)
                
                # Store in history for context
                self.message_history.append(HumanMessage(content=f"State: {state.get('page_title', 'Unknown')} - {state.get('current_url', '')}"))
                self.message_history.append(AIMessage(content=response_content))
                
                # Keep history manageable
                if len(self.message_history) > 20:
                    self.message_history = self.message_history[-20:]
                
                return action_response
                
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_retries:
                    # Add repair message
                    messages.append(AIMessage(content="I apologize, that was not valid JSON."))
                    messages.append(HumanMessage(content="""Your response was not valid JSON. Please respond with ONLY valid JSON:
{
  "action": "...",
  "args": { ... },
  "rationale": "...",
  "risk": "low|medium|high",
  "requires_approval": true/false,
  "final_answer": null or "..."
}"""))
                    continue
        
        raise ValueError(
            f"Failed to get valid JSON after {max_retries + 1} attempts: {last_error}"
        )
    
    def _parse_response(self, raw_response: str) -> ActionResponse:
        """Parse and validate the LLM response.
        
        Args:
            raw_response: Raw response string from the LLM
            
        Returns:
            Validated ActionResponse
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
            ValidationError: If validation fails
        """
        # Try to extract JSON from response
        content = raw_response.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Find JSON object
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end+1]
        
        # Parse JSON
        data = json.loads(content)
        
        # Validate with Pydantic
        return ActionResponse(**data)
    
    def get_recovery_action(
        self,
        state: dict[str, Any],
        error: str,
        failed_action: dict[str, Any],
    ) -> ActionResponse:
        """Get a recovery action after a failure.
        
        Args:
            state: Current page state
            error: Error message from the failed action
            failed_action: The action that failed
            
        Returns:
            A new action to try
        """
        # Add failure context to the state
        state = state.copy()
        state["additional_context"] = f"""
⚠️ PREVIOUS ACTION FAILED:
Action: {failed_action}
Error: {error}

YOU MUST TRY A DIFFERENT APPROACH:
- Use a simpler text-based selector like text="Link Text"
- Try clicking a different element
- Or use 'done' if you can answer from the visible text
DO NOT repeat the same failing action!"""
        
        return self.get_next_action(state)
    
    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        return f"""
Conversation History: {len(self.message_history)} messages
Visited URLs: {len(self.visited_urls)}
Extracted: {len(self.extracted_summaries)} items
"""
