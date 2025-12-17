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
1. SEARCH: After typing in a search box, you MUST press Enter to submit
2. FORMS: After filling form fields, click the submit button  
3. NEVER repeat the same action - always progress to the next step
4. RESEARCH TASKS: When asked to "research", "compare", "find information", or "look up":
   - Visit EXACTLY 3 different sources (no more, no less)
   - Workflow: search → pick 3 links → goto site 1 → extract text → goto site 2 → extract text → goto site 3 → extract text → done
   - Use URLs from "Top Links" with goto action, not click
   - After visiting 3 sites and extracting content, IMMEDIATELY call "done" with summary
   - Do NOT summarize search engine snippets - go to the real sites!

5. CLICKING LINKS: If a click selector fails, use goto with a URL from Top Links instead

WHEN TO USE "done":
- For research: After visiting EXACTLY 3 sites and extracting their content
- You can provide a comprehensive answer based on content from the sites you visited
- The task is truly complete (e.g., clicked a button, made a purchase, completed a form)
- STOP researching and summarize - don't visit more than 3 sites!

WHEN NOT TO USE "done":
- You're on a search results page - visit actual sites first!
- You've only visited one source when multiple were requested
- You haven't actually read the content of the pages you found

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
    
    def _build_state_message(self, state: dict[str, Any]) -> str:
        """Build the current state message for the LLM."""
        # Format visited URLs
        visited_str = "\n".join(f"- {url}" for url in self.visited_urls[-10:]) if self.visited_urls else "(none yet)"
        
        # Format links
        links = state.get('top_links', [])
        links_str = "\n".join(
            f"- [{l.get('text', 'no text')}]({l.get('href', '#')})"
            for l in links[:15]
        ) if links else "(no links found)"
        
        # Format recent history
        history = state.get('recent_history', [])
        history_str = "\n".join(
            f"- {item.get('action', 'unknown')}: {item.get('result', '')}"
            for item in history
        ) if history else "(no previous actions)"
        
        # Build the message
        message = f"""Current Page State:
- Goal: {state.get('goal', '')}
- URL: {state.get('current_url', '')}
- Title: {state.get('page_title', '')}

Visible Text (truncated):
{state.get('visible_text', '')[:4000]}

Top Links:
{links_str}

⚠️ ALREADY VISITED URLs (do NOT click these again):
{visited_str}

Recent Actions:
{history_str}

Extracted Data So Far:
{json.dumps(state.get('extracted_data', {}), indent=2)}

{state.get('additional_context', '')}

What is your next action? Respond with JSON only."""
        
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
