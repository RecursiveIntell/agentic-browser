"""
LLM client for Agentic Browser.

Provides OpenAI-compatible API client with JSON parsing and retry logic.
"""

import json
import re
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from .config import AgentConfig
from .utils import extract_json_from_response


class ActionResponse(BaseModel):
    """Validated action response from the LLM."""
    
    action: str = Field(
        description="The action to take",
        pattern=r"^(goto|click|type|press|scroll|wait_for|extract|extract_visible_text|screenshot|back|forward|done)$"
    )
    args: dict[str, Any] = Field(default_factory=dict, description="Action arguments")
    rationale: str = Field(description="Short reason for this action")
    risk: str = Field(
        default="low",
        pattern=r"^(low|medium|high)$",
        description="Risk level"
    )
    requires_approval: bool = Field(
        default=False, 
        description="Whether this action requires user approval"
    )
    final_answer: Optional[str] = Field(
        default=None, 
        description="Final answer when action is 'done'"
    )


class LLMClient:
    """Client for OpenAI-compatible LLM APIs."""
    
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
  "final_answer": "only when action=done"
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

IMPORTANT WORKFLOW PATTERNS:
1. SEARCH: After typing in a search box, you MUST use press with key "Enter" to submit
   - First: type the search query
   - Then: press Enter to submit
2. FORMS: After filling form fields, click the submit button
3. NEVER repeat the same action twice - always progress to the next step

Selector tips:
- Use text= selectors for visible text: text="Click Me"
- Use CSS selectors for elements with IDs/classes
- If a selector fails, use extract_visible_text and screenshot to understand the page

Risk classification:
- HIGH: purchases, payments, sending messages, account settings, deleting
- MEDIUM: logging in, uploading files, granting permissions
- LOW: navigation, reading, extracting data

Set requires_approval=true for HIGH risk and MEDIUM risk actions.

When the goal is complete, use action="done" with a clear final_answer summarizing what was accomplished."""

    REPAIR_PROMPT = """Your previous response was not valid JSON. 
    
Please respond with ONLY valid JSON, no markdown code blocks, no explanation.
Just the raw JSON object starting with { and ending with }

The required format is:
{
  "action": "...",
  "args": { ... },
  "rationale": "...",
  "risk": "low|medium|high",
  "requires_approval": true/false,
  "final_answer": null or "..."
}"""

    def __init__(self, config: AgentConfig):
        """Initialize the LLM client.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.endpoint = config.model_endpoint.rstrip("/")
        self.model = config.model
        self.api_key = config.api_key
        
        # Build headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # HTTP client
        self.client = httpx.Client(timeout=60.0)
    
    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
    
    def _build_messages(
        self, 
        state: dict[str, Any], 
        repair: bool = False
    ) -> list[dict[str, str]]:
        """Build the chat messages for the LLM.
        
        Args:
            state: Current page state
            repair: Whether this is a repair request after invalid JSON
            
        Returns:
            List of chat messages
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Format the state as a user message
        state_text = f"""Current Page State:
- Goal: {state.get('goal', '')}
- URL: {state.get('current_url', '')}
- Title: {state.get('page_title', '')}

Visible Text (truncated):
{state.get('visible_text', '')}

Top Links:
{self._format_links(state.get('top_links', []))}

Recent Actions:
{self._format_history(state.get('recent_history', []))}

Extracted Data So Far:
{json.dumps(state.get('extracted_data', {}), indent=2)}

{state.get('additional_context', '')}

What is your next action? Respond with JSON only."""

        messages.append({"role": "user", "content": state_text})
        
        if repair:
            messages.append({"role": "assistant", "content": "I apologize for the invalid response."})
            messages.append({"role": "user", "content": self.REPAIR_PROMPT})
        
        return messages
    
    def _format_links(self, links: list[dict[str, str]]) -> str:
        """Format links for the prompt."""
        if not links:
            return "(no links found)"
        return "\n".join(
            f"- [{l.get('text', 'no text')}]({l.get('href', '#')})"
            for l in links[:15]
        )
    
    def _format_history(self, history: list[dict[str, Any]]) -> str:
        """Format action history for the prompt."""
        if not history:
            return "(no previous actions)"
        
        formatted = []
        for item in history:
            action = item.get('action', 'unknown')
            result = item.get('result', '')
            formatted.append(f"- {action}: {result}")
        return "\n".join(formatted)
    
    def chat_completion(
        self, 
        messages: list[dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Send a chat completion request to the LLM with retry logic.
        
        Args:
            messages: List of chat messages
            max_retries: Maximum retries on rate limit errors
            
        Returns:
            The assistant's response content
            
        Raises:
            httpx.HTTPError: On network errors after retries exhausted
        """
        import time
        
        url = f"{self.endpoint}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for consistency
            "max_tokens": 1000,
        }
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Add delay between requests to avoid rate limiting
                if attempt > 0:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                
                response = self.client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    if attempt < max_retries:
                        continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    continue
                raise
        
        raise last_error
    
    def get_next_action(
        self, 
        state: dict[str, Any],
        max_retries: int = 2
    ) -> ActionResponse:
        """Get the next action from the LLM.
        
        Args:
            state: Current page state
            max_retries: Maximum number of retries for invalid JSON
            
        Returns:
            Validated action response
            
        Raises:
            ValueError: If unable to get valid JSON after retries
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            repair = attempt > 0
            messages = self._build_messages(state, repair=repair)
            
            try:
                raw_response = self.chat_completion(messages)
                return self._parse_response(raw_response)
                
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_retries:
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
        # Try to extract JSON if wrapped in markdown or prose
        json_str = extract_json_from_response(raw_response)
        if json_str is None:
            json_str = raw_response.strip()
        
        # Parse JSON
        data = json.loads(json_str)
        
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
        # Add error context to the state
        state = state.copy()
        state["additional_context"] = f"""
PREVIOUS ACTION FAILED:
Action: {failed_action}
Error: {error}

Please try a different approach. Consider:
1. Using extract_visible_text to understand the current page state
2. Taking a screenshot for debugging
3. Using a different selector or action

Do NOT repeat the same failing action."""
        
        return self.get_next_action(state)


def parse_json_with_recovery(raw_response: str) -> dict[str, Any]:
    """Parse JSON with multiple recovery strategies.
    
    This is a standalone function for testing.
    
    Args:
        raw_response: Raw response that should contain JSON
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        json.JSONDecodeError: If all parsing attempts fail
    """
    # Strategy 1: Try to extract from markdown/prose
    json_str = extract_json_from_response(raw_response)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try raw response
    try:
        return json.loads(raw_response.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try to fix common issues
    cleaned = raw_response.strip()
    
    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    
    # Try again
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Find first { to last }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end+1])
        except json.JSONDecodeError:
            pass
    
    # All strategies failed
    raise json.JSONDecodeError("Could not parse JSON from response", raw_response, 0)
