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
        pattern=r"^(goto|click|type|press|scroll|wait_for|extract|extract_visible_text|screenshot|back|forward|done|os_exec|os_list_dir|os_read_file|os_write_file)$"
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
  "action": "goto|click|type|press|scroll|wait_for|extract|extract_visible_text|screenshot|back|forward|done|os_exec|os_list_dir|os_read_file|os_write_file",
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

OS Action argument rules:
- os_exec: { "cmd": "command string", "timeout_s": 30, "cwd": "/optional/path" }
- os_list_dir: { "path": "/path/to/dir" }
- os_read_file: { "path": "/path/to/file", "max_bytes": 10000 }
- os_write_file: { "path": "/path/to/file", "content": "...", "mode": "overwrite|append" }

CRITICAL WORKFLOW PATTERNS:

🔍 SEARCH ENGINE: ALWAYS USE DUCKDUCKGO (https://duckduckgo.com)
- Google BLOCKS AI agents with CAPTCHAs - NEVER use Google!
- DuckDuckGo is AI-friendly and won't block you
- For ANY search/research task, start with: goto("https://duckduckgo.com")
- Then type your query in the search box and press Enter

⚠️ URL VALIDATION - AVOID 404 ERRORS:
- Only use URLs that start with https:// or http://
- NEVER use partial URLs like "/about" or "example.com/page"
- If a URL looks malformed or truncated, DON'T use it
- Prefer links that look like real websites, not tracking/redirect URLs
- If you get a 404, DON'T try variations - move to next site

1. SEARCH: After typing in a search box, you MUST press Enter to submit
2. FORMS: After filling form fields, click the submit button  
3. NEVER repeat the same action - always progress to the next step

4. RESEARCH TASKS: When asked to "research", "compare", "find information":
   - Start with DuckDuckGo search
   - Visit 2-3 different sources from results
   - Only use COMPLETE URLs starting with https://
   - Skip any links that look broken or truncated
   - Extract text from each site, then call "done" with summary
   - Do NOT summarize search snippets - visit the actual sites!

5. CLICKING LINKS: If a click selector fails, use goto with a URL from Top Links instead

Selector tips:
- Use text= selectors for visible text: text="Click Me"
- Use CSS selectors for elements with IDs/classes
- If clicks keep failing, use goto with a URL from Top Links instead

Risk classification:
- HIGH: purchases, payments, sending messages, account settings, deleting
- MEDIUM: logging in, uploading files, granting permissions
- LOW: navigation, reading, extracting data

OS Risk classification:
- HIGH: rm, dd, mkfs, sudo, chmod -R, chown -R, writing to /etc /usr /bin
- MEDIUM: writing files, running scripts, process management
- LOW: reading files, listing directories, inspecting system info

Set requires_approval=true for HIGH risk and MEDIUM risk actions."""

    OS_SYSTEM_PROMPT = """You are an autonomous Linux system agent that helps users accomplish tasks on their local machine.

You receive the current system state and must respond with a SINGLE action to take next.

CRITICAL: You MUST respond with ONLY valid JSON, no markdown, no prose, no explanation outside the JSON.

⚠️ STEP LIMIT: You have a MAXIMUM of 8 actions. Use 3-5 for exploration, then call "done" with findings!

Your response must follow this exact schema:
{
  "action": "os_exec|os_list_dir|os_read_file|os_write_file|done",
  "args": { ... },
  "rationale": "short reason for this action",
  "risk": "low|medium|high",
  "requires_approval": true/false,
  "final_answer": "only when action=done - put your complete answer/summary here"
}

Action argument rules:
- os_exec: { "cmd": "command string", "timeout_s": 30, "cwd": "/optional/path" }
- os_list_dir: { "path": "/path/to/dir" }
- os_read_file: { "path": "/path/to/file", "max_bytes": 10000 }
- os_write_file: { "path": "/path/to/file", "content": "...", "mode": "overwrite|append" }
- done: { "summary_style": "bullets|paragraph" }

CRITICAL WORKFLOW - STICK TO THIS PATTERN:
1. Run 1-2 commands to gather the information needed
2. If a command fails, try ONE alternative approach  
3. IMMEDIATELY call "done" with whatever information you gathered
4. DO NOT keep trying different commands - summarize what you found!

EXAMPLE (3 steps max):
User: "Look at my hard drive"
Step 1: os_exec(cmd="df -h") → get disk usage
Step 2: os_list_dir(path="~") → see home directory
Step 3: done(final_answer="Your disk has 50GB used, 100GB free. Home contains: Documents, Downloads, projects...")

WHEN TO CALL "done" IMMEDIATELY:
- You have gathered useful information (even partial)
- A command failed - summarize what you DO know, don't keep retrying
- You've run 3+ commands - STOP and summarize NOW
- The user's question can be answered from data you already have

HANDLE FAILURES GRACEFULLY:
- If a command fails with permission denied, say "I couldn't access X due to permissions"
- Don't retry the same failing command with slight variations
- Summarize what you COULD access

⚠️ CASE SENSITIVITY: Linux filesystems are case-sensitive!
- If user says "coding" but you see "Coding", they mean the same thing!
- Always look for case variations: coding = Coding = CODING
- Match directories/files by ignoring case when the user's intent is clear
- Don't say "directory doesn't exist" if a case variant exists!

🔍 FUZZY/CONTEXTUAL MATCHING - THIS IS CRITICAL:
Users describe things by meaning, not exact names. YOU MUST:

1. INTERPRET USER INTENT:
   - "cat app" could be: CatOS, cat-tracker, meow-app, feline-project, kitty-*, etc.
   - "coding directory" is likely: Coding, code, projects, dev, src, repos, etc.
   - "my notes" might be: Notes, notes.txt, journal, diary, etc.

2. EXPLORE PROMISING DIRECTORIES:
   - If user says "find X in Y directory", ACTUALLY GO INTO Y!
   - Don't just list home and give up - explore the obvious candidates
   - Example: "cat app in coding dir" → list ~/Coding, look for cat-related names

3. MATCH BY MEANING, NOT EXACT STRING:
   - A project about cats might not have "cat" in the name
   - Look for README files, package.json, main.py to understand what a project is
   - If unsure, explore first THEN summarize what you found

4. DON'T GIVE UP EASILY:
   - If you see a likely match, explore it before saying "not found"
   - ~/Coding exists? Look inside it before saying no coding directory!

Risk classification:
- HIGH: rm, dd, mkfs, sudo, chmod -R, chown -R, writing to /etc /usr /bin
- MEDIUM: writing files, running scripts, modifying configs
- LOW: reading files, listing directories, inspecting system info (df, ls, cat, grep)

Set requires_approval=true for HIGH and MEDIUM risk actions."""

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

    def __init__(self, config: AgentConfig, provider_config: "ProviderConfig | None" = None):
        """Initialize the LLM client.
        
        Args:
            config: Agent configuration
            provider_config: Optional provider configuration for native adapters
        """
        self.config = config
        self.endpoint = config.model_endpoint.rstrip("/")
        self.model = config.model
        self.api_key = config.api_key
        
        # If provider config supplied, use native adapter
        self._adapter = None
        if provider_config:
            try:
                from .adapters import create_adapter, Message
                from .providers import Provider
                self._adapter = create_adapter(provider_config.provider, provider_config)
                self._Message = Message
            except ImportError:
                pass  # Fall back to direct HTTP
        
        # Build headers (for fallback HTTP mode)
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # HTTP client (for fallback mode)
        self.client = httpx.Client(timeout=60.0)
    
    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
        if self._adapter:
            self._adapter.close()
    
    def _build_messages(
        self, 
        state: dict[str, Any], 
        repair: bool = False,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build the chat messages for the LLM.
        
        Args:
            state: Current page state
            repair: Whether this is a repair request after invalid JSON
            system_prompt: Optional override for system prompt (e.g., OS mode)
            
        Returns:
            List of chat messages
        """
        prompt = system_prompt if system_prompt else self.SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": prompt}
        ]
        
        # Format visited URLs for context
        visited_urls = state.get('visited_urls', [])
        visited_str = "\n".join(f"- {url}" for url in visited_urls[-10:]) if visited_urls else "(none yet)"
        
        # Format the state as a user message
        state_text = f"""Current Page State:
- Goal: {state.get('goal', '')}
- URL: {state.get('current_url', '')}
- Title: {state.get('page_title', '')}

Visible Text (truncated):
{state.get('visible_text', '')}

Top Links:
{self._format_links(state.get('top_links', []))}

⚠️ ALREADY VISITED URLs (do NOT click these again):
{visited_str}

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
        
        # Use adapter if available (for native Anthropic/Google support)
        if self._adapter:
            adapter_messages = [
                self._Message(m["role"], m["content"]) for m in messages
            ]
            return self._adapter.chat_completion(adapter_messages, max_retries)
        
        # Fallback: Direct OpenAI-compatible HTTP requests
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
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ) -> ActionResponse:
        """Get the next action from the LLM.
        
        Args:
            state: Current page state
            max_retries: Maximum number of retries for invalid JSON
            system_prompt: Optional override for system prompt (e.g., OS mode)
            
        Returns:
            Validated action response
            
        Raises:
            ValueError: If unable to get valid JSON after retries
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            repair = attempt > 0
            messages = self._build_messages(state, repair=repair, system_prompt=system_prompt)
            
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
