"""
Domain router for Agentic Browser.

Provides intelligent routing between Browser and OS domains based on user goals.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DomainDecision:
    """Result of domain routing decision."""
    
    domain: str  # "browser", "os", or "both"
    confidence: float  # 0.0 to 1.0
    reason: str
    forced_by_user: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "domain": self.domain,
            "confidence": self.confidence,
            "reason": self.reason,
            "forced_by_user": self.forced_by_user,
        }


class DomainRouter:
    """Routes user requests to appropriate domain (Browser or OS).
    
    Uses a two-pass approach:
    1. Fast heuristic pass using keyword matching
    2. LLM router pass for ambiguous cases (if confidence < threshold)
    """
    
    # Confidence threshold for heuristic pass
    CONFIDENCE_THRESHOLD = 0.75
    
    # Browser-related keywords (weighted)
    BROWSER_KEYWORDS = {
        # High weight (0.3)
        "website": 0.3, "webpage": 0.3, "browser": 0.3, "chrome": 0.3,
        "firefox": 0.3, "url": 0.3, "http": 0.3, "https": 0.3,
        "google": 0.25, "search engine": 0.25, "internet": 0.3,
        "research": 0.35, "find out": 0.25, "look up": 0.25,
        
        # Medium weight (0.2)
        "click": 0.2, "scroll": 0.2, "navigate": 0.2, "login": 0.2,
        "sign in": 0.2, "sign up": 0.2, "form": 0.2, "button": 0.2,
        "link": 0.2, "page": 0.15, "site": 0.15, "web": 0.15,
        
        # Lower weight (0.1)
        "search": 0.15, "download page": 0.15, "online": 0.15,
        "article": 0.1, "blog": 0.1, "news": 0.1,
    }
    
    # OS-related keywords (weighted)
    OS_KEYWORDS = {
        # High weight (0.3) - Strong OS signals
        "terminal": 0.3, "command": 0.25, "shell": 0.3, "bash": 0.3,
        "file system": 0.3, "filesystem": 0.3, "directory": 0.25,
        "folder": 0.25, "chmod": 0.3, "chown": 0.3,
        "hard drive": 0.4, "harddrive": 0.4, "drive": 0.3,
        "my files": 0.35, "my computer": 0.35, "my system": 0.3,
        "contents of": 0.25, "code": 0.25, "repo": 0.25, "repository": 0.25,
        "debug": 0.2, "fix": 0.2, "analyze": 0.2,
        
        # Medium weight (0.2)
        "file": 0.15, "files": 0.15, "create file": 0.25, "delete file": 0.25,
        "read file": 0.25, "write file": 0.25, "list files": 0.25,
        "process": 0.2, "port": 0.2, "service": 0.2,
        "install": 0.2, "apt": 0.25, "dnf": 0.25, "yum": 0.25,
        "pip": 0.2, "npm": 0.2, "systemctl": 0.3,
        "disk": 0.25, "storage": 0.2,
        "git": 0.25, "pull": 0.2, "push": 0.2,
        
        # Lower weight (0.1)
        "path": 0.1, "local": 0.15, "log": 0.1, "config": 0.1, "system": 0.1,
        "home": 0.15, "desktop": 0.15, "documents": 0.15,
    }
    
    # URL pattern for browser detection
    URL_PATTERN = re.compile(
        r'https?://[^\s]+|www\.[^\s]+|\b\w+\.(com|org|net|io|dev|edu|gov)\b',
        re.IGNORECASE
    )
    
    # File path pattern for OS detection
    PATH_PATTERN = re.compile(
        r'(?:^|[\s"])([~./][\w./-]+|/[\w./-]+)',
        re.MULTILINE
    )
    
    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize the domain router.
        
        Args:
            llm_client: Optional LLM client for ambiguous routing
        """
        self.llm_client = llm_client
    
    def route(
        self, 
        user_goal: str, 
        mode: str = "auto",
    ) -> DomainDecision:
        """Route a user request to the appropriate domain.
        
        Args:
            user_goal: The user's goal/request
            mode: Routing mode - "auto", "browser", "os", or "ask"
            
        Returns:
            DomainDecision with domain, confidence, and reason
        """
        # Handle manual override modes
        if mode == "browser":
            return DomainDecision(
                domain="browser",
                confidence=1.0,
                reason="User selected browser-only mode",
                forced_by_user=True,
            )
        elif mode == "os":
            return DomainDecision(
                domain="os",
                confidence=1.0,
                reason="User selected OS-only mode",
                forced_by_user=True,
            )
        elif mode == "ask":
            # Caller should prompt user
            return DomainDecision(
                domain="unknown",
                confidence=0.0,
                reason="User preference: ask every time",
                forced_by_user=True,
            )
        
        # Auto mode: try heuristic first
        decision = self._heuristic_route(user_goal)
        
        # If confidence is high enough, use heuristic result
        if decision.confidence >= self.CONFIDENCE_THRESHOLD:
            return decision
        
        # Low confidence: try LLM router if available
        if self.llm_client is not None:
            llm_decision = self._llm_route(user_goal)
            if llm_decision is not None:
                return llm_decision
        
        # Fall back to heuristic result even with low confidence
        return decision
    
    def _heuristic_route(self, goal: str) -> DomainDecision:
        """Fast keyword-based routing.
        
        Args:
            goal: User's goal text
            
        Returns:
            DomainDecision based on keyword analysis
        """
        goal_lower = goal.lower()
        
        # Calculate scores
        browser_score = 0.0
        os_score = 0.0
        browser_matches = []
        os_matches = []
        
        # Check URL patterns (strong browser signal)
        if self.URL_PATTERN.search(goal):
            browser_score += 0.5
            browser_matches.append("URL detected")
        
        # Check file path patterns (strong OS signal)
        if self.PATH_PATTERN.search(goal):
            os_score += 0.3
            os_matches.append("file path detected")
        
        # Check browser keywords
        for keyword, weight in self.BROWSER_KEYWORDS.items():
            if keyword in goal_lower:
                browser_score += weight
                browser_matches.append(keyword)
        
        # Check OS keywords
        for keyword, weight in self.OS_KEYWORDS.items():
            if keyword in goal_lower:
                os_score += weight
                os_matches.append(keyword)
        
        # Normalize scores (cap at 1.0)
        browser_score = min(browser_score, 1.0)
        os_score = min(os_score, 1.0)
        
        # Determine domain and confidence
        if browser_score > os_score:
            confidence = browser_score - os_score + 0.5
            confidence = min(confidence, 1.0)
            
            reason = f"Browser signals: {', '.join(browser_matches[:3])}"
            if os_matches:
                reason += f" (also OS: {', '.join(os_matches[:2])})"
            
            return DomainDecision(
                domain="browser",
                confidence=confidence,
                reason=reason,
            )
        elif os_score > browser_score:
            confidence = os_score - browser_score + 0.5
            confidence = min(confidence, 1.0)
            
            reason = f"OS signals: {', '.join(os_matches[:3])}"
            if browser_matches:
                reason += f" (also browser: {', '.join(browser_matches[:2])})"
            
            return DomainDecision(
                domain="os",
                confidence=confidence,
                reason=reason,
            )
        elif browser_score > 0 and os_score > 0:
            # Both domains detected equally
            return DomainDecision(
                domain="both",
                confidence=0.5,
                reason=f"Mixed signals - browser: {', '.join(browser_matches[:2])}; OS: {', '.join(os_matches[:2])}",
            )
        else:
            # No clear signals
            return DomainDecision(
                domain="browser",  # Default to browser for ambiguous web tasks
                confidence=0.3,
                reason="No clear domain signals, defaulting to browser",
            )
    
    def _llm_route(self, goal: str) -> Optional[DomainDecision]:
        """LLM-based routing for ambiguous cases.
        
        Args:
            goal: User's goal text
            
        Returns:
            DomainDecision from LLM, or None if call fails
        """
        if self.llm_client is None:
            return None
        
        router_prompt = """You are a request classifier. Determine if this user request should be handled by:
- "browser": Web browsing, searching online, interacting with websites
- "os": Local file operations, terminal commands, system administration
- "both": Requires both web and local system operations

User request: {goal}

Respond with ONLY a JSON object:
{{"domain": "browser|os|both", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            # Build minimal messages for routing (not using full agent prompt)
            messages = [
                {"role": "user", "content": router_prompt.format(goal=goal)},
            ]
            
            response = self.llm_client.chat_completion(messages, max_retries=1)
            
            # Parse JSON response
            import json
            from .utils import extract_json_from_response
            
            json_str = extract_json_from_response(response)
            if json_str is None:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            domain = data.get("domain", "browser")
            if domain not in ("browser", "os", "both"):
                domain = "browser"
            
            confidence = float(data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
            
            reason = data.get("reason", "LLM classification")
            
            return DomainDecision(
                domain=domain,
                confidence=confidence,
                reason=f"[LLM] {reason}",
            )
            
        except Exception as e:
            # LLM routing failed, return None to fall back to heuristic
            return None
    
    @staticmethod
    def is_browser_action(action: str) -> bool:
        """Check if an action is a browser action.
        
        Args:
            action: Action name
            
        Returns:
            True if browser action
        """
        browser_actions = {
            "goto", "click", "type", "press", "scroll", 
            "wait_for", "extract", "extract_visible_text",
            "screenshot", "back", "forward", "done",
            "download_file", "download_image",
        }
        return action in browser_actions
    
    @staticmethod
    def is_os_action(action: str) -> bool:
        """Check if an action is an OS action.
        
        Args:
            action: Action name
            
        Returns:
            True if OS action
        """
        os_actions = {
            "os_exec",
            "os_list_dir",
            "os_read_file",
            "os_write_file",
            "os_move_file",
            "os_copy_file",
            "os_delete_file",
        }
        return action in os_actions

    @staticmethod
    def is_memory_action(action: str) -> bool:
        """Check if an action is a memory action.

        Args:
            action: Action name

        Returns:
            True if memory action
        """
        memory_actions = {
            "memory_get_site",
            "memory_save_site",
            "memory_get_directory",
        }
        return action in memory_actions
