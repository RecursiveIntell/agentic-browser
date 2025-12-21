"""
Type definitions for Agentic Browser.

Provides typed dataclasses for common structures used across agents and tools.
These types improve IDE autocompletion, enable static type checking, and
serve as documentation for the data shapes used throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionData:
    """Parsed action from LLM response.
    
    Represents the structured output from an agent after parsing
    the LLM's JSON response.
    
    Attributes:
        action: The action name (e.g., "goto", "click", "done")
        args: Arguments for the action
        rationale: Optional explanation for why this action was chosen
    """
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action": self.action,
            "args": self.args,
            "rationale": self.rationale,
        }


@dataclass
class TokenUsage:
    """Token usage tracking for LLM cost calculations.
    
    Accumulates token counts and costs across agent invocations.
    
    Attributes:
        input_tokens: Total input tokens used
        output_tokens: Total output tokens generated
        total_tokens: Sum of input and output tokens
        total_cost: Calculated cost in USD
    """
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    total_tokens: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for state storage."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "TokenUsage":
        """Create from dictionary (e.g., from state)."""
        return cls(
            input_tokens=data.get("input_tokens", 0.0),
            output_tokens=data.get("output_tokens", 0.0),
            total_tokens=data.get("total_tokens", 0.0),
            total_cost=data.get("total_cost", 0.0),
        )

    def add(self, input_tokens: int, output_tokens: int, cost: float) -> "TokenUsage":
        """Add usage and return updated instance (immutable pattern)."""
        return TokenUsage(
            input_tokens=self.input_tokens + input_tokens,
            output_tokens=self.output_tokens + output_tokens,
            total_tokens=self.total_tokens + input_tokens + output_tokens,
            total_cost=self.total_cost + cost,
        )


@dataclass
class PageState:
    """Browser page state snapshot.
    
    Represents the current state of a browser page as seen by agents.
    
    Attributes:
        url: Current page URL
        title: Page title
        visible_text: Truncated visible text content
        links: List of clickable link dicts with text and href
    """
    url: str = ""
    title: str = ""
    visible_text: str = ""
    links: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_browser_tools(cls, page_state: dict[str, Any]) -> "PageState":
        """Create from BrowserTools.get_page_state() result."""
        return cls(
            url=page_state.get("current_url", "") or page_state.get("url", ""),
            title=page_state.get("page_title", "") or page_state.get("title", ""),
            visible_text=page_state.get("visible_text", ""),
            links=page_state.get("top_links", []) or page_state.get("links", []),
        )
