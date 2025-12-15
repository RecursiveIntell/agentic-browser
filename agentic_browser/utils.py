"""
Utility functions for Agentic Browser.

Provides helpers for text processing, selectors, and general utilities.
"""

import re
from typing import Any, Optional


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to a maximum number of characters.
    
    Args:
        text: Text to truncate
        max_chars: Maximum number of characters
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def extract_json_from_response(response: str) -> Optional[str]:
    """Extract JSON from a response that might contain markdown or extra text.
    
    Args:
        response: Raw response string
        
    Returns:
        Extracted JSON string, or None if not found
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(code_block_pattern, response)
    if match:
        return match.group(1)
    
    # Try to find a raw JSON object
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, response)
    if match:
        return match.group(0)
    
    return None


def format_selector(selector: str) -> str:
    """Normalize a selector for Playwright.
    
    Args:
        selector: Raw selector string
        
    Returns:
        Normalized selector
    """
    selector = selector.strip()
    
    # Already a Playwright selector format
    if selector.startswith(('text=', 'css=', 'xpath=', 'id=', '//')):
        return selector
    
    # Looks like a text selector (contains spaces and no CSS-like chars)
    if ' ' in selector and not any(c in selector for c in '.#[]:>+~'):
        # Wrap in text= selector
        return f'text="{selector}"'
    
    # Assume CSS selector
    return selector


def is_password_field(selector: str) -> bool:
    """Check if a selector likely refers to a password field.
    
    Args:
        selector: The selector to check
        
    Returns:
        True if likely a password field
    """
    password_patterns = [
        r'password',
        r'type=["\']?password',
        r'#pass',
        r'\.pass',
        r'passwd',
        r'pwd',
    ]
    selector_lower = selector.lower()
    return any(re.search(p, selector_lower) for p in password_patterns)


def parse_domain(url: str) -> str:
    """Extract the domain from a URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name (e.g., "example.com")
    """
    # Remove protocol
    domain = re.sub(r'^https?://', '', url)
    # Remove path
    domain = domain.split('/')[0]
    # Remove port
    domain = domain.split(':')[0]
    return domain.lower()


def format_action_for_history(
    action: str, 
    args: dict[str, Any], 
    result: str
) -> dict[str, Any]:
    """Format an action for the history summary.
    
    Args:
        action: Action name
        args: Action arguments
        result: Result of the action
        
    Returns:
        Formatted history entry
    """
    # Truncate long values
    formatted_args = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 100:
            formatted_args[key] = truncate_text(value, 100)
        else:
            formatted_args[key] = value
    
    return {
        "action": action,
        "args": formatted_args,
        "result": truncate_text(result, 200),
    }


def sanitize_filename(text: str) -> str:
    """Convert text to a safe filename.
    
    Args:
        text: Text to convert
        
    Returns:
        Safe filename string
    """
    # Remove or replace invalid characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Replace spaces
    safe = safe.replace(' ', '_')
    # Remove consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Trim and limit length
    safe = safe.strip('_')[:50]
    return safe or "unnamed"


PAYMENT_DOMAINS = {
    "paypal.com",
    "stripe.com", 
    "checkout.stripe.com",
    "pay.google.com",
    "apple.com/shop",
    "amazon.com/gp/buy",
    "checkout.shopify.com",
    "secure.checkout",
    "payment.",
    "pay.",
    "checkout.",
}


def is_payment_domain(url: str) -> bool:
    """Check if a URL is likely a payment domain.
    
    Args:
        url: URL to check
        
    Returns:
        True if likely a payment domain
    """
    domain = parse_domain(url)
    url_lower = url.lower()
    
    # Check exact domain matches
    for payment_domain in PAYMENT_DOMAINS:
        if payment_domain in domain or payment_domain in url_lower:
            return True
    
    # Check for payment-related paths
    payment_paths = ['/checkout', '/payment', '/pay/', '/cart/checkout', '/order/']
    return any(p in url_lower for p in payment_paths)


HIGH_RISK_KEYWORDS = {
    "buy", "purchase", "checkout", "pay", "order", "subscribe",
    "delete", "remove", "cancel subscription", "close account",
    "send", "post", "submit", "confirm payment", "place order",
    "unsubscribe", "terminate", "deactivate",
}


def contains_high_risk_keywords(text: str) -> bool:
    """Check if text contains high-risk keywords.
    
    Args:
        text: Text to check
        
    Returns:
        True if high-risk keywords found
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in HIGH_RISK_KEYWORDS)


MEDIUM_RISK_KEYWORDS = {
    "login", "sign in", "log in", "password",
    "upload", "attach", "file", "permission",
    "allow", "grant access", "authorize",
}


def contains_medium_risk_keywords(text: str) -> bool:
    """Check if text contains medium-risk keywords.
    
    Args:
        text: Text to check
        
    Returns:
        True if medium-risk keywords found
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in MEDIUM_RISK_KEYWORDS)
