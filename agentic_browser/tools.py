"""
Browser tools for Agentic Browser.

Provides browser action execution via Playwright.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

from .utils import format_selector, truncate_text


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    message: str
    data: Optional[Any] = None
    screenshot_path: Optional[Path] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "success": self.success,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.screenshot_path is not None:
            result["screenshot_path"] = str(self.screenshot_path)
        return result


class BrowserTools:
    """Executes browser actions via Playwright."""
    
    def __init__(
        self, 
        page: Page, 
        default_timeout: int = 10000,
        screenshots_dir: Optional[Path] = None,
    ):
        """Initialize browser tools.
        
        Args:
            page: Playwright page instance
            default_timeout: Default timeout in milliseconds
            screenshots_dir: Directory for saving screenshots
        """
        self.page = page
        self.default_timeout = default_timeout
        self.screenshots_dir = screenshots_dir
        self._step_count = 0
    
    def set_step(self, step: int) -> None:
        """Set the current step number for screenshot naming."""
        self._step_count = step
    
    def execute(self, action: str, args: dict[str, Any]) -> ToolResult:
        """Execute a browser action.
        
        Args:
            action: Action name
            args: Action arguments
            
        Returns:
            ToolResult with success status and message
        """
        # Dispatch to the appropriate method
        method_map = {
            "goto": self.goto,
            "click": self.click,
            "type": self.type_text,
            "press": self.press,
            "scroll": self.scroll,
            "wait_for": self.wait_for,
            "extract": self.extract,
            "extract_visible_text": self.extract_visible_text,
            "screenshot": self.screenshot,
            "back": self.back,
            "forward": self.forward,
            "done": self.done,
        }
        
        method = method_map.get(action)
        if method is None:
            return ToolResult(
                success=False,
                message=f"Unknown action: {action}",
            )
        
        try:
            return method(**args)
        except PlaywrightTimeoutError as e:
            return ToolResult(
                success=False,
                message=f"Timeout: {str(e)}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error: {type(e).__name__}: {str(e)}",
            )
    
    def goto(self, url: str) -> ToolResult:
        """Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            ToolResult
        """
        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        self.page.goto(url, wait_until="domcontentloaded")
        
        return ToolResult(
            success=True,
            message=f"Navigated to {url}",
            data={"url": self.page.url, "title": self.page.title()},
        )
    
    def click(
        self, 
        selector: str, 
        timeout_ms: int = 10000
    ) -> ToolResult:
        """Click an element with fallback strategies.
        
        Args:
            selector: Element selector
            timeout_ms: Timeout in milliseconds
            
        Returns:
            ToolResult
        """
        original_selector = selector
        selector = format_selector(selector)
        
        # Build list of selectors to try
        selectors_to_try = [selector]
        
        # If it's a text= selector, add partial match fallbacks
        if selector.startswith('text="') and selector.endswith('"'):
            full_text = selector[6:-1]  # Extract text between quotes
            
            # Try shorter versions of the text (first 50, 30, 20 chars)
            if len(full_text) > 50:
                selectors_to_try.append(f'text="{full_text[:50]}"')
            if len(full_text) > 30:
                selectors_to_try.append(f'text="{full_text[:30]}"')
            if len(full_text) > 20:
                selectors_to_try.append(f'text="{full_text[:20]}"')
            
            # Also try as a link with partial text
            selectors_to_try.append(f'a:has-text("{full_text[:40]}")')
            selectors_to_try.append(f'a:has-text("{full_text[:25]}")')
            
            # Try h1, h2, h3 with text (common for article titles)
            selectors_to_try.append(f':is(h1,h2,h3):has-text("{full_text[:30]}")')
        
        last_error = None
        for try_selector in selectors_to_try:
            try:
                # Check if element exists first
                locator = self.page.locator(try_selector)
                if locator.count() > 0:
                    locator.first.click(timeout=timeout_ms // len(selectors_to_try) or 3000)
                    
                    # Wait briefly for any navigation or updates
                    try:
                        self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except:
                        pass
                    
                    return ToolResult(
                        success=True,
                        message=f"Clicked: {try_selector}",
                        data={"url": self.page.url, "selector_used": try_selector},
                    )
            except Exception as e:
                last_error = e
                continue
        
        # If all selectors failed, return the error
        return ToolResult(
            success=False,
            message=f"Could not click element. Tried selectors: {selectors_to_try[:3]}... Error: {last_error}",
            data={"original_selector": original_selector},
        )
    
    def type_text(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
    ) -> ToolResult:
        """Type text into an element.
        
        Args:
            selector: Element selector
            text: Text to type
            clear_first: Whether to clear the field first
            
        Returns:
            ToolResult
        """
        original_selector = selector
        selector = format_selector(selector)
        
        # Define fallback selectors for common search boxes
        fallback_selectors = []
        
        # If it looks like a Google search selector, add fallbacks
        if "input" in selector.lower() or "search" in selector.lower() or "name='q'" in selector:
            fallback_selectors = [
                'textarea[name="q"]',  # Google's new textarea-based search
                'input[name="q"]',
                'input[aria-label="Search"]',
                'textarea[aria-label="Search"]',
                '[role="combobox"]',
                'input[type="search"]',
                'input[type="text"]',
            ]
        
        # Try the primary selector first
        selectors_to_try = [selector] + [s for s in fallback_selectors if s != selector]
        
        last_error = None
        for sel in selectors_to_try:
            try:
                # First try to click the element to focus it
                try:
                    self.page.click(sel, timeout=3000)
                    self.page.wait_for_timeout(200)
                except:
                    pass  # Click failed, still try to type
                
                if clear_first:
                    self.page.fill(sel, text, timeout=5000)
                else:
                    self.page.type(sel, text, timeout=5000)
                
                return ToolResult(
                    success=True,
                    message=f"Typed into: {sel}" + (f" (fallback from {original_selector})" if sel != selector else ""),
                    data={"chars_typed": len(text), "selector_used": sel},
                )
            except Exception as e:
                last_error = e
                continue
        
        # All selectors failed
        return ToolResult(
            success=False,
            message=f"Could not type into any selector. Last error: {last_error}. Tried: {selectors_to_try[:3]}",
        )
    
    def press(self, key: str) -> ToolResult:
        """Press a keyboard key.
        
        Args:
            key: Key to press (e.g., "Enter", "Tab", "ArrowDown")
            
        Returns:
            ToolResult
        """
        # Keys that might trigger navigation
        navigation_keys = ["Enter", "Return"]
        
        if key in navigation_keys:
            # Use expect_navigation pattern for Enter key
            try:
                # Press and wait for potential navigation
                self.page.keyboard.press(key)
                
                # Wait for any navigation to complete
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                except:
                    pass  # Page might not navigate, that's OK
                
                # Extra wait for dynamic content
                self.page.wait_for_timeout(1000)
                
            except Exception as e:
                # Even if there's an error, the key was pressed
                return ToolResult(
                    success=True,
                    message=f"Pressed: {key} (page may have navigated)",
                )
        else:
            self.page.keyboard.press(key)
            self.page.wait_for_timeout(300)
        
        return ToolResult(
            success=True,
            message=f"Pressed: {key}",
        )
    
    def scroll(self, amount: int = 800) -> ToolResult:
        """Scroll the page.
        
        Args:
            amount: Scroll amount (positive = down, negative = up)
            
        Returns:
            ToolResult
        """
        self.page.evaluate(f"window.scrollBy(0, {amount})")
        
        direction = "down" if amount > 0 else "up"
        return ToolResult(
            success=True,
            message=f"Scrolled {direction} by {abs(amount)}px",
        )
    
    def wait_for(
        self,
        selector: Optional[str] = None,
        timeout_ms: int = 10000,
    ) -> ToolResult:
        """Wait for an element or a timeout.
        
        Args:
            selector: Element selector to wait for (optional)
            timeout_ms: Timeout in milliseconds
            
        Returns:
            ToolResult
        """
        if selector:
            selector = format_selector(selector)
            self.page.wait_for_selector(selector, timeout=timeout_ms)
            return ToolResult(
                success=True,
                message=f"Found element: {selector}",
            )
        else:
            self.page.wait_for_timeout(timeout_ms)
            return ToolResult(
                success=True,
                message=f"Waited {timeout_ms}ms",
            )
    
    def extract(
        self,
        selector: str,
        attribute: str = "innerText",
    ) -> ToolResult:
        """Extract data from an element.
        
        Args:
            selector: Element selector
            attribute: Attribute to extract (innerText, href, value, etc.)
            
        Returns:
            ToolResult with extracted data
        """
        selector = format_selector(selector)
        
        element = self.page.query_selector(selector)
        if not element:
            return ToolResult(
                success=False,
                message=f"Element not found: {selector}",
            )
        
        if attribute == "innerText":
            value = element.inner_text()
        elif attribute == "innerHTML":
            value = element.inner_html()
        elif attribute == "value":
            value = element.get_attribute("value") or ""
        else:
            value = element.get_attribute(attribute)
        
        return ToolResult(
            success=True,
            message=f"Extracted {attribute} from {selector}",
            data={"value": value},
        )
    
    def extract_visible_text(self, max_chars: int = 8000) -> ToolResult:
        """Extract all visible text from the page.
        
        Includes retry logic for transient timing errors during navigation.
        
        Args:
            max_chars: Maximum characters to return
            
        Returns:
            ToolResult with visible text
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Wait for page to be ready
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=3000)
                except:
                    pass
                
                # Get text content, excluding scripts and styles
                text = self.page.evaluate("""
                    () => {
                        // Safety check for document.body
                        if (!document.body) {
                            return document.title || 'Page is loading...';
                        }
                        
                        try {
                            const walker = document.createTreeWalker(
                                document.body,
                                NodeFilter.SHOW_TEXT,
                                {
                                    acceptNode: (node) => {
                                        const parent = node.parentElement;
                                        if (!parent) return NodeFilter.FILTER_REJECT;
                                        const tag = parent.tagName.toLowerCase();
                                        if (['script', 'style', 'noscript'].includes(tag)) {
                                            return NodeFilter.FILTER_REJECT;
                                        }
                                        try {
                                            const style = window.getComputedStyle(parent);
                                            if (style.display === 'none' || style.visibility === 'hidden') {
                                                return NodeFilter.FILTER_REJECT;
                                            }
                                        } catch (e) {
                                            return NodeFilter.FILTER_ACCEPT;
                                        }
                                        return NodeFilter.FILTER_ACCEPT;
                                    }
                                }
                            );
                            
                            const texts = [];
                            while (walker.nextNode()) {
                                const text = walker.currentNode.textContent.trim();
                                if (text) texts.push(text);
                            }
                            return texts.join(' ') || document.body.innerText || '';
                        } catch (e) {
                            // Fallback to simple innerText
                            return document.body.innerText || document.title || 'Unable to extract text';
                        }
                    }
                """)
                
                # Clean and truncate
                text = " ".join(text.split())
                text = truncate_text(text, max_chars)
                
                return ToolResult(
                    success=True,
                    message=f"Extracted {len(text)} characters of visible text",
                    data={"text": text},
                )
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 500ms, 1000ms, 1500ms
                    self.page.wait_for_timeout(500 * (attempt + 1))
                    continue
        
        # Return safe empty result instead of crashing
        return ToolResult(
            success=True,
            message="Failed to extract text (timing issue), returning empty",
            data={"text": "", "warning": str(last_error) if last_error else "Unknown error"},
        )
    
    def screenshot(self, label: Optional[str] = None) -> ToolResult:
        """Take a screenshot of the page.
        
        Args:
            label: Optional label for the screenshot
            
        Returns:
            ToolResult with screenshot path
        """
        screenshot_bytes = self.page.screenshot(full_page=False)
        
        if self.screenshots_dir:
            label_part = f"_{label}" if label else ""
            filename = f"step_{self._step_count:03d}{label_part}.png"
            path = self.screenshots_dir / filename
            path.write_bytes(screenshot_bytes)
            
            return ToolResult(
                success=True,
                message=f"Screenshot saved: {filename}",
                screenshot_path=path,
            )
        
        return ToolResult(
            success=True,
            message="Screenshot captured (not saved - no directory configured)",
            data={"size_bytes": len(screenshot_bytes)},
        )
    
    def back(self) -> ToolResult:
        """Navigate back in history.
        
        Returns:
            ToolResult
        """
        self.page.go_back(wait_until="domcontentloaded")
        
        return ToolResult(
            success=True,
            message="Navigated back",
            data={"url": self.page.url},
        )
    
    def forward(self) -> ToolResult:
        """Navigate forward in history.
        
        Returns:
            ToolResult
        """
        self.page.go_forward(wait_until="domcontentloaded")
        
        return ToolResult(
            success=True,
            message="Navigated forward",
            data={"url": self.page.url},
        )
    
    def done(self, summary_style: str = "paragraph") -> ToolResult:
        """Mark the task as done.
        
        Args:
            summary_style: Style for the summary (bullets or paragraph)
            
        Returns:
            ToolResult
        """
        return ToolResult(
            success=True,
            message="Task completed",
            data={"summary_style": summary_style},
        )
    
    # Helper methods for state extraction
    def get_page_state(self, max_text_chars: int = 8000, max_links: int = 15) -> dict[str, Any]:
        """Get the current page state.
        
        Args:
            max_text_chars: Maximum characters for visible text
            max_links: Maximum number of links to include
            
        Returns:
            Dictionary with page state
        """
        # Get visible text
        visible_text = self.extract_visible_text(max_text_chars).data.get("text", "")
        
        # Get top links
        links = self._get_top_links(max_links)
        
        return {
            "current_url": self.page.url,
            "page_title": self.page.title(),
            "visible_text": visible_text,
            "top_links": links,
        }
    
    def _get_top_links(self, max_links: int) -> list[dict[str, str]]:
        """Get the top visible links on the page.
        
        Args:
            max_links: Maximum number of links
            
        Returns:
            List of link dictionaries with text and href
        """
        try:
            links = self.page.evaluate(f"""
                () => {{
                    const links = [];
                    const anchors = document.querySelectorAll('a[href]');
                    
                    for (const a of anchors) {{
                        if (links.length >= {max_links}) break;
                        
                        const rect = a.getBoundingClientRect();
                        const style = window.getComputedStyle(a);
                        
                        // Skip hidden or off-screen links
                        if (style.display === 'none' || 
                            style.visibility === 'hidden' ||
                            rect.width === 0 || 
                            rect.height === 0) {{
                            continue;
                        }}
                        
                        const text = a.innerText.trim().substring(0, 100);
                        const href = a.href;
                        
                        if (text && href) {{
                            links.push({{ text, href }});
                        }}
                    }}
                    
                    return links;
                }}
            """)
            return links
        except Exception:
            return []
