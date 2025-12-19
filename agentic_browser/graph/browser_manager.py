"""
Lazy browser manager for on-demand browser initialization.

The browser is only launched when an agent actually needs it.
"""

import atexit
import logging
import re
from typing import Optional, TYPE_CHECKING

from playwright.sync_api import sync_playwright, Playwright, Browser, BrowserContext, Page, Route

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..tools import BrowserTools


# Get logger for this module
logger = logging.getLogger(__name__)


# Track all browser managers for cleanup on exit
_active_managers: list["LazyBrowserManager"] = []


# Resource patterns to block in fast mode
FAST_MODE_BLOCKED_PATTERNS = [
    # Images
    r".*\.(png|jpg|jpeg|webp|gif|svg|ico|bmp|tiff)(\?.*)?$",
    # Fonts  
    r".*\.(woff|woff2|ttf|otf|eot)(\?.*)?$",
    # Media
    r".*\.(mp4|webm|mp3|wav|ogg|avi|mov|flv)(\?.*)?$",
]

# Compile patterns for performance
_blocked_patterns_compiled = [re.compile(p, re.IGNORECASE) for p in FAST_MODE_BLOCKED_PATTERNS]


def _cleanup_all_managers():
    """Cleanup all active browser managers on process exit."""
    for manager in _active_managers[:]:  # Copy list to avoid modification during iteration
        try:
            manager.close()
        except Exception:
            pass
    _active_managers.clear()


# Register cleanup on exit
atexit.register(_cleanup_all_managers)


class LazyBrowserManager:
    """Manages lazy browser initialization.
    
    The browser is only launched when get_browser_tools() is first called.
    This avoids opening a browser window for OS-only or code-analysis tasks.
    
    Features:
        - Fast mode: Block images, fonts, and media for faster page loads
        - Lazy initialization: Browser opens only when needed
        - Automatic cleanup on exit
    
    Usage:
        manager = LazyBrowserManager(config)
        
        # Browser NOT opened yet
        
        tools = manager.get_browser_tools()  # Browser opens now
        
        # When done
        manager.close()
    """
    
    def __init__(self, config: "AgentConfig"):
        """Initialize the lazy browser manager.
        
        Args:
            config: Agent configuration with headless settings etc.
        """
        self.config = config
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._browser_tools: Optional["BrowserTools"] = None
        self._closed = False
        
        # Register for cleanup tracking
        _active_managers.append(self)
    
    def _should_block_resource(self, url: str) -> bool:
        """Check if a resource URL should be blocked in fast mode.
        
        Args:
            url: Resource URL to check
            
        Returns:
            True if the resource should be blocked
        """
        for pattern in _blocked_patterns_compiled:
            if pattern.match(url):
                return True
        return False
    
    def _route_handler(self, route: Route) -> None:
        """Handle route interception for fast mode.
        
        Blocks images, fonts, and media resources.
        """
        url = route.request.url
        if self._should_block_resource(url):
            logger.debug(f"Fast mode: blocking {url}")
            route.abort()
        else:
            route.continue_()
    
    def _initialize_browser(self) -> None:
        """Initialize browser, context, and page.
        
        Called lazily on first access to browser tools.
        """
        if self._closed:
            raise RuntimeError("Browser manager has been closed")
        
        if self._browser_tools is not None:
            return  # Already initialized
        
        from ..tools import BrowserTools
        
        logger.debug("LazyBrowserManager: Initializing browser (first use)")
        
        # Start Playwright
        self._playwright = sync_playwright().start()
        
        # Launch browser
        self._browser = self._playwright.chromium.launch(
            headless=self.config.headless,
        )
        
        # Create context with standard viewport
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        )
        
        # Set up fast mode routing if enabled
        if getattr(self.config, 'browser_fast_mode', False):
            logger.info("Fast mode enabled: blocking images, fonts, and media")
            self._context.route("**/*", self._route_handler)
        
        # Create page
        self._page = self._context.new_page()
        
        # Create browser tools wrapper
        self._browser_tools = BrowserTools(self._page)
        
        logger.debug("LazyBrowserManager: Browser initialized successfully")
    
    def get_browser_tools(self) -> "BrowserTools":
        """Get browser tools, initializing browser if needed.
        
        Returns:
            BrowserTools instance for browser operations
            
        Raises:
            RuntimeError: If manager has been closed
        """
        self._initialize_browser()
        return self._browser_tools
    
    def is_browser_open(self) -> bool:
        """Check if browser has been initialized.
        
        Returns:
            True if browser is currently open
        """
        return self._browser is not None and not self._closed
    
    def close(self) -> None:
        """Close browser and cleanup resources.
        
        Safe to call multiple times.
        """
        if self._closed:
            return
        
        self._closed = True
        
        # Remove from active managers
        if self in _active_managers:
            _active_managers.remove(self)
        
        if self._browser_tools:
            logger.debug("LazyBrowserManager: Closing browser")
        
        # Close in reverse order
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        
        self._browser_tools = None
        self._page = None
    
    def __del__(self):
        """Destructor - ensures cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self) -> "LazyBrowserManager":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

