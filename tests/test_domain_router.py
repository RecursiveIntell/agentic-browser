"""
Tests for domain router functionality.
"""

import pytest

from agentic_browser.domain_router import DomainRouter, DomainDecision


class TestDomainDecision:
    """Tests for DomainDecision dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        decision = DomainDecision(
            domain="browser",
            confidence=0.85,
            reason="URL detected",
            forced_by_user=False,
        )
        result = decision.to_dict()
        
        assert result["domain"] == "browser"
        assert result["confidence"] == 0.85
        assert result["reason"] == "URL detected"
        assert result["forced_by_user"] is False


class TestDomainRouterManualModes:
    """Tests for manual routing mode overrides."""
    
    @pytest.fixture
    def router(self):
        return DomainRouter()
    
    def test_browser_mode(self, router):
        """Test browser-only mode forces browser domain."""
        decision = router.route("list files in my home directory", mode="browser")
        
        assert decision.domain == "browser"
        assert decision.confidence == 1.0
        assert decision.forced_by_user is True
    
    def test_os_mode(self, router):
        """Test OS-only mode forces OS domain."""
        decision = router.route("search google for python tutorials", mode="os")
        
        assert decision.domain == "os"
        assert decision.confidence == 1.0
        assert decision.forced_by_user is True
    
    def test_ask_mode(self, router):
        """Test ask mode returns unknown for caller to prompt user."""
        decision = router.route("do something", mode="ask")
        
        assert decision.domain == "unknown"
        assert decision.confidence == 0.0
        assert decision.forced_by_user is True


class TestDomainRouterHeuristics:
    """Tests for heuristic-based routing."""
    
    @pytest.fixture
    def router(self):
        return DomainRouter()
    
    def test_url_triggers_browser(self, router):
        """Test that URL presence triggers browser domain."""
        decision = router.route("go to https://google.com and search for news", mode="auto")
        
        assert decision.domain == "browser"
        assert decision.confidence >= 0.75
    
    def test_website_keywords_trigger_browser(self, router):
        """Test that browser keywords trigger browser domain."""
        test_cases = [
            "click the login button on the website",
            "search google for python tutorials",
            "scroll down the page",
            "navigate to the homepage",
        ]
        
        for goal in test_cases:
            decision = router.route(goal, mode="auto")
            assert decision.domain == "browser", f"Failed for: {goal}"
    
    def test_file_keywords_trigger_os(self, router):
        """Test that file/OS keywords trigger OS domain."""
        test_cases = [
            "list all files in the current directory",
            "read the contents of config.json",
            "create a new folder called projects",
            "run the install script in terminal",
            "check if port 8080 is in use",
        ]
        
        for goal in test_cases:
            decision = router.route(goal, mode="auto")
            assert decision.domain == "os", f"Failed for: {goal}"
    
    def test_mixed_signals_lower_confidence(self, router):
        """Test that mixed signals result in lower confidence."""
        # Contains both browser and OS signals
        decision = router.route("download the file from the website", mode="auto")
        
        # Should still make a decision but with potentially lower confidence
        assert decision.domain in ("browser", "os", "both")
    
    def test_no_signals_defaults_to_browser(self, router):
        """Test that ambiguous requests default to browser."""
        decision = router.route("help me with this task", mode="auto")
        
        assert decision.domain == "browser"
        assert decision.confidence < 0.5  # Low confidence
    
    def test_file_path_triggers_os(self, router):
        """Test that file paths trigger OS domain."""
        test_cases = [
            "read the file at /home/user/document.txt",
            "list contents of ~/projects",
            "check ./config.yaml",
        ]
        
        for goal in test_cases:
            decision = router.route(goal, mode="auto")
            assert decision.domain == "os", f"Failed for: {goal}"


class TestDomainRouterActionClassification:
    """Tests for action type classification."""
    
    def test_browser_actions(self):
        """Test browser action detection."""
        browser_actions = [
            "goto", "click", "type", "press", "scroll",
            "wait_for", "extract", "extract_visible_text",
            "screenshot", "back", "forward", "done",
            "download_file", "download_image",
        ]
        
        for action in browser_actions:
            assert DomainRouter.is_browser_action(action), f"Failed for: {action}"
            assert not DomainRouter.is_os_action(action), f"Should not be OS: {action}"
    
    def test_os_actions(self):
        """Test OS action detection."""
        os_actions = [
            "os_exec", "os_list_dir", "os_read_file", "os_write_file",
            "os_move_file", "os_copy_file", "os_delete_file",
        ]
        
        for action in os_actions:
            assert DomainRouter.is_os_action(action), f"Failed for: {action}"
            assert not DomainRouter.is_browser_action(action), f"Should not be browser: {action}"

    def test_memory_actions(self):
        """Test memory action detection."""
        memory_actions = [
            "memory_get_site",
            "memory_save_site",
            "memory_get_directory",
        ]

        for action in memory_actions:
            assert DomainRouter.is_memory_action(action), f"Failed for: {action}"
            assert not DomainRouter.is_browser_action(action), f"Should not be browser: {action}"
            assert not DomainRouter.is_os_action(action), f"Should not be OS: {action}"
    
    def test_unknown_action(self):
        """Test that unknown actions are neither browser nor OS."""
        assert not DomainRouter.is_browser_action("unknown_action")
        assert not DomainRouter.is_os_action("unknown_action")
        assert not DomainRouter.is_memory_action("unknown_action")
