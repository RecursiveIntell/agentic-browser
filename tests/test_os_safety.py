"""
Tests for OS safety classification functionality.
"""

import pytest

from agentic_browser.safety import SafetyClassifier, RiskLevel


class TestOSActionClassification:
    """Tests for OS action risk classification."""
    
    @pytest.fixture
    def classifier(self):
        return SafetyClassifier()
    
    # os_exec tests
    def test_os_exec_rm_rf_high_risk(self, classifier):
        """Test rm -rf is classified as high risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "rm -rf /some/path"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_exec_sudo_high_risk(self, classifier):
        """Test sudo commands are high risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "sudo apt update"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_exec_dd_high_risk(self, classifier):
        """Test dd command is high risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "dd if=/dev/zero of=/dev/sda"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_exec_shutdown_high_risk(self, classifier):
        """Test shutdown command is high risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "shutdown now"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_exec_chmod_recursive_high_risk(self, classifier):
        """Test recursive chmod is high risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "chmod -R 777 /"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_exec_mv_medium_risk(self, classifier):
        """Test mv command is medium risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "mv file1.txt file2.txt"},
        )
        assert risk == RiskLevel.MEDIUM
    
    def test_os_exec_mkdir_medium_risk(self, classifier):
        """Test mkdir is medium risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "mkdir /home/user/new_folder"},
        )
        assert risk == RiskLevel.MEDIUM
    
    def test_os_exec_ls_low_risk(self, classifier):
        """Test ls command is low risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "ls -la"},
        )
        assert risk == RiskLevel.LOW
    
    def test_os_exec_cat_low_risk(self, classifier):
        """Test cat command is low risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "cat /etc/os-release"},
        )
        assert risk == RiskLevel.LOW
    
    def test_os_exec_grep_low_risk(self, classifier):
        """Test grep command is low risk."""
        risk = classifier.classify_action(
            "os_exec",
            {"cmd": "grep -r 'pattern' ."},
        )
        assert risk == RiskLevel.LOW
    
    # os_write_file tests
    def test_os_write_file_etc_high_risk(self, classifier):
        """Test writing to /etc is high risk."""
        risk = classifier.classify_action(
            "os_write_file",
            {"path": "/etc/hosts", "content": "test"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_write_file_outside_home_high_risk(self, classifier):
        """Test writing outside home is high risk."""
        risk = classifier.classify_action(
            "os_write_file",
            {"path": "/tmp/outside_home.txt", "content": "test"},
        )
        assert risk == RiskLevel.HIGH
    
    def test_os_write_file_home_medium_risk(self, classifier):
        """Test writing to home is medium risk."""
        import os
        home = os.path.expanduser("~")
        risk = classifier.classify_action(
            "os_write_file",
            {"path": f"{home}/test.txt", "content": "test"},
        )
        assert risk == RiskLevel.MEDIUM
    
    # os_read_file tests
    def test_os_read_file_ssh_key_medium_risk(self, classifier):
        """Test reading SSH keys is medium risk."""
        risk = classifier.classify_action(
            "os_read_file",
            {"path": "/home/user/.ssh/id_rsa"},
        )
        assert risk == RiskLevel.MEDIUM
    
    def test_os_read_file_normal_low_risk(self, classifier):
        """Test reading normal files is low risk."""
        risk = classifier.classify_action(
            "os_read_file",
            {"path": "/home/user/document.txt"},
        )
        assert risk == RiskLevel.LOW
    
    # os_list_dir tests
    def test_os_list_dir_always_low_risk(self, classifier):
        """Test listing directories is always low risk."""
        risk = classifier.classify_action(
            "os_list_dir",
            {"path": "/etc"},
        )
        assert risk == RiskLevel.LOW


class TestDoubleConfirm:
    """Tests for double confirmation requirement."""
    
    @pytest.fixture
    def classifier(self):
        return SafetyClassifier()
    
    def test_rm_requires_double_confirm(self, classifier):
        """Test rm -rf requires double confirmation."""
        needs, reason = classifier.requires_double_confirm(
            "os_exec",
            {"cmd": "rm -rf /some/path"},
        )
        assert needs is True
        assert "deletion" in reason.lower()
    
    def test_dd_requires_double_confirm(self, classifier):
        """Test dd requires double confirmation."""
        needs, reason = classifier.requires_double_confirm(
            "os_exec",
            {"cmd": "dd if=/dev/zero of=/dev/sda"},
        )
        assert needs is True
        assert "disk" in reason.lower()
    
    def test_sudo_requires_double_confirm(self, classifier):
        """Test sudo requires double confirmation."""
        needs, reason = classifier.requires_double_confirm(
            "os_exec",
            {"cmd": "sudo rm file.txt"},
        )
        assert needs is True
        assert len(reason) > 0  # Has a reason
    
    def test_shutdown_requires_double_confirm(self, classifier):
        """Test shutdown requires double confirmation."""
        needs, reason = classifier.requires_double_confirm(
            "os_exec",
            {"cmd": "shutdown -h now"},
        )
        assert needs is True
    
    def test_ls_no_double_confirm(self, classifier):
        """Test ls does not require double confirmation."""
        needs, reason = classifier.requires_double_confirm(
            "os_exec",
            {"cmd": "ls -la"},
        )
        assert needs is False
        assert reason == ""
    
    def test_non_exec_no_double_confirm(self, classifier):
        """Test non-exec actions don't require double confirm."""
        needs, reason = classifier.requires_double_confirm(
            "os_write_file",
            {"path": "/home/user/file.txt", "content": "test"},
        )
        assert needs is False


class TestBrowserActionsUnchanged:
    """Ensure browser action classification still works."""
    
    @pytest.fixture
    def classifier(self):
        return SafetyClassifier()
    
    def test_click_purchase_still_high_risk(self, classifier):
        """Test browser purchase buttons still high risk."""
        risk = classifier.classify_action(
            "click",
            {"selector": 'button:text("Buy Now")'},
            current_url="https://store.com",
        )
        assert risk == RiskLevel.HIGH
    
    def test_goto_low_risk(self, classifier):
        """Test regular navigation still low risk."""
        risk = classifier.classify_action(
            "goto",
            {"url": "https://example.com"},
        )
        assert risk == RiskLevel.LOW
