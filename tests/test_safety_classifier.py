"""
Tests for safety classification functionality.
"""

import pytest

from agentic_browser.safety import (
    SafetyClassifier,
    RiskLevel,
    classify_risk,
)
from agentic_browser.utils import (
    contains_high_risk_keywords,
    contains_medium_risk_keywords,
    is_payment_domain,
    is_password_field,
)


class TestRiskKeywords:
    """Tests for risk keyword detection."""
    
    def test_high_risk_keywords(self):
        """Test high-risk keyword detection."""
        assert contains_high_risk_keywords("Buy Now")
        assert contains_high_risk_keywords("Complete Purchase")
        assert contains_high_risk_keywords("Delete Account")
        assert contains_high_risk_keywords("Send Message")
        assert not contains_high_risk_keywords("Learn More")
        assert not contains_high_risk_keywords("Read Article")
    
    def test_medium_risk_keywords(self):
        """Test medium-risk keyword detection."""
        assert contains_medium_risk_keywords("Login")
        assert contains_medium_risk_keywords("Sign In")
        assert contains_medium_risk_keywords("Upload File")
        assert contains_medium_risk_keywords("Grant Access")
        assert not contains_medium_risk_keywords("Download")
        assert not contains_medium_risk_keywords("View Details")
    
    def test_case_insensitive(self):
        """Test that keyword detection is case-insensitive."""
        assert contains_high_risk_keywords("BUY NOW")
        assert contains_high_risk_keywords("buy now")
        assert contains_high_risk_keywords("Buy Now")
        assert contains_medium_risk_keywords("LOGIN")
        assert contains_medium_risk_keywords("login")


class TestPaymentDomain:
    """Tests for payment domain detection."""
    
    def test_known_payment_domains(self):
        """Test detection of known payment domains."""
        assert is_payment_domain("https://www.paypal.com/checkout")
        assert is_payment_domain("https://checkout.stripe.com/pay")
        assert is_payment_domain("https://pay.google.com/transaction")
    
    def test_payment_paths(self):
        """Test detection of payment-related paths."""
        assert is_payment_domain("https://example.com/checkout")
        assert is_payment_domain("https://store.com/payment/process")
        assert is_payment_domain("https://shop.com/cart/checkout")
    
    def test_non_payment_domains(self):
        """Test that normal domains are not flagged."""
        assert not is_payment_domain("https://example.com")
        assert not is_payment_domain("https://github.com")
        assert not is_payment_domain("https://google.com/search")
        assert not is_payment_domain("https://docs.example.com/api-reference")


class TestPasswordField:
    """Tests for password field detection."""
    
    def test_password_selectors(self):
        """Test detection of password field selectors."""
        assert is_password_field('input[type="password"]')
        assert is_password_field('#password-input')
        assert is_password_field('.password-field')
        assert is_password_field('input#passwd')
    
    def test_non_password_selectors(self):
        """Test that non-password fields are not flagged."""
        assert not is_password_field('input[type="text"]')
        assert not is_password_field('#username')
        assert not is_password_field('.email-input')


class TestSafetyClassifier:
    """Tests for the SafetyClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create a SafetyClassifier instance."""
        return SafetyClassifier()
    
    def test_classify_click_high_risk(self, classifier):
        """Test high-risk click classification."""
        # Purchase button
        risk = classifier.classify_action(
            "click",
            {"selector": 'button:text("Buy Now")'},
            current_url="https://store.com",
        )
        assert risk == RiskLevel.HIGH
        
        # Delete button
        risk = classifier.classify_action(
            "click",
            {"selector": 'button:text("Delete Account")'},
            current_url="https://settings.example.com",
        )
        assert risk == RiskLevel.HIGH
    
    def test_classify_click_medium_risk(self, classifier):
        """Test medium-risk click classification."""
        # Submit button without high-risk context
        risk = classifier.classify_action(
            "click",
            {"selector": 'input[type="submit"]'},
            current_url="https://example.com/contact",
            page_content="Contact us form. Name, Email, Message.",
        )
        assert risk == RiskLevel.MEDIUM
    
    def test_classify_click_low_risk(self, classifier):
        """Test low-risk click classification."""
        # Regular link
        risk = classifier.classify_action(
            "click",
            {"selector": 'a:text("Learn More")'},
            current_url="https://example.com",
        )
        assert risk == RiskLevel.LOW
    
    def test_classify_type_password(self, classifier):
        """Test typing into password field is medium risk."""
        risk = classifier.classify_action(
            "type",
            {"selector": 'input[type="password"]', "text": "secret123"},
            current_url="https://example.com/login",
        )
        assert risk == RiskLevel.MEDIUM
    
    def test_classify_type_normal(self, classifier):
        """Test typing into normal field is low risk."""
        risk = classifier.classify_action(
            "type",
            {"selector": 'input[name="search"]', "text": "playwright"},
            current_url="https://google.com",
        )
        assert risk == RiskLevel.LOW
    
    def test_classify_payment_url(self, classifier):
        """Test that actions on payment URLs are high risk."""
        risk = classifier.classify_action(
            "click",
            {"selector": "button"},
            current_url="https://checkout.stripe.com/pay/123",
        )
        assert risk == RiskLevel.HIGH
    
    def test_classify_security_page(self, classifier):
        """Test that actions on security pages are high risk."""
        risk = classifier.classify_action(
            "click",
            {"selector": "button"},
            current_url="https://example.com/account/security",
        )
        assert risk == RiskLevel.HIGH
    
    def test_classify_goto_payment(self, classifier):
        """Test navigating to payment domain is medium risk."""
        risk = classifier.classify_action(
            "goto",
            {"url": "https://paypal.com/checkout"},
        )
        assert risk == RiskLevel.MEDIUM
    
    def test_classify_goto_normal(self, classifier):
        """Test navigating to normal site is low risk."""
        risk = classifier.classify_action(
            "goto",
            {"url": "https://example.com"},
        )
        assert risk == RiskLevel.LOW
    
    def test_classify_enter_on_payment_page(self, classifier):
        """Test pressing Enter on payment page is high risk."""
        risk = classifier.classify_action(
            "press",
            {"key": "Enter"},
            current_url="https://checkout.example.com/pay",
        )
        assert risk == RiskLevel.HIGH
    
    def test_classify_enter_normal(self, classifier):
        """Test pressing Enter normally is medium risk."""
        risk = classifier.classify_action(
            "press",
            {"key": "Enter"},
            current_url="https://google.com/search",
        )
        assert risk == RiskLevel.MEDIUM


class TestApprovalLogic:
    """Tests for approval requirement logic."""
    
    @pytest.fixture
    def classifier(self):
        """Create a SafetyClassifier instance."""
        return SafetyClassifier()
    
    def test_high_risk_always_requires_approval(self, classifier):
        """Test that high risk always requires approval."""
        assert classifier.should_require_approval(
            RiskLevel.HIGH, 
            model_says_approval=False,
            auto_approve=True,
        )
        assert classifier.should_require_approval(
            RiskLevel.HIGH,
            model_says_approval=False, 
            auto_approve=False,
        )
    
    def test_medium_risk_respects_auto_approve(self, classifier):
        """Test that medium risk respects auto_approve setting."""
        # Without auto-approve, requires approval
        assert classifier.should_require_approval(
            RiskLevel.MEDIUM,
            model_says_approval=False,
            auto_approve=False,
        )
        
        # With auto-approve, no approval needed
        assert not classifier.should_require_approval(
            RiskLevel.MEDIUM,
            model_says_approval=False,
            auto_approve=True,
        )
    
    def test_low_risk_no_approval(self, classifier):
        """Test that low risk doesn't require approval."""
        assert not classifier.should_require_approval(
            RiskLevel.LOW,
            model_says_approval=False,
            auto_approve=False,
        )
    
    def test_model_flag_respected(self, classifier):
        """Test that model's requires_approval flag is respected for medium risk."""
        # Model says approval needed for medium risk
        assert classifier.should_require_approval(
            RiskLevel.MEDIUM,
            model_says_approval=True,
            auto_approve=False,
        )


class TestConvenienceFunction:
    """Tests for the classify_risk convenience function."""
    
    def test_classify_risk_function(self):
        """Test the standalone classify_risk function."""
        risk = classify_risk(
            "click",
            {"selector": 'button:text("Buy Now")'},
            current_url="https://store.com",
        )
        assert risk == RiskLevel.HIGH
        
        risk = classify_risk(
            "goto",
            {"url": "https://example.com"},
        )
        assert risk == RiskLevel.LOW
