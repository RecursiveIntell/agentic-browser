# Agentic Browser

A Linux-first agentic browser runner that controls Chromium via Playwright and executes user goals step-by-step with LLM-powered decision making and safety guardrails.

## Features

- 🌐 **Browser Automation**: Controls Chromium with persistent profiles (cookies/logins preserved)
- 🤖 **LLM-Powered**: Uses any OpenAI-compatible API (LM Studio, OpenAI, Anthropic, Google AI)
- 🛡️ **Safety Guardrails**: Risk classification with approval prompts for dangerous actions
- 📝 **Comprehensive Logging**: Saves every step, screenshot, and decision to artifacts
- 🔄 **Self-Recovery**: Automatically handles errors and retries with alternative approaches
- 🎯 **Goal-Oriented**: Natural language goals executed through iterative planning
- 🖥️ **GUI & CLI**: Both graphical and command-line interfaces

## Installation

### Prerequisites

- Linux (tested on Fedora/Nobara)
- Python 3.10+ (3.11+ recommended)
- An LLM server (LM Studio, OpenAI, Anthropic, or Google AI)

### Install from source

```bash
# Clone the repository
git clone https://github.com/RecursiveIntell/agentic-browser.git
cd agentic-browser

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Install Playwright Chromium
python -m playwright install chromium
```

## Quick Start

### GUI Mode (Recommended)

```bash
agentic-browser gui
```

This opens a graphical interface where you can:
- Enter goals in natural language
- Configure your LLM provider and API key
- Watch the agent execute steps in real-time
- Approve or deny risky actions

### CLI Mode

```bash
# Simple example
agentic-browser run "Open example.com and tell me the title"

# Search the web
agentic-browser run "Search the web for Playwright and open the docs"

# With specific model
agentic-browser run "Check the weather" --model gpt-4o-mini

# Headless mode
agentic-browser run "Scrape headlines from news.ycombinator.com" --headless
```

## GUI Features

### Settings Dialog

The GUI settings dialog allows you to configure:

| Setting | Description |
|---------|-------------|
| **Provider** | LM Studio, OpenAI, Anthropic, or Google AI |
| **API Key** | Required for cloud providers (not needed for LM Studio) |
| **Model** | Model name (click 🔄 Refresh to fetch available models) |
| **Custom Endpoint** | Override the default API endpoint |
| **Profile Name** | Browser profile for persistent sessions |
| **Max Steps** | Maximum actions before stopping |
| **Headless** | Run browser without visible window |
| **Auto-Approve** | Skip approval for medium-risk actions |

### Provider Configuration

| Provider | Default Endpoint | API Key Required |
|----------|-----------------|------------------|
| LM Studio (Local) | http://127.0.0.1:1234/v1 | No |
| OpenAI | https://api.openai.com/v1 | Yes |
| Anthropic | https://api.anthropic.com/v1 | Yes |
| Google AI | https://generativelanguage.googleapis.com/v1beta | Yes |

**Model Refresh**: After entering your API key, click "🔄 Refresh" to fetch the list of available models from your provider.

## CLI Reference

```
agentic-browser run "GOAL" [OPTIONS]
agentic-browser gui
```

### Run Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile NAME` | `default` | Browser profile name |
| `--headless` | `false` | Run browser in headless mode |
| `--max-steps N` | `30` | Maximum steps before stopping |
| `--model-endpoint URL` | `http://127.0.0.1:1234/v1` | LLM API endpoint |
| `--model NAME` | `qwen2.5:7b` | LLM model name |
| `--auto-approve` | `false` | Auto-approve medium-risk actions |
| `--fresh-profile` | `false` | Create a fresh browser profile |
| `--no-persist` | `false` | Use a temporary profile |
| `--enable-tracing` | `false` | Enable Playwright tracing |

### Environment Variables

- `AGENTIC_BROWSER_ENDPOINT`: Default LLM endpoint
- `AGENTIC_BROWSER_MODEL`: Default model name  
- `AGENTIC_BROWSER_API_KEY`: API key (if required)

## Safety Model

The agent classifies every action into risk levels:

### High Risk (always requires approval)
- Purchase buttons: "Buy", "Checkout", "Pay", "Order"
- Account actions: "Delete", "Remove", "Cancel Account"
- Messaging: "Send", "Post", "Submit"
- Payment domains: PayPal, Stripe, etc.

### Medium Risk (requires approval unless `--auto-approve`)
- Login forms (password fields)
- File uploads
- Permission grants

### Low Risk (no approval needed)
- Navigation, scrolling, reading, screenshots

### Approval Dialog

When approval is required, you'll see the action details and can:
- **Approve** - Execute the action
- **Deny** - Ask the agent for an alternative
- **Edit** - Manually modify the action JSON

## Artifacts and Logging

Every run creates artifacts at:
```
~/.agentic_browser/runs/<timestamp>_<goal_slug>/
├── steps.jsonl          # Every step with state, action, result
├── screenshots/         # Screenshots at each step
├── snapshots/           # HTML/text page snapshots
└── trace.zip           # Playwright trace (if enabled)
```

Settings are stored at:
```
~/.agentic_browser/settings.json
```

Browser profiles are stored at:
```
~/.agentic_browser/profiles/<profile_name>/
```

## Action Schema

The LLM responds with JSON actions:

```json
{
  "action": "goto|click|type|press|scroll|wait_for|extract|extract_visible_text|screenshot|back|forward|done",
  "args": { ... },
  "rationale": "short reason",
  "risk": "low|medium|high",
  "requires_approval": true/false,
  "final_answer": "only when action=done"
}
```

### Available Actions

| Action | Arguments | Description |
|--------|-----------|-------------|
| `goto` | `url` | Navigate to URL |
| `click` | `selector`, `timeout_ms` | Click element |
| `type` | `selector`, `text`, `clear_first` | Type into input |
| `press` | `key` | Press keyboard key |
| `scroll` | `amount` | Scroll (positive=down) |
| `wait_for` | `selector`, `timeout_ms` | Wait for element |
| `extract` | `selector`, `attribute` | Extract element data |
| `extract_visible_text` | `max_chars` | Get visible text |
| `screenshot` | `label` | Capture screenshot |
| `back` | (none) | Navigate back |
| `forward` | (none) | Navigate forward |
| `done` | `summary_style` | Complete task |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_json_parser.py -v
pytest tests/test_safety_classifier.py -v
```

## Project Structure

```
agentic_browser/
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── agent.py            # Main agent loop
├── llm_client.py       # LLM API client with retry logic
├── tools.py            # Browser action implementations
├── safety.py           # Risk classification
├── logger.py           # Logging and artifacts
├── utils.py            # Utility functions
├── providers.py        # LLM provider configuration
├── model_fetcher.py    # Fetch models from provider APIs
├── settings_store.py   # Persistent settings storage
└── gui/
    ├── main_window.py      # Main application window
    ├── settings_dialog.py  # Settings configuration
    └── approval_dialog.py  # Action approval prompts
```

## Design Notes

### Adding New Tools

1. Add the action to `tools.py`:
```python
def my_new_action(self, param: str) -> ToolResult:
    """Implement your action."""
    # ... implementation
    return ToolResult(success=True, message="Done")
```

2. Register in the `execute()` dispatch map:
```python
method_map = {
    # ... existing actions
    "my_new_action": self.my_new_action,
}
```

3. Update the LLM system prompt in `llm_client.py` to document the new action.

### Adding New Providers

1. Add to `providers.py`:
```python
class Provider(str, Enum):
    # ... existing
    NEW_PROVIDER = "new_provider"

PROVIDER_ENDPOINTS[Provider.NEW_PROVIDER] = "https://api.newprovider.com/v1"
```

2. Add model fetching in `model_fetcher.py` if the provider has a models API.

### Improving Reliability

1. **Better selectors**: Teach the agent about aria-labels, data-testid, etc.
2. **Context injection**: Add more context to `_get_page_state()` in `agent.py`
3. **Recovery strategies**: Enhance `get_recovery_action()` with specific patterns
4. **Loop detection**: Adjust `max_repeat_actions` or implement smarter detection

## Troubleshooting

### "Connection refused" to LLM endpoint
- Ensure your LLM server is running
- Check the endpoint URL in settings

### Browser doesn't start
- Run `python -m playwright install chromium`
- Check for system dependencies: `playwright install-deps chromium`

### Invalid JSON from LLM
- The agent retries up to 2 times with a repair prompt
- Try a different/larger model if issues persist

### GUI doesn't launch
- Ensure PySide6 is installed: `pip install pyside6`
- On Wayland, try: `QT_QPA_PLATFORM=xcb agentic-browser gui`

## License

MIT License - see [LICENSE](LICENSE)
