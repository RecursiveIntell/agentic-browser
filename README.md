# Agentic Browser

A Linux-first **dual-domain agent** that controls both your **web browser** (via Playwright) and your **local Linux system** (via shell commands). Uses LLM-powered decision making with safety guardrails.

## 🆕 Dual-Domain Support

The agent now intelligently routes requests between two domains:

| Domain | Capabilities | Example Goals |
|--------|--------------|---------------|
| **Browser** | Web browsing, form filling, clicking, data extraction | "Search Google for...", "Log into my account" |
| **OS/Linux** | Shell commands, file operations, system inspection | "Look at my hard drive", "List files in ~/projects" |

The routing is automatic—just describe what you want and the agent figures out whether to use the browser or local system.

## Features

### Browser Domain
- 🌐 **Browser Automation**: Controls Chromium with persistent profiles
- 🔗 **Smart Navigation**: Tracks visited URLs to avoid loops
- 📝 **Data Extraction**: Extracts text, links, and page content

### OS Domain (NEW)
- 🖥️ **Shell Commands**: Execute `ls`, `df`, `grep`, etc. via `os_exec`
- 📁 **File Operations**: Read, write, and list files
- 🛡️ **Safe by Default**: Blocks writes outside `$HOME`, requires approval for dangerous commands

### Both Domains
- 🤖 **LLM-Powered**: Uses any OpenAI-compatible API (LM Studio, OpenAI, Anthropic, Google AI)
- 🛡️ **Safety Guardrails**: Risk classification with approval prompts
- 📝 **Comprehensive Logging**: Saves every step and decision
- 🖥️ **Dark Theme GUI**: Modern graphical interface

## Installation

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

### CLI Mode

```bash
# Browser tasks
agentic-browser run "Search DuckDuckGo for Python tutorials"

# OS tasks (auto-detected)
agentic-browser run "List files in my home directory"
agentic-browser run "Check disk usage on my system"

# Force a specific domain
agentic-browser run "Check my files" --routing-mode os
```

## OS Actions

| Action | Description | Risk |
|--------|-------------|------|
| `os_exec` | Execute shell commands | LOW-HIGH |
| `os_list_dir` | List directory contents | LOW |
| `os_read_file` | Read file contents | LOW |
| `os_write_file` | Write/append to files | MEDIUM-HIGH |

### Safety Controls

- **HIGH risk** (requires approval + double-confirm): `rm -rf`, `sudo`, `dd`, `mkfs`, writes to `/etc`
- **MEDIUM risk** (requires approval): File writes, running scripts
- **LOW risk** (auto-approved): `ls`, `df`, `cat`, `grep`, reading files

### Path Restrictions

By default, file writes are **blocked outside `$HOME`**. System paths are always protected:
```
/etc /usr /bin /sbin /boot /var /lib /opt /root
```

## Browser Actions

| Action | Description |
|--------|-------------|
| `goto` | Navigate to URL |
| `click` | Click element |
| `type` | Type into input |
| `press` | Press keyboard key |
| `scroll` | Scroll page |
| `extract` | Extract element data |
| `screenshot` | Capture screenshot |
| `done` | Complete task |

## Domain Routing

The agent uses heuristic keyword matching with optional LLM fallback:

| User Goal | Routed To | Signal |
|-----------|-----------|--------|
| "Search Google for..." | Browser | URL keyword |
| "List my files" | OS | File keywords |
| "Check disk usage" | OS | System keywords |
| "Open https://..." | Browser | URL pattern |

### Routing Modes

- **auto** (default): Automatic routing via keywords
- **browser**: Force all tasks to browser
- **os**: Force all tasks to OS
- **ask**: Prompt before each task

## Configuration

Settings stored at `~/.agentic_browser/settings.json`:

| Setting | Description |
|---------|-------------|
| Provider | LM Studio, OpenAI, Anthropic, Google AI |
| Routing Mode | auto, browser, os, ask |
| Max Steps | Maximum actions per task |
| Auto-Approve | Skip approval for medium-risk |

### Environment Variables

```bash
AGENTIC_BROWSER_ENDPOINT  # LLM endpoint
AGENTIC_BROWSER_MODEL     # Model name
AGENTIC_BROWSER_API_KEY   # API key
AGENTIC_OS_ENDPOINT       # Separate OS model endpoint (optional)
```

## Safety Model

### HIGH Risk (always requires approval)
- Browser: Purchase buttons, account deletion, sending messages
- OS: `rm -rf`, `sudo`, `dd`, `chmod -R`, system path writes

### MEDIUM Risk (requires approval unless auto-approve)
- Browser: Login forms, file uploads
- OS: File writes, running scripts

### LOW Risk (no approval needed)
- Browser: Navigation, reading, screenshots
- OS: `ls`, `df`, `cat`, `grep`, reading files

## Artifacts and Logging

Every run creates artifacts at:
```
~/.agentic_browser/runs/<timestamp>_<goal>/
├── steps.jsonl        # Every step with state and result
├── screenshots/       # Screenshots at each step
└── snapshots/         # Page snapshots
```

## Project Structure

```
agentic_browser/
├── agent.py            # Main agent loop (browser + OS routing)
├── domain_router.py    # Intelligent domain routing
├── tool_router.py      # Action dispatch to tools
├── os_tools.py         # OS action implementations
├── tools.py            # Browser action implementations
├── safety.py           # Risk classification (browser + OS)
├── llm_client.py       # LLM client with OS/browser prompts
├── config.py           # Configuration management
└── gui/                # Dark theme GUI
```

## Troubleshooting

### OS commands return permission errors
- The agent can only access what your user can access
- System directories require `sudo` which triggers HIGH risk approval

### Agent loops instead of summarizing
- Updated prompt now enforces 5-step limit
- Agent must call `done` after 3-5 commands

### Browser doesn't start
```bash
python -m playwright install chromium
```

## Recommended Models

| Provider | Model | Notes |
|----------|-------|-------|
| LM Studio | Llama 3.1 8B, Qwen 2.5 7B | Free, local |
| OpenAI | gpt-4o-mini, gpt-4o | Best instruction following |
| Anthropic | claude-3-sonnet | Good reasoning |
| Google | gemini-1.5-flash | Fast and capable |

## License

MIT License - see [LICENSE](LICENSE)
