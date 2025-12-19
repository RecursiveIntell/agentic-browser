# Agentic Browser

A Linux-first **multi-agent system** that controls your **web browser**, **local system**, and specialized tools. Powered by **LangGraph** for intelligent task orchestration.

## 🚀 Key Features

*   **12-Node Agent Architecture**: Built on [LangGraph](https://langchain-ai.github.io/langgraph/) with a Supervisor that routes to 11 specialized agents.
*   **Multi-Provider LLM Support**: 
    *   **OpenAI** (GPT-4o, o1-preview)
    *   **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus)
    *   **Google** (Gemini 1.5 Pro/Flash, Gemini 2.0)
    *   **LM Studio** (Local LLMs - Llama 3, Qwen 2.5)
*   **Comprehensive Automation**: Browser, OS, research, code analysis, and 6 new specialized agents.
*   **Safety Guardrails**: Risk-based permissions with user approval for dangerous operations.
*   **Modern GUI**: Dark-themed interface with real-time progress tracking.

## 🤖 Available Agents

| Agent | Purpose | Example |
|-------|---------|---------|
| **Browser** | Web navigation & interaction | "Navigate to github.com" |
| **Research** | Multi-source web research | "Research quantum computing" |
| **OS** | File operations & shell commands | "List files in Downloads" |
| **Code** | Project analysis & understanding | "Analyze my Python project" |
| **Data** | Format conversion & processing | "Convert data.json to CSV" |
| **Network** | Diagnostics & API testing | "Ping google.com" |
| **SysAdmin** | System monitoring & services | "Check memory usage" |
| **Media** | Video/audio/image processing | "Convert video to MP3" |
| **Package** | Python/Node.js environment setup | "Create a venv and install Flask" |
| **Automation** | Notifications & scheduling | "Remind me in 5 minutes" |

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/RecursiveIntell/agentic-browser.git
cd agentic-browser

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Install Playwright browsers
python -m playwright install chromium
```

### Optional: Media Agent Dependencies
```bash
# For video/audio processing
sudo dnf install ffmpeg  # or apt install ffmpeg

# For image processing
sudo dnf install ImageMagick  # or apt install imagemagick
```

## ⚡ Quick Start

### GUI Mode (Recommended)
```bash
agentic-browser gui
```

### CLI Mode
```bash
# Research task
agentic-browser run "Research the latest developments in AI agents"

# Code analysis
agentic-browser run "Analyze my project in ~/Coding/MyApp"

# System monitoring
agentic-browser run "Check what's using my disk space"

# Media conversion
agentic-browser run "Convert ~/Videos/presentation.mp4 to MP3"

# Network diagnostics
agentic-browser run "Check if my web server on port 8080 is responding"
```

## 🛠️ Configuration

Settings stored in `~/.agentic_browser/settings.json`.

### CLI Options
| Flag | Description |
|------|-------------|
| `--model-endpoint` | Custom LLM API endpoint URL |
| `--model` | Model name (e.g., `gpt-4o`, `claude-3-5-sonnet`) |
| `--headless` | Run browser in headless mode |
| `--auto-approve` | Skip approval prompts for medium-risk actions |
| `--fresh-profile` | Start with a clean browser profile |
| `--enable-tracing` | Save Playwright traces for debugging |
| `--fast` | Block images, fonts, and media for faster page loads |

### Environment Variables
```bash
# Provider (default: lm_studio for local)
AGENTIC_BROWSER_PROVIDER=openai  # openai, anthropic, google, lm_studio

# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Supervisor                         │
│     (Routes tasks based on goal analysis)               │
└─────────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    ▼                 ▼                 ▼
┌─────────┐     ┌─────────┐       ┌─────────┐
│ Browser │     │   OS    │       │ Research│
│  Agent  │     │  Agent  │       │  Agent  │
└─────────┘     └─────────┘       └─────────┘
    ▼                 ▼                 ▼
┌─────────┐     ┌─────────┐       ┌─────────┐
│  Code   │     │  Data   │       │ Network │
│  Agent  │     │  Agent  │       │  Agent  │
└─────────┘     └─────────┘       └─────────┘
    ▼                 ▼                 ▼
┌─────────┐     ┌─────────┐       ┌─────────┐
│SysAdmin │     │  Media  │       │ Package │
│  Agent  │     │  Agent  │       │  Agent  │
└─────────┘     └─────────┘       └─────────┘
                      ▼
              ┌─────────────┐
              │ Automation  │
              │   Agent     │
              └─────────────┘
```

## 🛡️ Safety & Permissions

Actions are classified by risk level:

| Risk | Examples | Behavior |
|------|----------|----------|
| **HIGH** | `rm`, `sudo`, service control, system files | Always asks |
| **MEDIUM** | Writing files, pip install, HTTP POST | Asks unless auto-approve |
| **LOW** | `ls`, `cat`, ping, read files | Allowed |

### Protected Resources
- Critical services (systemd, dbus, NetworkManager) cannot be managed
- pip install only works in virtual environments (not system-wide)
- Files outside home directory are restricted

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Browser doesn't start | Run `playwright install chromium` |
| Model 404 error | Check API key and model name |
| Permission errors | Agent uses your user permissions |
| ffmpeg not found | Install ffmpeg for Media agent |
| Service control fails | May need sudo (shown as command to run) |

## 📈 Recent Updates

### Safety & Reliability (v0.3.0)
- **Typed Tool Schemas**: All OS/browser actions now use Pydantic validation - no more raw shell strings
- **Policy Engine**: Hard denylist for dangerous commands (`rm -rf /`, fork bombs), dry-run mode
- **Structured Memory Store**: JSON-backed storage for known sites, directories, and recipes with PII redaction
- **Logging Framework**: Debug prints replaced with `logging` module (enable with `AGENTIC_BROWSER_DEBUG=1`)

### Agent Improvements
- **Smart Routing**: Supervisor correctly routes to all 10+ agent types (sysadmin, network, etc.)
- **Research Quality**: Increased minimum sources to 5, better instruction following
- **Dynamic Paths**: OS agent uses `Path.home()` instead of hardcoded paths
- **o3 Model Support**: Fixed empty response issues with OpenAI reasoning models

### CLI Features
- `agentic-browser memory --show`: View stored sites/directories/recipes
- `agentic-browser memory --bootstrap`: Pre-seed common directories
- `agentic-browser --dry-run`: See planned actions without execution
- `agentic-browser --explain`: Show risk analysis for commands

## 📄 License

MIT License - see [LICENSE](LICENSE)
