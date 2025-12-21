# Agentic Browser

A Linux-first **multi-agent system** that controls your **web browser**, **local system**, and specialized tools. Powered by **LangGraph** for intelligent task orchestration.

## 🚀 Key Features

*   **13-Node Agent Architecture**: Built on [LangGraph](https://langchain-ai.github.io/langgraph/) with a Supervisor that routes to 12 specialized agents including Planner and Retrospective agents for complex task handling.
*   **Multi-Provider LLM Support**: 
    *   **OpenAI** (GPT-4o, o1-preview, o3-mini)
    *   **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus)
    *   **Google** (Gemini 1.5 Pro/Flash, Gemini 2.0)
    *   **LM Studio** (Local LLMs - Llama 3, Qwen 2.5)
*   **Intelligent Research**: Unique URL tracking, CAPTCHA detection, and automatic content extraction with minimum source requirements.
*   **Safety Guardrails**: Policy engine with risk-based permissions, typed tool schemas, and user approval for dangerous operations.
*   **Strategy Bank 🧠**: Learns from success. Automatically "crystallizes" successful runs into reusable high-level strategies, allowing agents to solve new problems by recalling proven methods.
*   **Persistent Memory**: SQLite-based session storage (last 150 runs) with semantic search ("recall tool") so agents don't repeat mistakes.
*   **Modern GUI**: Dark-themed interface with real-time progress tracking, settings persistence, and live "Thinking" indicators.
*   **n8n Integration**: Trigger external workflows via webhooks with the specialized `WorkflowAgent`.
*   **Comprehensive Testing**: 14 test suites covering integration, tool schemas, policy engine, and provider adapters.

## 🤖 Available Agents

| Agent | Purpose | Example |
|-------|---------|---------|
| **Supervisor** | Task routing & orchestration | Routes to appropriate agents |
| **Planner** | Complex task decomposition | Multi-step task planning |
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
| **Workflow** | n8n/Webhook Integration | "Trigger the content pipeline" |
| **Retrospective** | Learning from past actions | Improves future task handling |

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

# n8n Integration (Optional)
N8N_URL=https://n8n.example.com
N8N_API_KEY=...
```

### OS & Routing Configuration
You can explicitly set a separate provider for OS/System tasks vs Browser tasks:
```json
// in settings.json
{
  "provider": "openai",           // Browser agent uses GPT-4o
  "os_provider": "lm_studio",     // OS agent uses local Llama 3
  "routing_mode": "auto"          // auto | browser | os | ask
}
```

## 🏗️ Architecture

**Planning-First Architecture**: Tasks start with the Planner, which decomposes complex goals before the Supervisor routes to specialized agents.

```
                    ┌───────────────┐
                    │    Planner    │ ◄── Entry Point
                    │  (Task Plan)  │
                    └───────┬───────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                       Supervisor                          │
│         (Routes tasks & synthesizes results)              │
└─────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬──────┘
          │     │     │     │     │     │     │     │
    ┌─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┐
    │     │     │     │     │     │     │     │     │     │
    ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│Browse│  OS  │Resear│ Code │ Data │Netwrk│SysAdm│Media │ Pkg  │ Auto │
│  r   │      │  ch  │      │      │      │  in  │      │      │      │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
    │     │     │     │     │     │     │     │     │     │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                            │
                            ▼
                    Return to Supervisor
                      (or END if done)
```

**Flow:**
1. **Planner** analyzes the goal and creates an execution plan
2. **Supervisor** receives the plan and routes to appropriate agents
3. **Worker Agents** execute tasks and return results to Supervisor
4. **Supervisor** continues routing or synthesizes final answer

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

### Code Quality & Best Practices (v0.7.1)
- **Type Safety**: New `types.py` module with typed dataclasses (`ActionData`, `TokenUsage`, `PageState`) for better IDE support
- **Ruff Integration**: Linting and formatting configured in `pyproject.toml` (run `ruff check --fix`)
- **mypy Support**: Static type checking configured, `py.typed` marker for PEP 561 compliance
- **Structured Logging**: Debug `print()` statements migrated to `logging` module across core agents
- **TypedDict State**: All internal state keys now properly typed in `AgentState` schema

### Development Setup
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter (auto-fixes most issues)
ruff check agentic_browser/ --fix

# Run type checker (informational)
mypy agentic_browser/

# Run tests
pytest tests/ -v
```

### Performance Revolution (v0.7.0)
- **Asynchronous Core**: Migration to non-blocking I/O for database writes, eliminating UI freezes during long runs.
- **Knowledge Base Caching**: 500-entry LRU embedding cache with MD5 hashing, delivering 1000x speedup for repeated queries.
- **Titanium State Management**: Bounded message history (40 msgs) and URL tracking (50 domains) to prevent memory leaks and OOM crashes.
- **Smart BrowserOps**: URL-based visual diffing skips redundant screenshot capture, saving ~300ms per step.
- **Tiered Recall**: Parallel execution of keyword, vector, and hybrid search strategies using `ThreadPoolExecutor`.

### Performance & Stability (v0.6.0)
- **Smart Context Compression**: Intelligent synthesis that extracts key info and compresses history to 18k chars (preferring 3.5k-5k), preventing context explosion and 429 errors.
- **Deep Performance Tuning**: 
    - Reduced Playwright `click` timeouts to 1.5s and `evaluate` to 5s.
    - Added 30s timeout guard to `page.goto` to prevent indefinite hangs.
    - Vision mode `detail: low` optimization, saving ~90% of screenshot tokens.
    - Disabled LangGraph `MemorySaver` to eliminate exponential memory growth and "Step 13" hangs.
- **Apocalypse Learning**: Automatically extracts error patterns from failures and saves them to a "Apocalypse Bank" to proactively avoid repeating mistakes.
- **Secure Strategy Storage**: AES-256 encryption for all learned strategies and apocalypse entries using a machine-specific hardware ID.
- **Improved GUI Snappiness**: Background pre-warming for embedding models and cleaner browser UI with `--disable-infobars`.

### Research Agent Improvements (v0.4.0)
- **Unique URL Tracking**: Counts unique content URLs visited (excluding search engines) rather than extraction count
- **CAPTCHA Detection**: Automatically detects blocked/CAPTCHA pages and marks them as visited to break loops
- **Hallucinated Selector Detection**: Detects when LLM suggests non-existent elements, forces scroll to reveal more content
- **Smart Scroll Integration**: Auto-scrolls when agent tries to click duplicate links, updates LLM context after scrolls
- **Click Failure Handling**: Prevents infinite retry loops on failed clicks with proper detection and recovery
- **Shared State**: `clicked_selectors` now shared between browser and research agents to avoid re-clicking

### Integration & Testing (v0.4.0)
- **14 Comprehensive Test Suites**: Including integration tests, tool schemas, policy engine, and provider adapters
- **State Schema Validation**: Tests verify all agent state fields and operator configurations
- **Function Signature Tests**: Validates all tool functions have correct return types and parameters

### Bug Fixes (v0.4.0)
- **Scroll Args Performance**: Fixed unbounded growth of scroll arguments in state
- **Auto-Extract Loop**: Marks URLs as visited after extraction to prevent re-extraction
- **Source Numbering**: Fixed research agent source numbering and scroll hints
- **Context Update**: Forces content extraction after scroll to update LLM context
- **State Fields**: Added `last_action_was_scroll` as proper state field with reducer

### Safety & Reliability (v0.3.0)
- **Typed Tool Schemas**: All OS/browser actions now use Pydantic validation - no more raw shell strings
- **Policy Engine**: Hard denylist for dangerous commands (`rm -rf /`, fork bombs), dry-run mode
- **Structured Memory Store**: JSON-backed storage for known sites, directories, and recipes with PII redaction
- **Logging Framework**: Debug prints replaced with `logging` module (enable with `AGENTIC_BROWSER_DEBUG=1`)

### Agent Improvements (v0.3.0)
- **Smart Routing**: Supervisor correctly routes to all 12+ agent types (sysadmin, network, etc.)
- **Dynamic Paths**: OS agent uses `Path.home()` instead of hardcoded paths
- **o3 Model Support**: Fixed empty response issues with OpenAI reasoning models
- **Provider Adapters**: Native adapters for Anthropic, Google GenAI, and OpenAI APIs

### Learning & Stability (v0.5.0)
- **Strategy Bank**: Agents now learn strategies from successful runs and store them in a persistent database.
- **Semantic Recall**: Agents use vector search to find relevant past runs and strategies before acting.
- **n8n Workflow Agent**: New agent for triggering external webhooks and n8n workflows.
- **Resource Management**: Fixed browser orphan leaks, improved CLI signal handling, and added auto-cleanup for old run logs.
- **GUI Improvements**: Added "Thinking" indicators for reasoning models, fix for Anthropic 404s, and better error visibility.

### CLI Features
- `agentic-browser run --json`: Output machine-readable JSON for pipeline integration
- `agentic-browser memory --show`: View stored sites/directories/recipes
- `agentic-browser memory --bootstrap`: Pre-seed common directories
- `agentic-browser --dry-run`: See planned actions without execution
- `agentic-browser --explain`: Show risk analysis for commands

## 📄 License

MIT License - see [LICENSE](LICENSE)
