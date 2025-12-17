# Domain Routing

This document describes the intelligent routing system that automatically directs user requests to the appropriate domain: **Browser** or **OS**.

## Routing Modes

Configure via GUI settings or `routing_mode` in settings:

| Mode | Behavior |
|------|----------|
| **Auto** (default) | Automatic routing via heuristics + LLM fallback |
| **Browser** | Force all requests to browser domain |
| **OS** | Force all requests to OS domain |
| **Ask** | Prompt user for every request |

## How Auto-Routing Works

### Step 1: Heuristic Pass (Fast)

The router uses keyword matching with weighted scores:

**Browser signals (+score)**
- URL patterns: `https://`, `www.`, `.com/.org/.net`
- Keywords: "website", "browser", "click", "login", "search", "scroll", "navigate"

**OS signals (+score)**
- File paths: `/path/to/file`, `~/folder`, `./file`
- Keywords: "file", "folder", "terminal", "command", "install", "chmod", "process"

If confidence ≥ 0.75, use heuristic result immediately.

### Step 2: LLM Router Pass (For Ambiguous)

If confidence < 0.75, query the LLM with a classification prompt:

```json
{
  "domain": "browser|os|both",
  "confidence": 0.85,
  "reason": "User wants to list local files"
}
```

### Step 3: Execute with Selected Domain

- **Browser** → BrowserTools (Playwright)
- **OS** → OSTools (subprocess)
- **Both** → Orchestrator can call either

## DomainDecision Object

Every routing produces a decision:

```python
@dataclass
class DomainDecision:
    domain: str      # "browser", "os", or "both"
    confidence: float  # 0.0 - 1.0
    reason: str        # Explanation
    forced_by_user: bool  # Manual override?
```

## Examples

| Request | Routed To | Reason |
|---------|-----------|--------|
| "Search Google for Python tutorials" | Browser | URL keyword "Google" |
| "List files in ~/projects" | OS | File path detected |
| "Open https://github.com" | Browser | URL pattern |
| "Install numpy with pip" | OS | "install", "pip" keywords |
| "Download the file from the website" | Browser (60%) | Mixed signals, browser stronger |

## Action Classification

Actions are statically classified:

**Browser Actions**
```
goto, click, type, press, scroll, wait_for,
extract, extract_visible_text, screenshot,
back, forward, done
```

**OS Actions**
```
os_exec, os_list_dir, os_read_file, os_write_file
```

## Configuration

In `~/.agentic_browser/settings.json`:

```json
{
  "routing_mode": "auto",
  "os_provider": "lm_studio",
  "os_model": null,
  "os_api_key": null
}
```

## Testing

```bash
pytest tests/test_domain_router.py -v
```

Tests cover:
- Manual mode overrides
- Browser keyword detection
- OS keyword detection
- URL/path pattern matching
- Mixed signal handling
- Action classification
