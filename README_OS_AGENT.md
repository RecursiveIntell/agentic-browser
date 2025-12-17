# OS Agent Extension

This document describes the OS control capabilities added to the Agentic Browser, enabling the agent to perform local Linux system operations alongside browser automation.

## Overview

The agent can now execute **four OS actions** for local machine control:

| Action | Description | Risk Level |
|--------|-------------|------------|
| `os_exec` | Execute shell commands | LOW-HIGH (varies) |
| `os_list_dir` | List directory contents | LOW |
| `os_read_file` | Read file contents | LOW-MEDIUM |
| `os_write_file` | Write/append to files | MEDIUM-HIGH |

## Action Arguments

### os_exec
```json
{
  "action": "os_exec",
  "args": {
    "cmd": "ls -la",
    "timeout_s": 30,
    "cwd": "/home/user/projects"
  }
}
```

- **cmd** (required): Command string (parsed with `shlex.split`, NO shell=True)
- **timeout_s** (optional): Timeout in seconds (default: 30, max: 120)
- **cwd** (optional): Working directory (default: sandbox)

### os_list_dir
```json
{
  "action": "os_list_dir",
  "args": {
    "path": "/home/user/documents"
  }
}
```

### os_read_file
```json
{
  "action": "os_read_file",
  "args": {
    "path": "/home/user/config.json",
    "max_bytes": 10000
  }
}
```

- **max_bytes** (optional): Truncate output (default: 1MB)

### os_write_file
```json
{
  "action": "os_write_file",
  "args": {
    "path": "/home/user/output.txt",
    "content": "Hello, World!",
    "mode": "overwrite"
  }
}
```

- **mode**: `overwrite` (default) or `append`

## Safety Controls

### Risk Classification

| Risk Level | Triggers | Approval Required |
|------------|----------|-------------------|
| **HIGH** | `rm -rf`, `sudo`, `dd`, `mkfs`, `shutdown`, `/etc` writes | Always + double-confirm |
| **MEDIUM** | `mv`, `cp`, `chmod`, file writes, scripts | Yes (unless auto-approve) |
| **LOW** | `ls`, `cat`, `grep`, reading, listing | No |

### Dangerous Command Patterns (HIGH risk)

```
rm -rf    # Recursive/forced deletion
dd        # Direct disk access
mkfs      # Filesystem creation
sudo      # Privilege escalation
shutdown  # System power operations
chmod -R  # Recursive permission changes
```

### Path Restrictions

By default, file writes are **blocked outside `$HOME`**. System paths are always blocked:

```
/etc /usr /bin /sbin /boot /var /lib /opt /root
```

### Output Limits

- Command output: truncated to 8000 characters
- File reads: truncated to 1MB (configurable)
- Directory listings: capped at 100 entries

## Configuration

Settings are in `~/.agentic_browser/settings.json`:

```json
{
  "routing_mode": "auto",
  "os_provider": "lm_studio",
  "os_model": "qwen2.5:7b",
  "os_api_key": null,
  "os_custom_endpoint": null
}
```

Environment variables:
- `AGENTIC_OS_ENDPOINT`: OS model endpoint
- `AGENTIC_OS_MODEL`: OS model name
- `AGENTIC_OS_API_KEY`: OS model API key

## Files

| File | Purpose |
|------|---------|
| `os_tools.py` | OSTools class with action implementations |
| `domain_router.py` | Routing between browser/OS domains |
| `tool_router.py` | Dispatching actions to correct tools |

## Testing

```bash
pytest tests/test_os_tools.py tests/test_os_safety.py -v
```
