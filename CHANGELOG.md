# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- **Provider support**: Implemented real adapters for Anthropic Messages API and Google GenAI API (Gemini) in addition to OpenAI/OpenAI-compatible endpoints. Each provider now uses its native API format.

- **Context persistence**: `additional_context` is now stored on the agent instance and persists across loop iterations until success or run ends. Recovery strategy context is now reliably injected into subsequent prompts.

- **Profile directory lifecycle**: Fixed `profile_dir` property generating a new temp directory on every access when `no_persist=True`. Now caches the directory once per config instance and properly cleans up at run end.

- **Browser lifecycle cleanup**: Browser is now always closed regardless of `no_persist` flag, preventing zombie Chromium processes. Proper ordering: page → context → browser → temp profile cleanup.

- **GUI approvals**: Created unified `Approver` abstraction with `ConsoleApprover` (CLI), `AutoApprover` (subprocess/testing), and `QtApprover` (GUI). GUI mode now uses auto-approve in subprocess mode to prevent stdin deadlocks.

- **Step numbering**: Fixed step numbering mismatch between printed output and logged steps. `print_step()` now takes explicit step number instead of using internal counter.

- **Extraction hardening**: `extract_visible_text()` now includes retry/backoff logic for transient Playwright timing errors. On permanent failure, returns empty string with warning instead of crashing the run.

- **Recovery action**: `get_recovery_action()` is now called on tool failures, with the LLM's recovery strategy merged into context. Added failed action tracking to prevent infinite retry loops.

### Added

- New `adapters.py` module with `LLMAdapter` protocol and provider implementations
- New `approver.py` module with unified approval abstraction
- Comprehensive test suites for bug fixes and provider adapters
- `cleanup_profile_dir()` method on `AgentConfig` for explicit cleanup

### Changed

- `LLMClient` now accepts optional `ProviderConfig` for native adapter usage
- `RunLogger.print_step()` now requires explicit step number parameter
- Browser cleanup is now unconditional in agent finalization
