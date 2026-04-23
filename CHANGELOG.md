# Changelog

## [Unreleased]

### Changed

- 将 `knowledge` 参数统一重命名为 `context`，与 CLI 参数 `--context` 保持一致
  - `run_ai_task()` 参数 `knowledge` → `context`
  - `execute_ai_task()` 参数 `knowledge` → `user_context`
  - `build_user_prompt_with_memory()` 参数 `knowledge` → `user_context`
  - 中文显示名从"背景知识"改为"任务上下文"
  - CLI `--context-file` (`-cf`) 和 `--context` (`-c`) 参数保持不变
  - README.md 中的示例和说明已同步更新
