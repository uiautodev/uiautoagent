# UIAutoAgent

[中文文档](README_CN.md)

AI-driven mobile UI automation framework with visual AI element detection and autonomous task execution.

## Features

- AI-powered visual element detection, no DOM required
- Autonomous task planning and execution
- Task memory with learning capabilities
- Android / iOS device support
- Flexible model configuration (different models per scenario with fallback chain)
- Visual HTML reports (annotated screenshots, token usage, timing)
- AI image content extraction (structured JSON output)
- Startup model availability check for all candidates
- Before/after screenshot comparison (AI judges whether an action took effect)

## Installation

```bash
uv sync
cp .env.example .env
# Edit .env to configure API_KEY and model
```

## Configuration

Configure an OpenAI-compatible API in `.env`:

```bash
# Core config (UIAUTO_ prefix recommended; legacy variable names still supported)
UIAUTO_BASE_URL=--openai-compatable--
UIAUTO_API_KEY=sk-xxx
UIAUTO_MODEL_NAME=doubao-seed-2.0-pro,glm-4.6v  # Default model candidates, tried in order

# Optional: different models per scenario
UIAUTO_MODEL_VISION=doubao-seed-2.0-pro  # Vision model candidates (planning + detection)
UIAUTO_MODEL_TEXT=gpt-4o-mini,deepseek-chat          # Text model candidates (summarization, etc.)

# Proxy (optional)
UIAUTO_MODEL_PROXY=http://127.0.0.1:7890

# Request timeout in seconds
UIAUTO_REQUEST_TIMEOUT=60

# Report output directory (optional)
UIAUTO_REPORT_DIR=/path/to/reports   # Reports written directly here; otherwise defaults to uiautoagent_reports/task_xxx/

# OpenRouter request tracking (optional)
OPENROUTER_SITE_URL=https://yoursite.com
OPENROUTER_SITE_NAME=YourAppName
SESSION_ID=my-session-123   # Auto-generated UUID if not set
```

> **Note**: Environment variables have been upgraded with a `UIAUTO_` prefix to avoid naming conflicts. Legacy variable names (e.g. `BASE_URL`, `API_KEY`) are still supported, but the new prefixed versions are recommended.
>
> Model env vars support comma-separated candidate lists; order defines the fallback sequence. When a call fails, the next candidate is tried automatically.

### All Environment Variables

| Variable | Legacy Fallback | Default | Description |
|----------|----------------|---------|-------------|
| `UIAUTO_BASE_URL` | `BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `UIAUTO_API_KEY` | `API_KEY` | — | API key |
| `UIAUTO_MODEL_NAME` | `MODEL_NAME` | `doubao-seed-2.0-pro` | Default model candidates (comma-separated, tried in order) |
| `UIAUTO_MODEL_VISION` | `MODEL_VISION` | same as `MODEL_NAME` | Vision model candidates (planning + detection, requires vision capability) |
| `UIAUTO_MODEL_TEXT` | `MODEL_TEXT` | same as `MODEL_NAME` | Text model candidates (summarization, clarification, search) |
| `UIAUTO_MODEL_PROXY` | `MODEL_PROXY` | — | HTTP proxy (e.g. `http://127.0.0.1:7890`) |
| `UIAUTO_REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | `60` | Request timeout in seconds |
| `UIAUTO_REPORT_DIR` | — | — | Report output directory. When set, reports are written directly here instead of `task_xxx/` subdirectories |
| `OPENROUTER_SITE_URL` | — | — | OpenRouter site URL (request tracking) |
| `OPENROUTER_SITE_NAME` | — | — | OpenRouter site name (request tracking) |
| `SESSION_ID` | — | auto-generated UUID | Session ID for request tracking |

## Quick Start

```bash
# AI autonomous task execution
uv run uiautoagent -m ai -t "Change nickname to kitty"

# Target an iOS device
uv run uiautoagent -m ai -t "Change nickname to kitty" -p ios

# Provide task context for higher success rate
uv run uiautoagent -m ai -t "Change nickname to kitty" -cf knowledge.txt

# Extract image content (structured JSON)
uv run uiautoagent -m extract -i screenshot.png -q "Extract all product prices"

# Extract with output format hint
uv run uiautoagent -m extract -i screenshot.png -q "Extract product info" --example '{"name":"Product","price":0}'

# Other modes
uv run uiautoagent -m find    # Find and click
uv run uiautoagent -m manual  # Manual control
```

### Task Context

Use `--context-file` (`-cf`) to specify a text file, or `--context` (`-c`) to pass text directly, providing background information to help the AI locate elements and plan actions more accurately.

Example knowledge:
```
Path to change WeChat nickname: tap "Me" at bottom → tap avatar area → tap "Nickname" → edit and tap "Save"
The settings button is in the top-right corner, a gear icon
```

Useful when:
- You know the specific path and want the AI to follow it directly
- The app UI is complex and needs element location hints
- The task requires domain-specific knowledge (e.g. special app behaviors)

All configured model candidates are checked at startup; at least one must be available per scenario:

```
🔍 Checking model availability (4 candidates)...
  ✅ 'glm-4.6v' [default #1]
  ❌ 'doubao-seed-2.0-pro' [vision #1]
  ✅ 'glm-5v-turbo' [vision #2]
  ✅ 'gpt-4o-mini' [text #1]
```

## Task Reports

After each task execution, the following are generated under `uiautoagent_reports/task_xxx/`:

| File | Description |
|------|-------------|
| `report.html` | Visual HTML report with annotated screenshots, raw AI responses, token usage, and timing |
| `history.json` | Full step-by-step record (with token stats) |
| `log.txt` | Real-time step log (appended after each step, human-readable text) |
| `summary.txt` | Text summary |
| `screenshots/` | Original screenshots |
| `annotated/` | Screenshots annotated with tap locations and bounding boxes |

### Screenshot Similarity Feedback

The system compares screenshots before and after actions, computing a similarity score (0–1, 1 = identical), and feeds this back to the AI:

- **Similarity > 95%**: Almost no change; AI may conclude the action had no effect
- **Similarity 85%–95%**: Minor change
- **Similarity 70%–85%**: Notable change; action likely took effect
- **Similarity < 70%**: Major change

This helps the AI judge whether taps, swipes, etc. actually worked, informing its next move.

## Python API

### AI Autonomous Task Execution

```python
from uiautoagent import run_ai_task

# Simplest usage — AI completes the task autonomously
result = run_ai_task("Change nickname to kitty")
if result.success:
    print(f"Task completed: {result.result}")
else:
    print(f"Task failed: {result.result}")

# Provide task context for higher success rate
result = run_ai_task(
    "Change nickname to kitty",
    context="WeChat path: tap 'Me' at bottom → tap avatar → tap 'Nickname' → edit → tap 'Save'",
)

# For observation tasks (e.g. "how many friends do I have")
result = run_ai_task("Check how many friends")
if result.success:
    print(f"Friend count: {result.result}")  # e.g. "5 friends"
```

### Image Content Extraction

```python
from uiautoagent import extract_content, ExtractionResult

# Free-form extraction — AI decides the JSON structure
result = extract_content("screenshot.png", "Extract all pricing info")
if result.success:
    print(result.content)  # dict or list

# Typed extraction — AI outputs in the given JSON format
result = extract_content(
    "screenshot.png",
    query="Extract product info",
    example={"name": "Product", "price": 0},
)
```

### Element Detection

```python
from uiautoagent import detect_element, draw_bbox

# Detect an element
result = detect_element("screenshot.png", "Login button")
if result.found:
    print(f"Position: {result.bbox}")
    draw_bbox("screenshot.png", result, "result.png")
```

### Device Control

```python
from uiautoagent import AndroidController, IOSController, SwipeDirection

# Android device control
controller = AndroidController()
controller.tap(500, 1000)
controller.long_press(500, 1000, duration_ms=1200)
controller.swipe_direction(SwipeDirection.UP)
controller.input_text("hello")
controller.back()
controller.app_launch("com.tencent.mm")  # Launch WeChat
controller.app_stop("com.tencent.mm")    # Stop WeChat
controller.app_reboot("com.tencent.mm")  # Restart WeChat

# iOS device control
controller = IOSController()  # Auto-detects USB device
controller.tap(500, 1000)
controller.long_press(500, 1000, duration_ms=1200)
controller.swipe_direction(SwipeDirection.UP)
controller.input_text("hello")
controller.home()
controller.app_launch("com.tencent.xin")  # Launch WeChat
controller.app_stop("com.tencent.xin")    # Stop WeChat
controller.app_reboot("com.tencent.xin")  # Restart WeChat
```

### Direct AI Calls

```python
from uiautoagent import Category, chat_completion

response = chat_completion(
    category=Category.TEXT,
    messages=[{"role": "user", "content": "Summarize this text"}],
    max_tokens=500,
)
content = response.choices[0].message.content

# When model is not explicitly passed, candidates for the category are tried in order

# Vision scenario (requires image)
vision_response = chat_completion(
    category=Category.VISION,
    messages=[{"role": "user", "content": "Analyze this image"}],
)
```

### Token Statistics

```python
from uiautoagent import TokenTracker

stats = TokenTracker.get_stats()
for category, stat in stats.items():
    print(f"{category}: {stat.total} tokens")

total = TokenTracker.get_total()
print(f"Total: {total.total} tokens")
```

AI-powered visual detection precisely identifies UI elements on screen:

**Original screenshot**
![sample.png](assets/sample.png)

**Detection result** — query "close button"
![result.png](assets/result.png)

## Requirements

- Python 3.10+
- OpenAI-compatible API
  - Vision scenarios (`VISION`) require a vision-capable model
  - Text scenarios (`TEXT`) work with any chat model
- Android requires ADB
- iOS requires WebDriverAgent and [wdapy](https://github.com/openatx/wdapy); device listing requires `idevice_id` (libimobiledevice) or `tidevice`

## Reference

- Google Paper: Repeated Prompters Improve Accuracy https://arxiv.org/pdf/2512.14982

## License

[LICENSE](LICENSE)
