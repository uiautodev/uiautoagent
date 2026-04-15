# UIAutoAgent

AI 驱动的 UI 自动化框架，支持视觉定位和自主任务执行。

## 特性

- 🎯 AI 视觉定位元素，无需 DOM
- 🤖 自主规划执行任务
- 🧠 任务记忆学习
- 📱 Android / iOS 设备支持
- 🔧 灵活的模型配置（支持不同场景使用不同模型）
- 📊 可视化 HTML 报告（标注截图、token 消耗、耗时）
- 🔍 启动时自动检查模型可用性

## 安装

```bash
uv sync
cp .env.example .env
# 编辑 .env 配置 API_KEY 和模型
```

## 配置

在 `.env` 文件中配置 OpenAI 兼容的 API：

```bash
# 基础配置
BASE_URL=https://api.openai.com/v1
API_KEY=sk-xxx
MODEL_NAME=doubao-seed-2.0-pro

# 可选：为不同场景配置不同的模型
MODEL_PLAN=                # AI规划模型（需要视觉能力）
MODEL_DETECT=.             # 元素检测模型（需要视觉能力）
MODEL_TEXT=.               # 文本处理模型（总结、澄清等）

# 代理配置（可选）
MODEL_PROXY=http://127.0.0.1:7890

# 请求超时时间（秒）
REQUEST_TIMEOUT=60

# OpenRouter 请求追踪（可选）
OPENROUTER_SITE_URL=https://yoursite.com
OPENROUTER_SITE_NAME=YourAppName
SESSION_ID=my-session-123   # 默认自动生成 UUID
```

推荐的配置

```
# 方案1
MODEL_NAME=doubao-seed-2.0-pro

# 方案2 (Openrouter)
BASE_URL=https://openrouter.ai/api/v1
API_KEY=sk-...
# 由于openrouter不让国外访问，需要配置个国外的代理才行
MODEL_PROXY=http://localhost:1080

# 这两个配合价格便宜一些，10步以内的话，费用不到2毛钱
# openai/gpt-5-mini   $0.25/M input tokens $2/M output tokens
# openai/gpt-5.4-mini $0.75/M input tokens $4.50/M output tokens
# z.ai/glm-5v-turbo   $1.20/M input tokens $4/M output tokens
# z.ai/glm-4.6v       $0.30/M input tokens $0.90/M output tokens
MODEL_NAME=openai/gpt-5-mini
MODEL_DETECT=z-ai/glm-4.6v
```


### 场景说明

| 场景 | 环境变量 | 说明 | 模型要求 |
|------|----------|------|----------|
| `PLAN` | `MODEL_PLAN` | AI 规划下一步操作 | 需要视觉能力 |
| `DETECT` | `MODEL_DETECT` | UI 元素检测定位 | 需要视觉能力 |
| `TEXT` | `MODEL_TEXT` | 文本处理（总结、澄清、搜索） | 纯文本，无视觉要求 |

## 快速开始

```bash
# AI 自主执行任务
uv run uiautoagent -m ai -t "修改昵称为 kitty"

# 指定iOS设备
uv run uiautoagent -m ai -t "修改昵称为 kitty" -p ios

# 其他模式
uv run uiautoagent -m find    # 查找并点击
uv run uiautoagent -m manual  # 手动控制
```

启动时会自动检查所有配置模型的可用性：

```
🔍 检查模型可用性（共 2 个）...
  ✅ 'gpt-4o' [default, plan]
  ✅ 'gpt-4o-mini' [detect, text]
```

## 任务报告

每次任务执行完成后，会在 `tasks/task_xxx/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `report.html` | 可视化 HTML 报告，包含标注截图、AI 原始响应、token 消耗、耗时 |
| `history.json` | 完整步骤记录（含 token 统计） |
| `log.jsonl` | 实时追加的步骤日志（每步执行后立即写入） |
| `summary.txt` | 文本摘要 |
| `screenshots/` | 原始截图 |
| `annotated/` | 标注了操作位置和 bbox 的截图 |

## Python API

### AI 自主执行任务

```python
from uiautoagent import run_ai_task

# 最简单的用法 - AI 自主完成任务
result = run_ai_task("修改昵称为 kitty")
if result.success:
    print(f"任务完成: {result.result}")
else:
    print(f"任务失败: {result.result}")

# 如果任务需要返回观察结果（如"查看有多少个好友"）
result = run_ai_task("查看有多少个好友")
if result.success:
    print(f"好友数量: {result.result}")  # 例如: "有5个好友"
```

### 元素检测

```python
from uiautoagent import detect_element, draw_bbox

# 检测元素
result = detect_element("screenshot.png", "登录按钮")
if result.found:
    print(f"位置: {result.bbox}")
    draw_bbox("screenshot.png", result, "result.png")
```

### 设备控制

```python
from uiautoagent import AndroidController, IOSController, SwipeDirection

# 控制Android设备
controller = AndroidController()
controller.tap(500, 1000)
controller.swipe_direction(SwipeDirection.UP)
controller.input_text("hello")
controller.back()

# 控制iOS设备
controller = IOSController()  # 自动检测USB设备
controller.tap(500, 1000)
controller.swipe_direction(SwipeDirection.UP)
controller.input_text("hello")
controller.home()
```

### 直接调用 AI

```python
from uiautoagent import Category, chat_completion

response = chat_completion(
    category=Category.TEXT,
    messages=[{"role": "user", "content": "总结这段文本"}],
    max_tokens=500,
)
content = response.choices[0].message.content

# 规划场景（需要图片）
plan_response = chat_completion(
    category=Category.PLAN,
    messages=[{"role": "user", "content": "分析这张图片"}],
)
```

### Token 统计

```python
from uiautoagent import TokenTracker

stats = TokenTracker.get_stats()
for category, stat in stats.items():
    print(f"{category}: {stat.total} tokens")

total = TokenTracker.get_total()
input_cost, output_cost, total_cost = TokenTracker.calculate_cost(
    total.prompt, total.completion
)
print(f"费用: ¥{total_cost:.4f}")
```

AI 视觉定位可以精准识别屏幕上的 UI 元素：

**原始截图**
![sample.png](assets/sample.png)

**检测结果** - 查询"登录按钮"
![result.png](assets/result.png)

## 要求

- Python 3.10+
- OpenAI 兼容的 API
  - 视觉场景（`PLAN`、`DETECT`）需要支持 Vision 的模型
  - 文本场景（`TEXT`）使用普通聊天模型即可
- Android 需要 ADB
- iOS 需要 WebDriverAgent 和 [wdapy](https://github.com/openatx/wdapy)，设备列表需要 `idevice_id`（libimobiledevice）或 `tidevice`

## License

[LICENSE](LICENSE)
