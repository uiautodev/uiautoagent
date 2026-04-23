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
- 🖼️ 操作前后截图对比（AI 可根据界面变化判断操作是否生效）

## 安装

```bash
uv sync
cp .env.example .env
# 编辑 .env 配置 API_KEY 和模型
```

## 配置

在 `.env` 文件中配置 OpenAI 兼容的 API：

```bash
# 基础配置（推荐使用 UIAUTO_ 前缀，旧版变量名仍然支持）
UIAUTO_BASE_URL=--openai-compatable--
UIAUTO_API_KEY=sk-xxx
UIAUTO_MODEL_NAME=--suport-vision-model--

# 可选：为不同场景配置不同的模型
UIAUTO_MODEL_PLAN=         # AI规划模型（需要视觉能力）
UIAUTO_MODEL_DETECT=       # 元素检测模型（需要视觉能力）
UIAUTO_MODEL_TEXT=         # 文本处理模型（总结、澄清等）

# 代理配置（可选）
UIAUTO_MODEL_PROXY=http://127.0.0.1:7890

# 请求超时时间（秒）
UIAUTO_REQUEST_TIMEOUT=60

# OpenRouter 请求追踪（可选）
OPENROUTER_SITE_URL=https://yoursite.com
OPENROUTER_SITE_NAME=YourAppName
SESSION_ID=my-session-123   # 默认自动生成 UUID
```

> **注意**：环境变量已升级为 `UIAUTO_` 前缀以避免命名冲突。旧版变量名（如 `BASE_URL`、`API_KEY` 等）仍然支持，但推荐使用新的前缀版本。

推荐的配置

```sh
# 方案1 (Openrouter)(推荐)
UIAUTO_BASE_URL=https://openrouter.ai/api/v1
UIAUTO_API_KEY=sk-...
# 由于openrouter不让国外访问，需要配置个国外的代理才行
UIAUTO_MODEL_PROXY=http://localhost:1080

# 这两个配合价格便宜一些，10步以内的话，费用不到2毛钱
# openai/gpt-5-mini   $0.25/M input tokens $2/M output tokens
# z.ai/glm-4.6v       $0.30/M input tokens $0.90/M output tokens
UIAUTO_MODEL_NAME=openai/gpt-5-mini
UIAUTO_MODEL_DETECT=z-ai/glm-4.6v
# 下面两个模型稍微贵点
# openai/gpt-5.4-mini $0.75/M input tokens $4.50/M output tokens
# z.ai/glm-5v-turbo   $1.20/M input tokens $4/M output tokens
```

```sh
# 方案2 (Doubao)
UIAUTO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
UIAUTO_API_KEY=...
UIAUTO_MODEL_NAME=doubao-seed-2.0-pro
```


### 场景说明

| 场景 | 环境变量 | 说明 | 模型要求 |
|------|----------|------|----------|
| `PLAN` | `UIAUTO_MODEL_PLAN` | AI 规划下一步操作 | 需要视觉能力 |
| `DETECT` | `UIAUTO_MODEL_DETECT` | UI 元素检测定位 | 需要视觉能力 |
| `TEXT` | `UIAUTO_MODEL_TEXT` | 文本处理（总结、澄清、搜索） | 纯文本，无视觉要求 |

## 快速开始

```bash
# AI 自主执行任务
uv run uiautoagent -m ai -t "修改昵称为 kitty"

# 指定iOS设备
uv run uiautoagent -m ai -t "修改昵称为 kitty" -p ios

# 提供任务上下文提高成功率
uv run uiautoagent -m ai -t "修改昵称为 kitty" -cf knowledge.txt

# 其他模式
uv run uiautoagent -m find    # 查找并点击
uv run uiautoagent -m manual  # 手动控制
```

### 任务上下文

通过 `--context-file` (`-cf`) 指定一个文本文件，或通过 `--context` (`-c`) 直接传入文本，为 AI 提供任务相关的背景信息，帮助 AI 更准确地定位元素和规划操作路径。

知识示例：
```
微信修改昵称路径：点击底部"我" → 点击头像区域 → 点击"昵称" → 修改后点击"保存"
设置按钮在右上角，是一个齿轮图标
```

适用于以下场景：
- 用户知道具体操作路径，希望 AI 直接参考
- 应用 UI 比较复杂，需要提供元素位置提示
- 任务需要特定领域的知识（如某个 App 的特殊操作方式）

启动时会自动检查所有配置模型的可用性：

```
🔍 检查模型可用性（共 2 个）...
  ✅ 'gpt-4o' [default, plan]
  ✅ 'gpt-4o-mini' [detect, text]
```

## 任务报告

每次任务执行完成后，会在 `uiautoagent_tasks/task_xxx/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `report.html` | 可视化 HTML 报告，包含标注截图、AI 原始响应、token 消耗、耗时 |
| `history.json` | 完整步骤记录（含 token 统计） |
| `log.txt` | 实时追加的步骤日志（每步执行后立即写入，可读文本格式） |
| `summary.txt` | 文本摘要 |
| `screenshots/` | 原始截图 |
| `annotated/` | 标注了操作位置和 bbox 的截图 |

### 界面相似度反馈

系统会自动对比操作前后的截图，计算界面相似度（0-1，1 表示完全相同），并将此信息反馈给 AI：

- **相似度 > 95%**：界面几乎无变化，AI 会判断操作可能未生效
- **相似度 85%-95%**：界面轻微变化
- **相似度 70%-85%**：界面明显变化，操作可能已生效
- **相似度 < 70%**：界面大幅变化

这有助于 AI 判断点击/滑动等操作是否真正生效，从而决定下一步策略。

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

# 提供任务上下文提高成功率
result = run_ai_task(
    "修改昵称为 kitty",
    context="微信修改昵称路径：点击底部'我' → 点击头像 → 点击'昵称' → 修改后点'保存'",
)

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
controller.long_press(500, 1000, duration_ms=1200)
controller.swipe_direction(SwipeDirection.UP)
controller.input_text("hello")
controller.back()
controller.app_launch("com.tencent.mm")  # 启动微信
controller.app_stop("com.tencent.mm")    # 停止微信
controller.app_reboot("com.tencent.mm")  # 重启微信

# 控制iOS设备
controller = IOSController()  # 自动检测USB设备
controller.tap(500, 1000)
controller.long_press(500, 1000, duration_ms=1200)
controller.swipe_direction(SwipeDirection.UP)
controller.input_text("hello")
controller.home()
controller.app_launch("com.tencent.xin")  # 启动微信
controller.app_stop("com.tencent.xin")    # 停止微信
controller.app_reboot("com.tencent.xin")  # 重启微信
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
print(f"总计: {total.total} tokens")
```

AI 视觉定位可以精准识别屏幕上的 UI 元素：

**原始截图**
![sample.png](assets/sample.png)

**检测结果** - 查询"关闭按钮"
![result.png](assets/result.png)

## 要求

- Python 3.10+
- OpenAI 兼容的 API
  - 视觉场景（`PLAN`、`DETECT`）需要支持 Vision 的模型
  - 文本场景（`TEXT`）使用普通聊天模型即可
- Android 需要 ADB
- iOS 需要 WebDriverAgent 和 [wdapy](https://github.com/openatx/wdapy)，设备列表需要 `idevice_id`（libimobiledevice）或 `tidevice`

## 参考

- 谷歌Paper，重复提示器提高准确度 https://arxiv.org/pdf/2512.14982

## License

[LICENSE](LICENSE)
