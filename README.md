# UIAutoAgent

AI 驱动的 UI 自动化框架，支持视觉定位和自主任务执行。

## 特性

- 🎯 AI 视觉定位元素，无需 DOM
- 🤖 自主决策执行任务
- 🧠 任务记忆学习
- 📱 Android 设备支持

## 安装

```bash
uv sync
cp .env.example .env
# 编辑 .env 配置 API_KEY
```

## 快速开始

```bash
# AI 自主执行任务
uv run uiautoagent -m ai -t "修改昵称为 kitty"

# 其他模式
uv run uiautoagent -m find    # 查找并点击
uv run uiautoagent -m manual  # 手动控制
```

## Python API

### AI 自主执行任务

```python
from uiautoagent import run_ai_task

# 最简单的用法 - AI 自主完成任务
success = run_ai_task("修改昵称为 kitty")
```

### 元素检测

```python
from uiautoagent import detect_element, draw_bbox

# 检测元素
result = detect_element("screenshot.png", "登录按钮")
if result.found:
    print(f"位置: {result.bbox}")  # BBox(x, y, width, height)
    draw_bbox("screenshot.png", result.bbox, "result.png")
```

### 设备控制

```python
from uiautoagent import AndroidController, SwipeDirection

# 控制设备
controller = AndroidController()
controller.tap(500, 1000)  # 点击坐标
controller.swipe_direction(SwipeDirection.UP)  # 向上滑动
controller.input_text("hello")  # 输入文本
controller.back()  # 返回
```

### Agent 手动控制

```python
from uiautoagent import DeviceAgent, Action, ActionType, AgentConfig

agent = DeviceAgent(
    AndroidController(),
    config=AgentConfig(max_steps=20, save_screenshots=True)
)

# 执行动作
agent.step(Action(type=ActionType.TAP, thought="点击登录", target="登录按钮"))
agent.step(Action(ActionType.WAIT, wait_ms=2000))
agent.step(Action(type=ActionType.INPUT, text="username"))
```

### 任务记忆

```python
from uiautoagent import get_task_memory

memory = get_task_memory()
similar = memory.find_similar_tasks("修改昵称")
for task in similar:
    print(f"{task['task']} - {'成功' if task['success'] else '失败'}")
```

## 检测效果示例

AI 视觉定位可以精准识别屏幕上的 UI 元素：

**原始截图**
![sample.png](assets/sample.png)

**检测结果** - 查询"登录按钮"
![result.png](assets/result.png)

检测到元素位置：`BBox(x=540, y=1320, width=240, height=120)`

## 要求

- Python 3.10+
- 支持 Vision 的模型（已测试：doubao-seed-2.0-pro）
- 兼容 OpenAI API 格式
- Android 需要 ADB

## License

[LICENSE](LICENSE)
