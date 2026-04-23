# CLAUDE.md

## Project Overview

UIAutoAgent - AI 驱动的移动端 UI 自动化框架。通过视觉 AI 模型识别屏幕元素，自主规划并执行任务。

核心流程：截图 → AI 规划动作(JSON) → 执行动作 → 循环直到完成。

## Tech Stack

- Python 3.10+, uv 包管理
- Pydantic 数据模型（优先使用，不用 dict）
- OpenAI 兼容 API（视觉模型 + 文本模型）
- Android: ADB; iOS: WebDriverAgent + wdapy

## Development Commands

```bash
uv sync                    # 安装依赖
uv run ruff check .        # lint 检查
uv run pytest              # 运行测试
uv run pytest tests/test_xxx.py -v  # 单个测试文件
uv run uiautoagent -m ai -t "任务描述"  # 运行 AI 任务
```

## Project Structure

```
src/uiautoagent/
├── __init__.py          # 公开 API，加载 .env
├── ai.py                # OpenAI client, chat_completion(), TokenTracker, Category 枚举
├── types.py             # 共享 Pydantic 类型 (TokenUsage)
├── cli/main.py          # CLI 入口: -m ai|find|manual
├── controller/          # 设备控制抽象层
│   ├── base.py          # DeviceController ABC
│   ├── android.py       # AndroidController (ADB)
│   └── ios.py           # IOSController (wdapy)
├── detector/            # AI 视觉元素检测
│   └── bbox_detector.py # detect_element(), 1000x1000 归一化坐标
└── agent/               # 核心编排层
    ├── plan.py          # Action/ActionType 模型, parse_plan_response()
    ├── device_agent.py  # DeviceAgent: 步骤执行, 截图管理, 录制
    ├── executor.py      # run_ai_task() 主循环, 系统提示词
    ├── memory.py        # TaskMemory: YAML 持久化任务记忆
    ├── ai_utils.py      # clarify_task(), summarize_task()
    ├── report.py        # HTML 可视化报告生成
    └── image_similarity.py  # 截图相似度计算
```

## Key Architecture Concepts

- **归一化坐标系**: AI 在 1000x1000 虚拟空间输出坐标，DeviceAgent 转换为实际像素
- **截图复用**: 截图有 1 秒 TTL，避免决策和执行间重复截图
- **RecordingController**: 装饰器模式，记录操作坐标用于报告可视化
- **任务记忆**: YAML 持久化，线程安全，支持 original_task 精确匹配复用
- **AI JSON 修复**: detector 中 safe_validate_json() 用 AI 修复格式错误的响应

## Coding Conventions

- 修改代码后必须 `ruff check` 检查
- 使用 `pytest` 运行测试
- 涉及 API 变更时同步更新 README.md
- 优先使用 Pydantic 模型，不用 dict
- 测试中使用 DummyController 模拟设备，不需要真实设备
- Pre-commit hooks: ruff-format, detect-secrets, detect-private-key

## Model Configuration

通过 `.env` 配置，三个场景可使用不同模型：
- `PLAN`: AI 规划（需要视觉能力）
- `DETECT`: 元素检测（需要视觉能力）
- `TEXT`: 文本处理（总结、澄清，纯文本即可）

## Public API

```python
from uiautoagent import run_ai_task, detect_element, draw_bbox
from uiautoagent import AndroidController, IOSController, SwipeDirection
from uiautoagent import Category, chat_completion, TokenTracker
```
