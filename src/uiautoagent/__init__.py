"""uiautoagent - AI-powered UI automation framework"""

import dotenv

# 加载 .env 文件环境变量
dotenv.load_dotenv()

from uiautoagent.ai import (  # noqa: E402
    Category,
    TokenStats,
    TokenTracker,
    chat_completion,
    get_ai_config,
    get_ai_model,
)
from uiautoagent.agent import (  # noqa: E402
    Action,
    ActionType,
    AgentConfig,
    DeviceAgent,
    TaskStep,
)
from uiautoagent.agent.ai_utils import (  # noqa: E402
    clarify_task,
    summarize_task,
)
from uiautoagent.agent.executor import execute_ai_task, run_ai_task  # noqa: E402
from uiautoagent.agent.memory import TaskMemory, get_task_memory  # noqa: E402
from uiautoagent.controller import (  # noqa: E402
    AndroidController,
    DeviceController,
    IOSController,
    SwipeDirection,
)
from uiautoagent.detector import BBox, DetectionResult, draw_bbox, detect_element  # noqa: E402

__all__ = [
    # AI client
    "Category",
    "chat_completion",
    "get_ai_model",
    "get_ai_config",
    "TokenTracker",
    "TokenStats",
    # Agent
    "DeviceAgent",
    "Action",
    "ActionType",
    "AgentConfig",
    "TaskStep",
    "run_ai_task",
    "execute_ai_task",
    # Memory
    "TaskMemory",
    "get_task_memory",
    "summarize_task",
    "clarify_task",
    # Controller
    "DeviceController",
    "AndroidController",
    "IOSController",
    "SwipeDirection",
    # Detector
    "BBox",
    "DetectionResult",
    "detect_element",
    "draw_bbox",
]

__version__ = "0.1.0"
