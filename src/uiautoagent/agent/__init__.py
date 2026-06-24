"""AI agent module for autonomous device automation."""

from uiautoagent.agent.ai_utils import clarify_task, summarize_task
from uiautoagent.agent.plan import (
    Action,
    ActionType,
    TaskProposal,
)
from uiautoagent.agent.device_agent import (
    ActionDetail,
    AgentConfig,
    DeviceAgent,
    RecordingController,
    TaskStep,
)
from uiautoagent.agent.executor import (
    TaskResult,
    execute_ai_task,
    replay_ai_task,
    run_ai_task,
)
from uiautoagent.agent.memory import TaskMemory, get_task_memory
from uiautoagent.agent.replay import (
    Recording,
    ReplayConfig,
    ReplayResult,
    ReplayStepResult,
    list_recordings,
    load_recording,
    replay_task,
    save_recording,
)

__all__ = [
    # Core agent
    "DeviceAgent",
    "Action",
    "ActionDetail",
    "ActionType",
    "AgentConfig",
    "RecordingController",
    "TaskStep",
    "TaskProposal",
    # Memory
    "TaskMemory",
    "get_task_memory",
    # AI utils
    "summarize_task",
    "clarify_task",
    # Executor
    "TaskResult",
    "execute_ai_task",
    "run_ai_task",
    "replay_ai_task",
    # Replay
    "Recording",
    "ReplayConfig",
    "ReplayResult",
    "ReplayStepResult",
    "save_recording",
    "load_recording",
    "list_recordings",
    "replay_task",
]
