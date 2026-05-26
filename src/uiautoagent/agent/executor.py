"""AI 任务执行器 - 自主决策并执行任务"""

from __future__ import annotations

import time
from pathlib import Path

import dictlog

from pydantic import BaseModel, ConfigDict

from uiautoagent import Category, chat_completion
from uiautoagent.agent import AgentConfig, DeviceAgent
from uiautoagent.agent.ai_utils import compress_markdown, summarize_task
from uiautoagent.agent.memory import TaskMemory, get_task_memory
from uiautoagent.agent.plan import (
    Action,
    ActionType,
    DoneParams,
    HistoryEntry,
    TaskProposal,
    TapParams,
    get_action_examples_prompt,
    parse_plan_response,
)
from uiautoagent.controller import AndroidController, IOSController
from uiautoagent.types import TokenUsage

# 模块级 logger
log = dictlog.get_logger(__name__)


def _setup_android_device(serial: str | None) -> tuple:
    """设置Android设备，返回 (controller, serial) 或 (None, None)"""
    devices = AndroidController.list_devices()
    if not devices:
        log.error("未检测到Android设备")
        return None, None

    if serial:
        if serial not in devices:
            log.error("设备未找到", serial=serial)
            return None, None
        device_serial = serial
    else:
        device_serial = devices[0]

    return AndroidController(device_serial), device_serial


def _setup_ios_device(udid: str | None) -> tuple:
    """设置iOS设备，返回 (controller, udid) 或 (None, None)"""
    if udid:
        return IOSController(udid=udid), udid

    devices = IOSController.list_devices()
    if not devices:
        log.error("未检测到iOS设备")
        return None, None

    return IOSController(udid=devices[0]), devices[0]


class TaskResult(BaseModel):
    """AI任务执行结果"""

    model_config = ConfigDict(use_enum_values=True)

    success: bool  # 任务是否成功完成
    result: str | None = None  # 任务执行结果（如"有5个好友"），失败时为错误信息


_TAP_EXAMPLE = Action(
    type=ActionType.TAP,
    thought="需要点击搜索按钮进入搜索页面",
    log="点击搜索按钮",
    params=TapParams(target="搜索按钮", bbox=[120, 80, 350, 140]),
).model_dump_json(exclude_none=True)


def get_system_prompt() -> str:
    """获取系统提示词"""
    examples = get_action_examples_prompt()
    return f"""你是一个手机操作专家。根据任务和截图，输出下一步操作的JSON。

## 可用操作
{examples}

## 坐标规则（必须遵守）
- tap/long_press: 必须输出bbox [x1,y1,x2,y2]（1000x1000归一化坐标系，左上角+右下角）
- swipe位置模式: 必须输出swipe_start_xy [x,y] + swipe_end_xy [x,y]（中心点坐标）
- 坐标必须紧贴目标元素边界，不要随意估算

## 输出格式
只输出JSON，不要任何额外文本。字段只包含当前操作类型所需的，不要有空值。

## 示例

```json
{_TAP_EXAMPLE}
```

## 决策原则
- 参考历史任务中的成功经验，但不要被失败步骤干扰
- 优先用app_launch启动目标应用，确保从正确界面开始
- 找不到元素时尝试滑动或返回，不要重复相同操作
- input前必须先tap输入框
- 任务完成用done，需要返回结果时设置return_result:true
- 界面无变化说明操作未生效，换一种方式重试
"""


def build_history_summary(history: list[HistoryEntry]) -> str:
    """构建历史摘要字符串"""
    if not history:
        return "（这是第一步，无历史记录）"
    lines = []
    for h in history:
        action = h.action
        step_lines = [f"- step_number: {h.step_number}"]
        if action.log:
            step_lines.append(f"  log: {action.log}")
        if action.thought:
            step_lines.append(f"  thought: {action.thought}")
        step_lines.append(f"  result: {'成功' if h.success else '失败'}")
        if h.image_similarity is not None:
            sim = h.image_similarity
            if sim > 0.95:
                step_lines.append("  similarity: 界面几乎无变化")
            elif sim > 0.85:
                step_lines.append("  similarity: 界面轻微变化")
            elif sim > 0.7:
                step_lines.append("  similarity: 界面明显变化")
            else:
                step_lines.append("  similarity: 界面大幅变化")
        lines.append("\n".join(step_lines))
    return "\n\n".join(lines)


def build_user_prompt_with_memory(
    task: str, context: dict, memory_reference: str, user_context: str | None = None
) -> str:
    """构建用户消息（包含历史任务参考和任务上下文）"""
    from datetime import datetime

    history_summary = build_history_summary(context["history"])
    device_info = context["device_info"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    context_section = ""
    if user_context:
        context_section = f"""## 任务上下文
以下是用户提供的关于当前任务的上下文信息，请优先参考：
{user_context}

"""

    return f"""任务：{task}

设备信息：{device_info["model"]} ({device_info["width"]}x{device_info["height"]})
当前时间：{current_time}

{context_section}{memory_reference}

## 当前任务执行历史
{history_summary}

## 当前屏幕
请分析截图，任务是「{task}」，输出下一步操作的JSON。tap/long_press必须包含bbox坐标，swipe位置模式必须包含swipe_start_xy/swipe_end_xy坐标。
"""


def encode_screenshot(screenshot_path: str | Path) -> str:
    """编码截图为base64"""
    import base64

    with open(screenshot_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_ai_action(system_prompt: str, user_prompt: str, screenshot_b64: str) -> Action:
    """调用AI获取下一步动作

    Returns:
        Action AI规划的动作
    """
    response = chat_completion(
        category=Category.VISION,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=2048,
        temperature=0.0,
    )

    plan_text = response.choices[0].message.content
    if not plan_text:
        raise ValueError("AI返回空响应")

    log.debug("AI Response", plan_preview=plan_text[:200])

    return parse_plan_response(plan_text)


def handle_task_status(
    action: Action,
    agent: DeviceAgent,
    task: str,
    task_memory: TaskMemory,
    original_task: str | None = None,
) -> TaskResult | None:
    """
    处理任务状态并保存记忆

    Returns:
        TaskResult表示任务应该结束，None表示继续执行
    """
    if action.type == ActionType.DONE:
        log.info("任务完成")

        result = None
        # 如果需要返回结果
        assert isinstance(action.params, DoneParams)
        if action.params.return_result and action.params.result:
            result = action.params.result
            log.info(
                "任务结果", result=result, screenshot=agent.get_current_screenshot()
            )

        # 先保存任务记忆（会调用summarize，产生token）
        summary = summarize_task(
            task, agent.history, success=True, original_task=original_task
        )
        task_memory.save_task(
            task,
            agent.history,
            success=True,
            summary=summary,
            original_task=original_task,
        )
        log.info("已保存任务记忆")

        # 然后保存历史和打印统计（包含summarize的token）
        agent.save_history()
        agent.print_summary()

        return TaskResult(success=True, result=result)

    if action.type == ActionType.FAIL:
        log.error("AI认为任务无法完成", thought=action.thought)

        # 先保存任务记忆（会调用summarize，产生token）
        summary = summarize_task(
            task, agent.history, success=False, original_task=original_task
        )
        task_memory.save_task(
            task,
            agent.history,
            success=False,
            summary=summary,
            original_task=original_task,
        )

        # 然后保存历史和打印统计（包含summarize的token）
        agent.save_history()
        agent.print_summary()

        return TaskResult(success=False, result=action.thought)

    return None


def handle_ai_error(agent: DeviceAgent, error: Exception):
    """处理AI决策错误（非JSON解析错误时执行返回兜底）"""
    log.error("AI决策出错", error=str(error), error_type=type(error).__name__)
    step = agent.step(
        Action(
            type=ActionType.BACK,
            thought="AI决策出错，尝试返回",
        )
    )
    agent._append_step_log(step)


def execute_ai_task(
    agent: DeviceAgent, proposal: TaskProposal, user_context: str | None = None
) -> TaskResult:
    """
    使用AI自主执行任务，支持任务记忆复用

    Args:
        agent: 设备Agent
        proposal: 任务提案（包含原始任务和澄清后的任务）
        user_context: 用户提供的任务上下文

    Returns:
        TaskResult: 任务执行结果
    """
    max_steps = agent.config.max_steps

    # 获取任务记忆
    task_memory = get_task_memory()
    similar_tasks = task_memory.find_similar_tasks(proposal.clarified_task)

    if similar_tasks:
        log.info(
            f"找到 {len(similar_tasks)} 个相似历史任务，将作为参考",
            similar_tasks_count=len(similar_tasks),
            tasks=[
                {"index": i, "success": t["success"], "task": t["task"]}
                for i, t in enumerate(similar_tasks, 1)
            ],
        )
    else:
        log.info("未找到相似历史任务")

    # 缓存系统提示词
    system_prompt = get_system_prompt()

    for step in range(max_steps):
        log.debug("AI决策中", step=step + 1, max_steps=max_steps)

        # 准备数据
        screenshot_path = agent.get_current_screenshot()
        ai_context = agent.get_context_for_ai()
        screenshot_b64 = encode_screenshot(screenshot_path)

        # 构建用户消息（包含历史任务参考和任务上下文）
        memory_reference = task_memory.format_for_ai(similar_tasks)
        user_prompt = build_user_prompt_with_memory(
            proposal.clarified_task,
            ai_context,
            memory_reference,
            user_context=user_context,
        )
        user_prompt = compress_markdown(user_prompt)

        # 调用AI决策
        try:
            from uiautoagent.ai import TokenTracker

            tokens_before = TokenTracker.get_total()
            step_start = time.time()

            action = get_ai_action(system_prompt, user_prompt, screenshot_b64)

            # 执行动作（复用已截好的图，避免重复截图）
            task_step = agent.step(
                action, screenshot_path=screenshot_path, step_start=step_start
            )

            # 计算本步总 token 消耗（action + detect 等所有 AI 调用）
            tokens_after = TokenTracker.get_total()
            task_step.ai_tokens = TokenUsage(
                prompt=tokens_after.prompt - tokens_before.prompt,
                completion=tokens_after.completion - tokens_before.completion,
                total=tokens_after.total - tokens_before.total,
            )
            task_step.ai_response = action.model_dump_json(exclude_none=True)
            task_step.ai_system_prompt = system_prompt
            task_step.ai_user_prompt = user_prompt

            # 所有字段已填充，写入实时日志
            agent._append_step_log(task_step)

            # 检查任务状态
            result = handle_task_status(
                action,
                agent,
                proposal.clarified_task,
                task_memory,
                original_task=proposal.original_task,
            )
            if result is not None:
                return result

        except ValueError as e:
            # AI返回格式错误且修复失败，跳过本步重试
            log.warning("AI返回格式错误，跳过本步重试", error=str(e))

        except Exception as e:
            handle_ai_error(agent, e)

    # 达到最大步数
    log.warning(
        f"达到最大步数限制 ({max_steps})，任务可能未完成",
        max_steps=max_steps,
        actual_steps=agent.history,
    )
    agent.save_history()
    agent.print_summary()

    # 保存未完成任务记忆（含summary）
    summary = summarize_task(
        proposal.clarified_task,
        agent.history,
        success=False,
        original_task=proposal.original_task,
    )
    task_memory.save_task(
        proposal.clarified_task,
        agent.history,
        success=False,
        summary=summary,
        original_task=proposal.original_task,
    )

    return TaskResult(success=False, result=f"达到最大步数限制 ({max_steps})")


def run_ai_task(
    task: str,
    serial: str | None = None,
    max_steps: int = 30,
    verbose: bool = True,
    platform: str = "android",
    context: str | None = None,
) -> TaskResult:
    """
    运行 AI 自主任务 - 便捷函数

    这是主要的对外接口函数，用于执行 AI 自主任务。

    Args:
        task: 任务描述
        serial: 设备序列号/UDID，None 表示使用第一个可用设备
        max_steps: 最大执行步数
        verbose: 是否打印详细日志
        platform: 设备平台，"android" 或 "ios"
        context: 用户提供的任务上下文，帮助AI更好地理解任务

    Returns:
        TaskResult: 任务执行结果，包含 success 和 result 字段

    Example:
        >>> from uiautoagent.agent import run_ai_task
        >>> result = run_ai_task("查看有多少个好友")
        >>> if result.success:
        ...     log.info("任务完成", result=result.result)
    """
    log.info("=" * 50)
    log.info("📱 设备Agent - AI自主决策模式")
    log.info("=" * 50)

    platform = platform.lower()
    if platform == "ios":
        controller, device_id = _setup_ios_device(serial)
    else:
        controller, device_id = _setup_android_device(serial)

    if controller is None:
        return TaskResult(success=False, result=f"未检测到{platform}设备")

    agent = DeviceAgent(
        controller,
        config=AgentConfig(
            max_steps=max_steps,
            save_screenshots=True,
            verbose=verbose,
        ),
        task=task,
    )

    info = controller.get_device_info()
    log.info(
        "设备信息",
        model=info["model"],
        width=info["width"],
        height=info["height"],
        report_dir=agent.report_dir,
    )

    log.info("任务", task=task, has_context=context is not None)
    if context:
        log.info(
            "任务上下文",
            context=context[:100] + "..." if len(context) > 100 else context,
        )

    # 用AI澄清任务描述
    from uiautoagent.agent.ai_utils import clarify_task

    original_task = task  # 保存原始输入

    # 先检查历史任务，如果有完全匹配的 original_task，直接复用其 clarified_task
    task_memory = get_task_memory()
    history_match = task_memory.find_by_original_task(original_task)
    if history_match:
        task = history_match["task"]
        log.info("历史任务匹配，复用已澄清的任务", task=task)
    else:
        task = clarify_task(task)

    # 创建任务提案
    proposal = TaskProposal(original_task=original_task, clarified_task=task)
    agent.proposal = proposal

    log.info("AI将自主分析屏幕并决策每一步操作")

    # 执行AI自主任务
    try:
        return execute_ai_task(agent, proposal, user_context=context)
    except Exception as e:
        log.error("任务执行出错", error=str(e), error_type=type(e).__name__)
        return TaskResult(success=False, result=str(e))
