"""AI 任务执行器 - 自主决策并执行任务"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from uiautoagent import Category, chat_completion
from uiautoagent.agent import AgentConfig, DeviceAgent, Action, ActionType
from uiautoagent.agent.ai_utils import summarize_task
from uiautoagent.agent.memory import TaskMemory, get_task_memory
from uiautoagent.controller import AndroidController, IOSController
from uiautoagent.detector.bbox_detector import safe_validate_json
from uiautoagent.types import TokenUsage


class PlanResponse(BaseModel):
    """AI 规划响应结构"""

    type: str
    thought: str = ""
    target: str | None = None
    text: str | None = None
    direction: str | None = None
    swipe_start: str | None = None
    swipe_end: str | None = None
    wait_ms: int = Field(default=1000, ge=0)
    return_result: bool = False
    result: str | None = None


def _setup_android_device(serial: str | None) -> tuple:
    """设置Android设备，返回 (controller, serial) 或 (None, None)"""
    devices = AndroidController.list_devices()
    if not devices:
        print("❌ 未检测到Android设备")
        return None, None

    if serial:
        if serial not in devices:
            print(f"❌ 设备 {serial} 未找到")
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
        print("❌ 未检测到iOS设备")
        return None, None

    return IOSController(udid=devices[0]), devices[0]


class TaskResult(BaseModel):
    """AI任务执行结果"""

    success: bool  # 任务是否成功完成
    result: str | None = None  # 任务执行结果（如"有5个好友"），失败时为错误信息

    class Config:
        use_enum_values = True


def get_system_prompt() -> str:
    """获取系统提示词"""
    return """你是一个手机操作专家。用户会给你一个任务和当前手机屏幕截图，你需要分析屏幕并决定下一步操作。

## 利用历史经验
用户会提供相似历史任务的执行步骤，请参考这些成功经验：
- 优先尝试历史任务中成功的操作模式
- 如果历史任务显示某个元素描述有效，使用相同的描述
- 注意历史任务中的关键步骤顺序

可用操作类型：
1. tap - 点击屏幕上的元素（需要指定target描述元素，如"搜索按钮"）
2. input - 输入文本（需要指定text内容）
3. swipe - 滑动屏幕
   - 方式1：指定direction: up/down/left/right （适用于整体的滑动，区域滑动请使用方式2）
   - 方式2：指定swipe_start和swipe_end来描述起始和结束位置（如从"头像图标"滑动到"设置按钮"）
4. back - 返回上一页
5. wait - 等待（需要指定wait_ms毫秒数）
6. done - 任务完成（当任务已完成时）
7. fail - 任务失败（当无法继续时）

请以JSON格式返回你的决策：
{
  "type": "操作类型",
  "thought": "为什么执行这个操作",
  "target": "目标元素描述（仅tap时需要，其他操作省略此字段）",
  "text": "输入文本（仅input时需要，其他操作省略此字段）",
  "direction": "滑动方向（仅swipe按方向滑动时需要，值为up/down/left/right之一，其他操作省略此字段）",
  "swipe_start": "滑动起始位置描述（仅swipe按位置描述时需要，与swipe_end配合使用）",
  "swipe_end": "滑动结束位置描述（仅swipe按位置描述时需要，与swipe_start配合使用）",
  "wait_ms": 等待毫秒数（仅wait时需要，默认1000，其他操作省略此字段）",
  "return_result": true（仅done时需要，表示任务需要返回观察结果）
  "result": "任务返回的结果或答案（仅done时需要）
}

重要：
- 只包含你使用的操作类型所需的字段，不要包含空字符串或null值
- 例如：tap操作只需要type、thought、target三个字段
- swipe操作可以选择direction或swipe_start+swipe_end，不要同时提供
- 当任务需要返回观察结果时，done操作必须包含return_result和result字段

注意：
- 优先参考历史任务的成功步骤
- 分析屏幕时要仔细，确保能找到目标元素
- 如果找不到元素，可以尝试滑动或返回
- 任务完成后使用done
- 无法继续时使用fail
- input类型操作前需要先tap对应的输入框
- 如果任务要求返回信息（如"查看好友发了什么消息"），done时必须设置return_result:true并在result中描述结果

**避免重复失败：**
- 如果同样的操作（如点击某个元素、向某个方向滑动）连续失败超过2次，必须立即更换思路
- 例如：如果点击"设置按钮"3次都失败，尝试：1)换种描述（如"齿轮图标"）；2)先滑动页面再找；3)考虑从其他入口进入
- 重复同样的无效操作是浪费步数，观察失败原因后必须调整策略"""


def build_history_summary(history: list) -> str:
    """构建历史摘要字符串"""
    if not history:
        return "（这是第一步，无历史记录）"

    lines = []
    for h in history:
        status = "✅" if h["success"] else "❌"
        action = h["action"]

        # 构建动作详情
        parts = [f"类型: {action['type']}"]
        if action.get("thought"):
            parts.append(f"思考: {action['thought']}")
        if action.get("target"):
            parts.append(f"目标: {action['target']}")
        if action.get("text"):
            parts.append(f"输入: {action['text']}")
        if action.get("direction"):
            parts.append(f"方向: {action['direction']}")
        if action.get("swipe_start") and action.get("swipe_end"):
            parts.append(f"滑动: {action['swipe_start']} → {action['swipe_end']}")
        if action.get("wait_ms"):
            parts.append(f"等待: {action['wait_ms']}ms")

        details = ", ".join(parts)
        obs = f" → {h['observation']}" if h.get("observation") else ""
        lines.append(f"[步骤{h['step']}] {status} {details}{obs}")

    return "\n".join(lines)


def build_user_prompt_with_memory(
    task: str, context: dict, memory_reference: str
) -> str:
    """构建用户消息（包含历史任务参考）"""
    from datetime import datetime

    history_summary = build_history_summary(context["history"])
    device_info = context["device_info"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""任务：{task}

设备信息：{device_info["model"]} ({device_info["width"]}x{device_info["height"]})
当前时间：{current_time}

{memory_reference}

## 当前任务执行历史
{history_summary}

## 当前屏幕
请参考上方相似历史任务的经验，分析当前屏幕并决定下一步操作："""


def encode_screenshot(screenshot_path: str | Path) -> str:
    """编码截图为base64"""
    import base64

    with open(screenshot_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_ai_plan(
    system_prompt: str, user_prompt: str, screenshot_b64: str
) -> PlanResponse:
    """调用AI规划API

    Returns:
        PlanResponse 规划结果
    """
    response = chat_completion(
        category=Category.PLAN,
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
        max_tokens=1024,
        temperature=0.0,
    )

    plan_text = response.choices[0].message.content
    if not plan_text:
        raise ValueError("AI返回空响应")

    print(f"[AI思考] {plan_text[:200]}...")

    return safe_validate_json(plan_text, PlanResponse)


def parse_action_from_plan(plan: PlanResponse) -> Action:
    """从AI规划解析出Action对象"""
    action_type = ActionType(plan.type if plan.type else "fail")

    kwargs: dict = {
        "type": action_type,
        "thought": plan.thought or "",
    }

    if plan.target:
        kwargs["target"] = plan.target
    if plan.text:
        kwargs["text"] = plan.text
    if plan.direction and plan.direction in ("up", "down", "left", "right"):
        kwargs["direction"] = plan.direction
    if plan.swipe_start:
        kwargs["swipe_start"] = plan.swipe_start
    if plan.swipe_end:
        kwargs["swipe_end"] = plan.swipe_end
    if plan.wait_ms:
        kwargs["wait_ms"] = plan.wait_ms
    if plan.return_result:
        kwargs["return_result"] = True
    if plan.result:
        kwargs["result"] = plan.result

    return Action(**kwargs)


def handle_task_status(
    action: Action, agent: DeviceAgent, task: str, task_memory: TaskMemory
) -> TaskResult | None:
    """
    处理任务状态并保存记忆

    Returns:
        TaskResult表示任务应该结束，None表示继续执行
    """
    if action.type == ActionType.DONE:
        print("\n🎉 任务完成！")

        result = None
        # 如果需要返回结果
        if action.return_result and action.result:
            result = action.result
            print("\n📋 任务结果:")
            print(f"   {result}")
            print(f"\n📸 当前截图: {agent.get_current_screenshot()}")

        # 先保存任务记忆（会调用summarize，产生token）
        summary = summarize_task(task, agent.history, success=True)
        task_memory.save_task(task, agent.history, success=True, summary=summary)
        print("💾 已保存任务记忆")

        # 然后保存历史和打印统计（包含summarize的token）
        agent.save_history()
        agent.print_summary()

        return TaskResult(success=True, result=result)

    if action.type == ActionType.FAIL:
        print(f"\n❌ AI认为任务无法完成: {action.thought}")

        # 先保存任务记忆（会调用summarize，产生token）
        summary = summarize_task(task, agent.history, success=False)
        task_memory.save_task(task, agent.history, success=False, summary=summary)

        # 然后保存历史和打印统计（包含summarize的token）
        agent.save_history()
        agent.print_summary()

        return TaskResult(success=False, result=action.thought)

    return None


def handle_ai_error(agent: DeviceAgent, error: Exception):
    """处理AI决策错误（非JSON解析错误时执行返回兜底）"""
    print(f"❌ AI决策出错: {error}")
    step = agent.step(
        Action(
            type=ActionType.BACK,
            thought="AI决策出错，尝试返回",
        )
    )
    agent._append_step_log(step)


def execute_ai_task(agent: DeviceAgent, task: str) -> TaskResult:
    """
    使用AI自主执行任务，支持任务记忆复用

    Args:
        agent: 设备Agent
        task: 任务描述

    Returns:
        TaskResult: 任务执行结果
    """
    max_steps = agent.config.max_steps

    # 获取任务记忆
    task_memory = get_task_memory()
    similar_tasks = task_memory.find_similar_tasks(task)

    if similar_tasks:
        print(f"💡 找到 {len(similar_tasks)} 个相似历史任务，将作为参考:")
        for i, task_mem in enumerate(similar_tasks, 1):
            status = "✅" if task_mem["success"] else "❌"
            print(f"   {i}. {status} {task_mem['task']}")
    else:
        print("💡 未找到相似历史任务")

    # 缓存系统提示词
    system_prompt = get_system_prompt()

    for step in range(max_steps):
        print(f"\n{'=' * 50}")
        print(f"🤖 AI决策 - 步骤 {step + 1}/{max_steps}")
        print(f"{'=' * 50}")

        # 准备数据
        screenshot_path = agent.get_current_screenshot()
        context = agent.get_context_for_ai()
        screenshot_b64 = encode_screenshot(screenshot_path)

        # 构建用户消息（包含历史任务参考）
        memory_reference = task_memory.format_for_ai(similar_tasks)
        user_prompt = build_user_prompt_with_memory(task, context, memory_reference)

        # 调用AI决策
        try:
            from uiautoagent.ai import TokenTracker

            tokens_before = TokenTracker.get_total()

            plan = call_ai_plan(system_prompt, user_prompt, screenshot_b64)
            action = parse_action_from_plan(plan)

            # 执行动作（复用已截好的图，避免重复截图）
            task_step = agent.step(action, screenshot_path=screenshot_path)

            # 计算本步总 token 消耗（plan + detect 等所有 AI 调用）
            tokens_after = TokenTracker.get_total()
            task_step.ai_tokens = TokenUsage(
                prompt=tokens_after.prompt - tokens_before.prompt,
                completion=tokens_after.completion - tokens_before.completion,
                total=tokens_after.total - tokens_before.total,
            )
            task_step.ai_response = plan.model_dump_json(exclude_none=True)
            task_step.ai_system_prompt = system_prompt
            task_step.ai_user_prompt = user_prompt

            # 所有字段已填充，写入实时日志
            agent._append_step_log(task_step)

            # 检查任务状态
            result = handle_task_status(action, agent, task, task_memory)
            if result is not None:
                return result

        except ValueError as e:
            # AI返回格式错误且修复失败，跳过本步重试
            print(f"⚠️  AI返回格式错误，跳过本步重试: {e}")

        except Exception as e:
            handle_ai_error(agent, e)

    # 达到最大步数
    print(f"\n⚠️  达到最大步数限制 ({max_steps})，任务可能未完成")
    agent.save_history()
    agent.print_summary()

    # 保存未完成任务记忆
    task_memory.save_task(task, agent.history, success=False)

    return TaskResult(success=False, result=f"达到最大步数限制 ({max_steps})")


def run_ai_task(
    task: str,
    serial: str | None = None,
    max_steps: int = 30,
    verbose: bool = True,
    platform: str = "android",
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

    Returns:
        TaskResult: 任务执行结果，包含 success 和 result 字段

    Example:
        >>> from uiautoagent.agent import run_ai_task
        >>> result = run_ai_task("查看有多少个好友")
        >>> if result.success:
        ...     print(f"任务完成: {result.result}")
    """

    print("=" * 50)
    print("📱 设备Agent - AI自主决策模式")
    print("=" * 50)

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
    )

    info = controller.get_device_info()
    print(f"📋 设备信息: {info['model']} ({info['width']}x{info['height']})")
    print(f"📁 任务目录: {agent.task_dir}")

    print(f"\n🎯 任务: {task}")

    # 用AI澄清任务描述
    from uiautoagent.agent.ai_utils import clarify_task

    task = clarify_task(task)

    print("🤖 AI将自主分析屏幕并决策每一步操作...\n")

    # 执行AI自主任务
    try:
        return execute_ai_task(agent, task)
    except Exception as e:
        print(f"❌ 任务执行出错: {e}")
        return TaskResult(success=False, result=str(e))
