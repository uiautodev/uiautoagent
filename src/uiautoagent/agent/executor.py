"""AI 任务执行器 - 自主决策并执行任务"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from uiautoagent import Category, chat_completion
from uiautoagent.agent import AgentConfig, DeviceAgent, ActionType
from uiautoagent.agent.ai_utils import summarize_task
from uiautoagent.agent.memory import TaskMemory, get_task_memory
from uiautoagent.agent.plan import (
    Action,
    get_action_examples_prompt,
    parse_plan_response,
)
from uiautoagent.controller import AndroidController, IOSController
from uiautoagent.types import TokenUsage


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

    model_config = ConfigDict(use_enum_values=True)

    success: bool  # 任务是否成功完成
    result: str | None = None  # 任务执行结果（如"有5个好友"），失败时为错误信息


def get_system_prompt() -> str:
    """获取系统提示词"""
    examples = get_action_examples_prompt()
    return f"""你是一个手机操作专家。用户会给你一个任务和当前手机屏幕截图，你需要分析屏幕并决定下一步操作。

## 利用历史经验
<historical_tasks>标签内的内容是相似历史任务的执行步骤参考，请参考这些成功经验：
- 优先尝试历史任务中成功的操作模式
- 如果历史任务显示某个元素描述有效，使用相同的描述
- 注意历史任务中的关键步骤顺序
- 标签外的内容是当前任务，不要被历史任务干扰

## 可用操作类型及示例

{examples}

## 常用包名（参考）

- 微信：Android `com.tencent.mm`，iOS `com.tencent.xin`
- QQ：Android `com.tencent.mobileqq`，iOS `com.tencent.mqq`
- 抖音：Android `com.ss.android.ugc.aweme`，iOS `com.ss.iphone.ugc.Aweme`
- 小红书：Android `com.xingin.xhs`，iOS `com.xingin.discover`
- 支付宝：Android `com.eg.android.AlipayGphone`，iOS `com.alipay.iphoneclient`
- 淘宝：Android `com.taobao.taobao`，iOS `com.taobao.taobao4iphone`
- 哔哩哔哩：Android `tv.danmaku.bili`，iOS `com.bilibili.app`

## 重要说明

**字段使用规则：**
- 只包含你使用的操作类型所需的字段，不要包含空字符串或null值
- 每种操作类型只需要必需的字段，参考上面的示例
- swipe操作可以选择direction或swipe_start+swipe_end，不要同时提供
- 当任务需要返回观察结果时，done操作必须包含return_result和result字段

**注意事项：**
- 优先参考历史任务的成功步骤
- 分析屏幕时要仔细，确保能找到目标元素
- 如果任务需要操作特定应用，优先使用app_launch启动该应用，确保从正确的界面开始
- 如果找不到元素，可以尝试滑动或返回
- 任务完成后使用done
- 无法继续时使用fail
- input类型操作前需要先tap对应的输入框
- 如果任务要求返回信息（如"查看好友发了什么消息"），done时必须设置return_result:true并在result中描述结果

**界面相似度参考：**
- 历史步骤中会标注界面变化情况（[界面几乎无变化]、[界面明显变化]等）
- 如果点击/滑动后界面几乎无变化，说明操作可能未生效，需要：
  - 检查是否点到了正确的位置（尝试更精确的描述）
  - 考虑增加等待时间（界面响应可能有延迟）
  - 尝试其他操作方式（如用swipe_direction代替点击特定位置）
- 如果界面有明显变化，说明操作生效，可以继续下一步

**避免重复失败：**
- 如果同样的操作（如点击某个元素、向某个方向滑动）一直失败，必须立即更换思路
- 重复同样的无效操作是浪费步数，观察失败原因后必须调整策略

**输出要求**
根据示列中的格式，输出你认为最合适的下一步操作。只需要输出JSON，不要任何额外文本。
"""


def build_history_summary(history: list) -> str:
    """构建历史摘要字符串"""
    if not history:
        return "（这是第一步，无历史记录）"

    lines = []
    for h in history:
        status = "✅" if h["success"] else "❌"
        action = h["action"]

        # 构建动作详情
        parts = []
        if action.get("log"):
            parts.append(f"操作: {action['log']}")
        elif action.get("thought"):
            parts.append(f"思考: {action['thought']}")

        details = ", ".join(parts)

        # 添加相似度信息（如果有）
        similarity_info = ""
        if h.get("image_similarity") is not None:
            sim = h["image_similarity"]
            if sim > 0.95:
                similarity_info = " [界面几乎无变化，操作可能未生效]"
            elif sim > 0.85:
                similarity_info = " [界面轻微变化]"
            elif sim > 0.7:
                similarity_info = " [界面明显变化]"
            else:
                similarity_info = " [界面大幅变化]"

        lines.append(f"- [步骤{h['step']}] {status} {details}{similarity_info}")

    return "\n".join(lines)


def build_user_prompt_with_memory(
    task: str, context: dict, memory_reference: str, knowledge: str | None = None
) -> str:
    """构建用户消息（包含历史任务参考和背景知识）"""
    from datetime import datetime

    history_summary = build_history_summary(context["history"])
    device_info = context["device_info"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    knowledge_section = ""
    if knowledge:
        knowledge_section = f"""## 背景知识
以下是用户提供的关于当前任务的背景知识，请优先参考：
{knowledge}

"""

    return f"""任务：{task}

设备信息：{device_info["model"]} ({device_info["width"]}x{device_info["height"]})
当前时间：{current_time}

{knowledge_section}{memory_reference}

## 当前任务执行历史
{history_summary}

## 当前屏幕
任务: {task}
请参考上方相似历史任务的经验，分析当前屏幕并决定下一步操作：
JSON输出
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
        max_tokens=2048,
        temperature=0.0,
    )

    plan_text = response.choices[0].message.content
    if not plan_text:
        raise ValueError("AI返回空响应")

    print(f"[AI思考] {plan_text[:200]}...")

    return parse_plan_response(plan_text)


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
        return_result = getattr(action.params, "return_result", False)
        result_value = getattr(action.params, "result", None)
        if return_result and result_value:
            result = result_value
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


def execute_ai_task(
    agent: DeviceAgent, task: str, knowledge: str | None = None
) -> TaskResult:
    """
    使用AI自主执行任务，支持任务记忆复用

    Args:
        agent: 设备Agent
        task: 任务描述
        knowledge: 用户提供的背景知识

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

        # 构建用户消息（包含历史任务参考和背景知识）
        memory_reference = task_memory.format_for_ai(similar_tasks)
        user_prompt = build_user_prompt_with_memory(
            task, context, memory_reference, knowledge=knowledge
        )

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
    knowledge: str | None = None,
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
        knowledge: 用户提供的背景知识，帮助AI更好地理解任务

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
        task=task,
    )

    info = controller.get_device_info()
    print(f"📋 设备信息: {info['model']} ({info['width']}x{info['height']})")
    print(f"📁 任务目录: {agent.task_dir}")

    print(f"\n🎯 任务: {task}")
    if knowledge:
        print(f"📖 背景知识: {knowledge[:100]}{'...' if len(knowledge) > 100 else ''}")

    # 用AI澄清任务描述
    from uiautoagent.agent.ai_utils import clarify_task

    task = clarify_task(task)

    print("🤖 AI将自主分析屏幕并决策每一步操作...\n")

    # 执行AI自主任务
    try:
        return execute_ai_task(agent, task, knowledge=knowledge)
    except Exception as e:
        print(f"❌ 任务执行出错: {e}")
        return TaskResult(success=False, result=str(e))
