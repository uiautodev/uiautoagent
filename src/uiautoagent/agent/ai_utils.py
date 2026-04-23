"""AI 辅助函数 - 任务总结、澄清等"""

from __future__ import annotations

from uiautoagent.agent.device_agent import ActionType


def summarize_task(
    task: str, history: list, success: bool, original_task: str | None = None
) -> str:
    """
    记录任务执行结果，生成Markdown格式的执行日志

    Args:
        task: 任务描述（澄清后）
        history: 执行历史
        success: 是否成功
        original_task: 用户原始输入的任务描述

    Returns:
        Markdown格式的任务执行记录
    """
    lines = [
        f"# {'成功' if success else '失败'}",
        f"**任务**: {task}",
    ]
    if original_task and original_task != task:
        lines.append(f"**原始输入**: {original_task}")
    lines.append(f"**步数**: {len(history)}")

    # 获取最终结果（DONE 的 result 或 FAIL 的 thought）
    if history:
        last_action = history[-1].action
        if success and last_action.type == ActionType.DONE:
            from uiautoagent.agent.plan import DoneParams

            assert isinstance(last_action.params, DoneParams)
            if last_action.params.result:
                lines.append(f"**结果**: {last_action.params.result}")
        elif not success and last_action.type == ActionType.FAIL:
            if last_action.thought:
                lines.append(f"**原因**: {last_action.thought}")

    lines.append("\n## 执行日志")

    # 收集每步的 log
    for h in history:
        status = "✅" if h.success else "❌"
        if h.action.log:
            lines.append(f"{status} {h.action.log}")
        else:
            lines.append(f"{status} {h.action.type}")

    return "\n\n".join(lines)


def clarify_task(task: str) -> str:
    """
    使用AI将用户输入的任务描述重新表述为清晰、无歧义的操作指令。

    Args:
        task: 用户原始任务描述（可能存在语法或语义问题）

    Returns:
        经过AI重新表述的清晰任务描述
    """
    try:
        from uiautoagent.ai import Category, chat_completion

        response = chat_completion(
            category=Category.TEXT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个手机自动化任务解析专家。"
                        "用户会给你一段手机操作任务描述，可能存在语法错误、表达不清或语义歧义。"
                        "请将其重新表述为一句清晰、完整、无歧义的操作指令。"
                        "只返回重新表述后的任务描述，不要添加任何解释。"
                    ),
                },
                {"role": "user", "content": task},
            ],
            max_tokens=256,
            temperature=0.0,
        )

        clarified = (response.choices[0].message.content or "").strip()

        if clarified and clarified != task:
            print(f"✏️  任务已澄清: {clarified!r}")
            return clarified
        return task

    except Exception as e:
        print(f"⚠️  任务澄清失败，使用原始描述: {e}")
        return task
