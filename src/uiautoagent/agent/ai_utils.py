"""AI 辅助函数 - 任务总结、澄清等"""

from __future__ import annotations
import re

from uiautoagent.agent.device_agent import ActionType, TaskStep


def summarize_task(
    task: str, history: list[TaskStep], success: bool, original_task: str | None = None
) -> str:
    """
    记录任务执行结果，生成纯文本格式的执行日志

    Args:
        task: 任务描述（澄清后）
        history: 执行历史
        success: 是否成功
        original_task: 用户原始输入的任务描述

    Returns:
        纯文本格式的任务执行记录
    """
    lines = []
    steps = []
    for h in history[:-1]:  # 最后一步通常是 DONE 或 FAIL，不计入步骤列表
        step_status = "成功" if h.success else "失败"
        action_desc = h.action.log or h.action.type
        steps.append(f"- {action_desc} [{step_status}]")
    last_action = history[-1].action
    if success and last_action.type == ActionType.DONE:
        from uiautoagent.agent.plan import DoneParams

        assert isinstance(last_action.params, DoneParams)
        if last_action.params.result:
            lines.append(f"结果: {last_action.params.result}")
    elif not success and last_action.type == ActionType.FAIL:
        if last_action.thought:
            lines.append(f"失败原因: {last_action.thought}")
    lines.append("步骤:\n" + "\n".join(steps))
    return "\n".join(lines)


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


def compress_markdown(md: str) -> str:
    """压缩 Markdown 内容，移除多余空行和代码块标记"""
    # 移除代码块标记
    if md.startswith("```"):
        md = re.sub(r"^```[\w]*\n", "", md)  # 移除开头的```和可选的语言标记
        md = re.sub(r"\n```\s*$", "", md)  # 移除结尾的```

    # 多个空行 -> 一个
    md = re.sub(r"\n{3,}", "\n\n", md)

    # 列表之间的空行去掉
    md = re.sub(r"^(\s*[-*+] .+)\n+(?=\s*[-*+] )", r"\1\n", md, flags=re.MULTILINE)

    # 数字列表
    md = re.sub(r"^(\s*\d+\. .+)\n+(?=\s*\d+\. )", r"\1\n", md, flags=re.MULTILINE)

    return md.strip()
