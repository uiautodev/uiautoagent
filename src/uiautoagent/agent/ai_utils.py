"""AI 辅助函数 - 任务总结、澄清等"""

from __future__ import annotations

import re

from uiautoagent.agent.device_agent import ActionType


def summarize_task(task: str, history: list, success: bool) -> str:
    """
    使用AI总结任务执行结果，生成Markdown格式的经验总结

    Args:
        task: 任务描述
        history: 执行历史
        success: 是否成功

    Returns:
        Markdown格式的任务总结
    """
    try:
        from uiautoagent.ai import Category, chat_completion

        # 构建操作历史摘要
        steps_summary = []
        for h in history:
            status = "✅" if h.success else "❌"
            action = h.action
            if action.type == ActionType.TAP:
                target = getattr(action.params, "target", None)
                if target:
                    steps_summary.append(f"{status} 点击: {target}")
            elif action.type == ActionType.LONG_PRESS:
                target = getattr(action.params, "target", None)
                if target:
                    steps_summary.append(f"{status} 长按: {target}")
            elif action.type == ActionType.INPUT:
                text = getattr(action.params, "text", None)
                if text:
                    steps_summary.append(f"{status} 输入: {text}")
            elif action.type == ActionType.SWIPE:
                direction = getattr(action.params, "direction", None)
                swipe_start = getattr(action.params, "swipe_start", None)
                swipe_end = getattr(action.params, "swipe_end", None)
                if direction:
                    steps_summary.append(f"{status} 滑动: {direction}")
                elif swipe_start and swipe_end:
                    steps_summary.append(f"{status} 滑动: {swipe_start} → {swipe_end}")
                else:
                    steps_summary.append(f"{status} 滑动")
            elif action.type == ActionType.BACK:
                steps_summary.append(f"{status} 返回")
            elif action.type == ActionType.WAIT:
                steps_summary.append(f"{status} 等待")
            elif action.type == ActionType.APP_LAUNCH:
                app_id = getattr(action.params, "app_id", None)
                if app_id:
                    steps_summary.append(f"{status} 启动应用: {app_id}")
            elif action.type == ActionType.APP_STOP:
                app_id = getattr(action.params, "app_id", None)
                if app_id:
                    steps_summary.append(f"{status} 停止应用: {app_id}")
            elif action.type == ActionType.APP_REBOOT:
                app_id = getattr(action.params, "app_id", None)
                if app_id:
                    steps_summary.append(f"{status} 重启应用: {app_id}")

        steps_text = "\n".join(steps_summary)

        prompt = f"""请分析以下任务执行历史，生成经验总结（Markdown格式）。

任务: {task}
结果: {"成功" if success else "失败"}

执行步骤:
{steps_text}

请以Markdown格式返回经验总结，包含成功做法和错误尝试。

成功任务示例：
```markdown
成功完成，共6步

# 正确操作步骤

1. 点击 设置
2. 点击 个人资料
3. 点击 昵称
4. 输入 kitty
5. 滑动 down
6. 点击 保存

# 错误尝试（请避免）

- 点击"修改资料"按钮无效（此按钮无法修改昵称，应该点击"昵称"）
- 向左滑动没找到目标（页面需要向下滑动才能看到昵称选项）
```

失败任务示例：
```markdown
任务失败，共尝试5步

# 执行过的操作

1. 点击 设置
2. 点击 个人资料
3. 点击 账号设置（无效，此按钮不存在）
4. 向下滑动（没找到目标）

# 失败原因

- 找不到"账号设置"入口，可能需要先完成其他步骤
- 向下滑动后仍未发现目标选项
```

要求：
1. 按照示例格式生成Markdown
2. 只返回Markdown内容，不要用代码块包裹
3. 紧凑格式，列表项之间不要空行"""

        response = chat_completion(
            category=Category.TEXT,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个任务总结专家，擅长分析操作历史并提取关键信息。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("AI返回空响应")

        return compress_markdown(content)

    except Exception as e:
        print(f"⚠️  AI总结失败，不保存: {e}")
        return ""


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
