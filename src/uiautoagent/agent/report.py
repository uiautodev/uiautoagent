"""HTML报告生成器 - 在截图上标注操作位置并生成可视化报告"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from uiautoagent.agent.device_agent import ActionDetail, TaskStep


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int = 20,
    color: str = "red",
    width: int = 3,
):
    """在图片上绘制十字准心"""
    # 十字线
    draw.line([(x - size, y), (x + size, y)], fill=color, width=width)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=width)
    # 圆圈
    r = size // 2
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=width)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "green",
    width: int = 4,
):
    """在图片上绘制带箭头的线"""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # 箭头
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_len = 20
    arrow_angle = math.pi / 6
    ax1 = x2 - arrow_len * math.cos(angle - arrow_angle)
    ay1 = y2 - arrow_len * math.sin(angle - arrow_angle)
    ax2 = x2 - arrow_len * math.cos(angle + arrow_angle)
    ay2 = y2 - arrow_len * math.sin(angle + arrow_angle)
    draw.polygon([(x2, y2), (int(ax1), int(ay1)), (int(ax2), int(ay2))], fill=color)


def annotate_screenshot(
    screenshot_path: Path,
    detail: ActionDetail,
    output_path: Path,
) -> Path:
    """在截图上标注操作位置并保存"""
    img = Image.open(screenshot_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if detail.tap_bbox:
        x1, y1, x2, y2 = detail.tap_bbox
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)

    if detail.tap_position:
        x, y = detail.tap_position
        _draw_crosshair(
            draw, x, y, size=max(img.width, img.height) // 30, color="red", width=4
        )
        draw.text((x + 15, y - 25), f"TAP ({x}, {y})", fill="red")

    if detail.swipe_start and detail.swipe_end:
        x1, y1 = detail.swipe_start
        x2, y2 = detail.swipe_end
        _draw_arrow(draw, x1, y1, x2, y2, color="green", width=4)
        draw.text((x1 + 10, y1 - 25), f"({x1}, {y1})", fill="green")
        draw.text((x2 + 10, y2 + 5), f"({x2}, {y2})", fill="green")

    if detail.swipe_direction:
        cx, cy = img.width // 2, img.height // 2
        dirs = {
            "up": (cx, cy + 40, cx, cy - 40),
            "down": (cx, cy - 40, cx, cy + 40),
            "left": (cx + 40, cy, cx - 40, cy),
            "right": (cx - 40, cy, cx + 40, cy),
        }
        if detail.swipe_direction in dirs:
            sx1, sy1, sx2, sy2 = dirs[detail.swipe_direction]
            _draw_arrow(draw, sx1, sy1, sx2, sy2, color="green", width=4)
            draw.text(
                (cx - 40, cy - 40), f"SWIPE {detail.swipe_direction}", fill="green"
            )

    img.save(output_path)
    return output_path


def _image_to_base64(path: Path) -> str:
    """将图片转为base64 data URI"""
    suffix = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix.lstrip("."), "image/png"
    )
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _action_icon(action_type: str) -> str:
    icons = {
        "tap": "👆",
        "input": "⌨️",
        "swipe": "👆",
        "back": "⬅️",
        "wait": "⏳",
        "done": "✅",
        "fail": "❌",
    }
    return icons.get(action_type, "❓")


def generate_html_report(
    steps: list[TaskStep],
    task_dir: Path,
) -> Path:
    """生成HTML可视化报告

    Args:
        steps: 任务步骤列表
        task_dir: 任务目录

    Returns:
        报告文件路径
    """
    annotated_dir = task_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    success_count = sum(1 for s in steps if s.success)
    fail_count = len(steps) - success_count

    # 生成标注截图
    annotated_images: dict[int, str] = {}
    for step in steps:
        screenshot_path = Path(step.screenshot_path)
        if not screenshot_path.exists():
            continue
        detail = step.action_detail
        if detail and (
            detail.tap_position
            or detail.swipe_start
            or detail.swipe_direction
            or detail.is_back
        ):
            out_path = annotated_dir / f"step_{step.step_number:03d}.png"
            annotate_screenshot(screenshot_path, detail, out_path)
            annotated_images[step.step_number] = _image_to_base64(out_path)
        else:
            annotated_images[step.step_number] = _image_to_base64(screenshot_path)

    # 构建HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    steps_html = ""
    for idx, step in enumerate(steps):
        icon = _action_icon(step.action.type)
        status_class = "success" if step.success else "fail"
        status_text = "成功" if step.success else "失败"
        thought_html = (
            f'<p class="thought">💭 {html.escape(step.action.thought)}</p>'
            if step.action.thought
            else ""
        )
        detail_html = ""
        if step.action_detail:
            d = step.action_detail
            parts = []
            if d.tap_position:
                parts.append(f"点击坐标: ({d.tap_position[0]}, {d.tap_position[1]})")
            if d.swipe_start and d.swipe_end:
                parts.append(
                    f"滑动: ({d.swipe_start[0]}, {d.swipe_start[1]}) → ({d.swipe_end[0]}, {d.swipe_end[1]})"
                )
            if d.swipe_direction:
                parts.append(f"方向滑动: {d.swipe_direction}")
            if d.is_back:
                parts.append("返回手势")
            if parts:
                detail_html = '<p class="detail">' + " | ".join(parts) + "</p>"

        # AI 信息：token 消耗 + 耗时
        ai_parts = []
        if step.elapsed is not None:
            ai_parts.append(f"⏱ {step.elapsed:.2f}s")
        if step.ai_tokens:
            t = step.ai_tokens
            ai_parts.append(f"🪙 {t.total} tokens (↑{t.prompt} ↓{t.completion})")
        if steps:
            delta = int(step.timestamp - steps[0].timestamp)
            ai_parts.append(f"🕐 {delta // 60:02d}:{delta % 60:02d}")
        ai_html = (
            f'<p class="ai-meta">{" &nbsp;|&nbsp; ".join(ai_parts)}</p>'
            if ai_parts
            else ""
        )

        # AI 详细信息（可折叠）
        ai_detail_parts = []
        if step.ai_response:
            ai_detail_parts.append(("AI 响应", html.escape(step.ai_response)))
        if step.ai_system_prompt:
            ai_detail_parts.append(
                ("System Prompt", html.escape(step.ai_system_prompt))
            )
        if step.ai_user_prompt:
            ai_detail_parts.append(("User Prompt", html.escape(step.ai_user_prompt)))

        ai_response_html = ""
        if ai_detail_parts:
            details_html = ""
            for label, content in ai_detail_parts:
                details_html += f"""<details class="ai-response-nested">
                            <summary>{label}</summary>
                            <pre>{content}</pre>
                        </details>"""
            ai_response_html = f"""<details class="ai-response">
                        <summary>AI 详细信息</summary>
                        {details_html}
                    </details>"""

        # 操作前截图（标注操作位置）
        before_src = annotated_images.get(step.step_number, "")
        before_html = (
            f'<img src="{before_src}" alt="步骤{step.step_number} 操作前" loading="lazy">'
            if before_src
            else ""
        )

        # 操作后截图（使用下一步的操作前截图）
        after_src = ""
        if idx + 1 < len(steps):
            next_step = steps[idx + 1]
            next_path = Path(next_step.screenshot_path)
            if next_path.exists():
                after_src = _image_to_base64(next_path)
        after_html = (
            f'<img src="{after_src}" alt="步骤{step.step_number} 操作后" loading="lazy">'
            if after_src
            else ""
        )

        # 截图区域：左边操作前（标注），右边操作后（结果）
        screenshots_html = f"""
                <div class="screenshots">
                    <div class="screenshot-item">
                        <div class="screenshot-label">操作前</div>
                        {before_html}
                    </div>
                    <div class="screenshot-item">
                        <div class="screenshot-label">操作后</div>
                        {after_html}
                    </div>
                </div>"""

        steps_html += f"""
        <div class="step-card {status_class}">
            <div class="step-header">
                <span class="step-num">步骤 {step.step_number}</span>
                <span class="status-badge {status_class}">{status_text}</span>
                <span class="action-type">{icon} {html.escape(step.action.type)}</span>
            </div>
            <div class="step-body">
                {screenshots_html}
                <div class="step-info">
                    {thought_html}
                    <p class="observation">👁️ {html.escape(step.observation)}</p>
                    {detail_html}
                    {ai_html}
                    {ai_response_html}
                </div>
            </div>
        </div>"""

    report_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>任务执行报告 - {now}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ text-align: center; margin-bottom: 10px; font-size: 24px; }}
    .summary {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
    .summary-item {{ background: white; border-radius: 8px; padding: 15px 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
    .summary-item .num {{ font-size: 28px; font-weight: bold; }}
    .summary-item .label {{ font-size: 13px; color: #666; }}
    .step-card {{ background: white; border-radius: 8px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }}
    .step-card.success {{ border-left: 4px solid #4caf50; }}
    .step-card.fail {{ border-left: 4px solid #f44336; }}
    .step-header {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px; background: #fafafa; border-bottom: 1px solid #eee; }}
    .step-num {{ font-weight: bold; font-size: 14px; }}
    .status-badge {{ padding: 2px 10px; border-radius: 12px; font-size: 12px; color: white; }}
    .status-badge.success {{ background: #4caf50; }}
    .status-badge.fail {{ background: #f44336; }}
    .action-type {{ font-size: 14px; }}
    .step-body {{ display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; }}
    .screenshots {{ display: flex; gap: 12px; flex: 0 0 auto; }}
    .screenshot-item {{ flex: 0 0 280px; }}
    .screenshot-item img {{ width: 100%; border-radius: 4px; border: 1px solid #eee; }}
    .screenshot-label {{ font-size: 12px; color: #999; text-align: center; margin-bottom: 4px; }}
    .step-info {{ flex: 1; font-size: 14px; line-height: 1.8; min-width: 200px; }}
    .thought {{ color: #666; font-style: italic; }}
    .observation {{ color: #333; }}
    .detail {{ color: #1976d2; font-family: monospace; font-size: 13px; }}
    .ai-meta {{ color: #888; font-size: 12px; margin-top: 6px; }}
    .ai-response {{ margin-top: 8px; }}
    .ai-response summary {{ font-size: 12px; color: #999; cursor: pointer; }}
    .ai-response-nested {{ margin: 4px 0; }}
    .ai-response-nested summary {{ font-size: 12px; color: #666; cursor: pointer; }}
    .ai-response pre {{ margin-top: 4px; padding: 8px; background: #f8f8f8; border-radius: 4px; font-size: 12px; white-space: pre-wrap; word-break: break-all; max-height: 400px; overflow-y: auto; }}
    @media (max-width: 768px) {{
        .step-body {{ flex-direction: column; }}
        .screenshots {{ flex-direction: column; }}
        .screenshot-item {{ flex: none; width: 100%; }}
    }}
</style>
</head>
<body>
<div class="container">
    <h1>任务执行报告</h1>
    <p style="text-align:center; color:#999; margin-bottom:20px;">{now}</p>
    <div class="summary">
        <div class="summary-item"><div class="num">{len(steps)}</div><div class="label">总步骤</div></div>
        <div class="summary-item"><div class="num" style="color:#4caf50">{success_count}</div><div class="label">成功</div></div>
        <div class="summary-item"><div class="num" style="color:#f44336">{fail_count}</div><div class="label">失败</div></div>
    </div>
    {steps_html}
</div>
</body>
</html>"""

    report_path = task_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path
