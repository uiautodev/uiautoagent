"""通用设备AI Agent - 支持自主决策和执行任务"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from uiautoagent.controller.base import DeviceController, SwipeDirection
from uiautoagent.types import TokenUsage
from uiautoagent.detector import DetectionResult
from uiautoagent.agent.plan import (
    Action,
    ActionType,
    AppIdParams,
    InputParams,
    LongPressParams,
    SwipeParams,
    TapParams,
    WaitParams,
)


class ActionDetail(BaseModel):
    """操作详情（坐标等可视化信息）"""

    model_config = ConfigDict(use_enum_values=True)

    tap_position: tuple[int, int] | None = None
    tap_bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    swipe_start: tuple[int, int] | None = None
    swipe_end: tuple[int, int] | None = None
    swipe_direction: SwipeDirection | None = None
    is_back: bool = False


class RecordingController(DeviceController):
    """代理模式的Controller包装，记录操作坐标用于可视化报告"""

    def __init__(self, inner: DeviceController):
        self._inner = inner
        self.last_detail: ActionDetail = ActionDetail()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def get_device_info(self) -> dict:
        return self._inner.get_device_info()

    def tap(self, x: int, y: int) -> None:
        self.last_detail = ActionDetail(tap_position=(x, y))
        self._inner.tap(x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.last_detail = ActionDetail(swipe_start=(x1, y1), swipe_end=(x2, y2))
        self._inner.swipe(x1, y1, x2, y2, duration_ms)

    def swipe_direction(
        self, direction: SwipeDirection, ratio: float = 0.5, duration_ms: int = 300
    ) -> None:
        self.last_detail = ActionDetail(swipe_direction=direction)
        self._inner.swipe_direction(direction, ratio, duration_ms)

    def back(self) -> None:
        self.last_detail = ActionDetail(is_back=True)
        self._inner.back()

    def home(self) -> None:
        self._inner.home()

    def input_text(self, text: str) -> None:
        self.last_detail = ActionDetail()
        self._inner.input_text(text)

    def clear_text(self, length: int = 100) -> None:
        self._inner.clear_text(length)

    def press_key(self, keycode: int) -> None:
        self._inner.press_key(keycode)

    def app_launch(self, app_id: str) -> None:
        self._inner.app_launch(app_id)

    def app_stop(self, app_id: str) -> None:
        self._inner.app_stop(app_id)

    def app_reboot(self, app_id: str) -> None:
        self._inner.app_reboot(app_id)

    def long_press(self, x: int, y: int, duration_ms: int = 800) -> None:
        self.last_detail = ActionDetail(tap_position=(x, y))
        self._inner.long_press(x, y, duration_ms)

    def screenshot(self, output_path: str | Path) -> Path:
        return self._inner.screenshot(output_path)

    def tap_bbox(self, bbox) -> None:
        x, y = bbox.center
        self.last_detail = ActionDetail(
            tap_position=(x, y),
            tap_bbox=(bbox.x1, bbox.y1, bbox.x2, bbox.y2),
        )
        self._inner.tap(x, y)

    @staticmethod
    def list_devices() -> list[str]:
        return DeviceController.list_devices()


class TaskStep(BaseModel):
    """任务执行步骤记录"""

    model_config = ConfigDict(use_enum_values=True)

    step_number: int
    screenshot_path: str
    action: Action
    observation: str  # 执行后的观察结果
    action_detail: ActionDetail | None = None  # 操作详情（坐标等）
    success: bool
    timestamp: float
    elapsed: float | None = None  # 执行耗时（秒）
    ai_tokens: TokenUsage | None = None  # AI token 消耗
    ai_response: str | None = None  # AI 原始响应文本
    ai_system_prompt: str | None = None  # AI 系统提示词
    ai_user_prompt: str | None = None  # AI 用户提示词（不含截图）
    screenshot_after_path: str | None = None  # 操作后截图路径
    image_similarity: float | None = None  # 操作前后图片相似度 (0-1)


class AgentConfig(BaseModel):
    """Agent配置"""

    max_steps: int = 20  # 最大执行步数
    tasks_dir: str = "uiautoagent_tasks"  # 任务目录父目录
    save_screenshots: bool = True
    verbose: bool = True


class DeviceAgent:
    """通用设备自动化AI代理（支持Android/iOS等）"""

    def __init__(
        self,
        controller: DeviceController,
        config: AgentConfig | None = None,
        task: str | None = None,
    ):
        """
        初始化Agent

        Args:
            controller: 设备控制器（Android/iOS等）
            config: Agent配置
            task: 任务描述
        """
        self.controller = RecordingController(controller)
        self.config = config or AgentConfig()
        self.history: list[TaskStep] = []
        self.step_count = 0
        self.task = task
        self._last_screenshot_path: Path | None = None  # 上一步操作后的截图路径
        self._last_screenshot_time: float = 0  # 上一步操作后的截图时间戳
        self._screenshot_ttl: float = 1  # 截图有效期（秒）

        # 创建带时间戳的唯一任务目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task_dir = Path(self.config.tasks_dir) / f"task_{timestamp}"
        self.task_dir.mkdir(parents=True, exist_ok=True)

        # 截图子目录
        self.screenshot_dir = self.task_dir / "screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)

    def _take_screenshot(self) -> Path:
        """截取屏幕并保存"""
        current_time = time.time()

        # 如果有上一步的操作后截图，检查是否在有效期内
        if self._last_screenshot_path is not None:
            age = current_time - self._last_screenshot_time
            if age <= self._screenshot_ttl:
                # 截图在有效期内，可以复用
                path = self._last_screenshot_path
                self._last_screenshot_path = None
                return path
            else:
                # 截图过期，清空记录
                self._last_screenshot_path = None

        if self.config.save_screenshots:
            path = self.screenshot_dir / f"step_{self.step_count:03d}.png"
            if path.exists():
                suffix = 1
                while path.exists():
                    path = (
                        self.screenshot_dir / f"step_{self.step_count:03d}_{suffix}.png"
                    )
                    suffix += 1
        else:
            path = Path("temp_screenshot.png")
        return self.controller.screenshot(path)

    def _log(self, message: str):
        """打印日志"""
        if self.config.verbose:
            print(message)

    def _should_compare_screenshots(self, action: Action) -> bool:
        """判断是否需要对比操作前后的截图"""
        return action.type in (
            ActionType.TAP,
            ActionType.LONG_PRESS,
            ActionType.INPUT,
            ActionType.SWIPE,
            ActionType.BACK,
            ActionType.APP_LAUNCH,
            ActionType.APP_REBOOT,
        )

    def _should_capture_after(self, action: Action) -> bool:
        """判断操作后是否需要截图（用于下一步复用）"""
        return action.type in (
            ActionType.TAP,
            ActionType.LONG_PRESS,
            ActionType.INPUT,
            ActionType.SWIPE,
            ActionType.BACK,
            ActionType.WAIT,
            ActionType.APP_LAUNCH,
            ActionType.APP_STOP,
            ActionType.APP_REBOOT,
        )

    def _compare_screenshots(
        self, before_path: Path
    ) -> tuple[str | None, float | None]:
        """
        对比操作前后的截图

        Returns:
            (操作后截图路径, 相似度)
        """
        try:
            from uiautoagent.agent.image_similarity import calculate_image_similarity

            # 操作后截图
            after_path = self._take_screenshot()

            # 保存供下一步复用
            self._last_screenshot_path = after_path
            self._last_screenshot_time = time.time()

            # 计算相似度
            similarity = calculate_image_similarity(before_path, after_path)

            return str(after_path), similarity
        except Exception:
            # 相似度计算失败不影响主流程
            return None, None

    def _create_task_step(
        self,
        action: Action,
        screenshot_path: Path,
        observation: str,
        success: bool,
        elapsed: float,
        screenshot_after_path: str | None = None,
        image_similarity: float | None = None,
    ) -> TaskStep:
        """创建任务步骤记录"""
        # 获取操作详情
        detail = None
        if isinstance(self.controller, RecordingController):
            detail = self.controller.last_detail

        return TaskStep(
            step_number=self.step_count,
            screenshot_path=str(screenshot_path),
            action=action,
            observation=observation,
            action_detail=detail,
            success=success,
            timestamp=time.time(),
            elapsed=round(elapsed, 3),
            screenshot_after_path=screenshot_after_path,
            image_similarity=image_similarity,
        )

    def _log_step(self, step: TaskStep):
        """打印步骤日志"""
        status = "✅" if step.success else "❌"
        self._log(f"\n[步骤 {step.step_number}] {status} 动作: {step.action}")
        if step.action.thought:
            self._log(f"思考: {step.action.thought}")
        self._log(f"观察: {step.observation}")
        if step.elapsed:
            self._log(f"耗时: {step.elapsed:.2f}s")

    def _detect_and_tap(self, screenshot_path: Path, target: str) -> DetectionResult:
        """检测并点击元素，返回检测结果"""
        from uiautoagent.detector import detect_element

        result = detect_element(screenshot_path, target)
        if result.found and result.bbox:
            self.controller.tap_bbox(result.bbox)
        return result

    def _detect_and_swipe(
        self, screenshot_path: Path, start: str, end: str
    ) -> dict[str, DetectionResult]:
        """检测起始和结束位置并执行滑动，返回检测结果"""
        from uiautoagent.detector import detect_elements

        results = detect_elements(screenshot_path, [start, end])

        start_result = results.get(start)
        end_result = results.get(end)

        if (
            start_result
            and start_result.found
            and start_result.bbox
            and end_result
            and end_result.found
            and end_result.bbox
        ):
            x1, y1 = start_result.bbox.center
            x2, y2 = end_result.bbox.center
            self.controller.swipe(x1, y1, x2, y2)

        return results

    def _execute_action(self, action: Action, screenshot_path: Path) -> str:
        """执行动作并返回观察结果"""
        try:
            if action.type == ActionType.TAP:
                assert isinstance(action.params, TapParams)
                target = action.params.target
                result = self._detect_and_tap(screenshot_path, target)
                if result.found:
                    return f"已点击: {result.description or target}"
                return f"未找到元素: {target}"

            elif action.type == ActionType.LONG_PRESS:
                assert isinstance(action.params, LongPressParams)
                target = action.params.target
                long_press_ms = action.params.long_press_ms
                from uiautoagent.detector import detect_element

                result = detect_element(screenshot_path, target)
                if not result.found or not result.bbox:
                    return f"未找到元素: {target}"
                x, y = result.bbox.center
                self.controller.long_press(x, y, long_press_ms)
                return f"已长按: {result.description or target} ({long_press_ms}ms)"

            elif action.type == ActionType.INPUT:
                assert isinstance(action.params, InputParams)
                text = action.params.text
                self.controller.input_text(text)
                return f"已输入: {text}"

            elif action.type == ActionType.SWIPE:
                assert isinstance(action.params, SwipeParams)
                if action.params.swipe_start and action.params.swipe_end:
                    results = self._detect_and_swipe(
                        screenshot_path,
                        action.params.swipe_start,
                        action.params.swipe_end,
                    )
                    start_r = results.get(action.params.swipe_start)
                    end_r = results.get(action.params.swipe_end)
                    if not start_r or not start_r.found or not start_r.bbox:
                        return f"未找到起始位置: {action.params.swipe_start}"
                    if not end_r or not end_r.found or not end_r.bbox:
                        return f"未找到结束位置: {action.params.swipe_end}"
                    return f"已从 {start_r.description or action.params.swipe_start} 滑动到 {end_r.description or action.params.swipe_end}"
                elif action.params.direction:
                    self.controller.swipe_direction(action.params.direction)
                    return f"已向{action.params.direction}滑动"
                return "未提供滑动参数（方向或起止位置描述）"

            elif action.type == ActionType.BACK:
                self.controller.back()
                return "已点击返回键"

            elif action.type == ActionType.WAIT:
                assert isinstance(action.params, WaitParams)
                wait_ms = action.params.wait_ms
                time.sleep(wait_ms / 1000)
                return f"已等待 {wait_ms}ms"

            elif action.type == ActionType.APP_LAUNCH:
                assert isinstance(action.params, AppIdParams)
                app_id = action.params.app_id
                self.controller.app_launch(app_id)
                return f"已启动应用: {app_id}"

            elif action.type == ActionType.APP_STOP:
                assert isinstance(action.params, AppIdParams)
                app_id = action.params.app_id
                self.controller.app_stop(app_id)
                return f"已停止应用: {app_id}"

            elif action.type == ActionType.APP_REBOOT:
                assert isinstance(action.params, AppIdParams)
                app_id = action.params.app_id
                self.controller.app_reboot(app_id)
                return f"已重启应用: {app_id}"

            elif action.type in (ActionType.DONE, ActionType.FAIL):
                return action.thought or ""

            return f"未知动作类型: {action.type}"

        except Exception as e:
            return f"执行出错: {e}"

    def step(
        self,
        action: Action,
        screenshot_path: Path | None = None,
        step_start: float | None = None,
    ) -> TaskStep:
        """
        执行一步操作

        Args:
            action: 要执行的动作
            screenshot_path: 操作前截图路径；若为 None 则自动截图
            step_start: 步骤开始时间（Unix 时间戳）；若为 None 则使用当前时间

        Returns:
            执行的步骤记录
        """
        if step_start is None:
            step_start = time.time()
        self.step_count += 1

        # 截图（记录操作前的屏幕状态）
        if screenshot_path is None:
            screenshot_path = self._take_screenshot()

        # 重置操作详情，避免残留上一步的坐标
        if isinstance(self.controller, RecordingController):
            self.controller.last_detail = ActionDetail()

        # 执行动作
        observation = self._execute_action(action, screenshot_path)

        # 判断是否成功
        success = (
            not observation.startswith("未找到")
            and not observation.startswith("执行出错")
            and action.type != ActionType.FAIL
        )

        # 操作后截图和相似度计算（仅对会产生界面变化的操作）
        screenshot_after_path: str | None = None
        image_similarity: float | None = None

        if self._should_compare_screenshots(action) and success:
            screenshot_after_path, image_similarity = self._compare_screenshots(
                screenshot_path
            )
            if image_similarity is not None:
                from uiautoagent.agent.image_similarity import format_similarity_change

                # action.type 可能是字符串或 ActionType 枚举
                action_type_str = (
                    action.type.value
                    if isinstance(action.type, ActionType)
                    else action.type
                )
                similarity_info = format_similarity_change(
                    image_similarity, action_type_str
                )
                observation = f"{observation} | {similarity_info}"
        elif self._should_capture_after(action) and success:
            # 不需要计算相似度，但需要截图供下一步复用
            after_path = self._take_screenshot()
            self._last_screenshot_path = after_path
            self._last_screenshot_time = time.time()
            screenshot_after_path = str(after_path)

        # 计算执行耗时
        elapsed = time.time() - step_start

        # 创建并记录步骤
        step = self._create_task_step(
            action=action,
            screenshot_path=screenshot_path,
            observation=observation,
            success=success,
            elapsed=elapsed,
            screenshot_after_path=screenshot_after_path,
            image_similarity=image_similarity,
        )
        self.history.append(step)

        # 日志输出
        self._log_step(step)

        return step

    def get_current_screenshot(self) -> Path:
        """获取当前屏幕截图（用于AI决策）"""
        return self._take_screenshot()

    def _append_step_log(self, step: TaskStep) -> None:
        """将单步执行记录实时追加到 log.txt"""
        log_path = self.task_dir / "log.txt"

        # 格式化步骤信息
        status = "✅ 成功" if step.success else "❌ 失败"
        lines = [
            "=" * 60,
            f"[步骤 {step.step_number}] {status}",
            f"时间: {datetime.fromtimestamp(step.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
            f"动作: {step.action}",
        ]

        if step.action.thought:
            lines.append(f"思考: {step.action.thought}")

        lines.append(f"观察: {step.observation}")

        if step.elapsed is not None:
            lines.append(f"耗时: {step.elapsed:.2f}s")

        if step.image_similarity is not None:
            from uiautoagent.agent.image_similarity import format_similarity_change

            similarity_info = format_similarity_change(
                step.image_similarity, step.action.type
            )
            lines.append(f"界面相似度: {similarity_info}")

        if step.ai_tokens:
            t = step.ai_tokens
            lines.append(f"Token消耗: {t.total} (输入:{t.prompt}, 输出:{t.completion})")

        lines.append("=" * 60)
        lines.append("")  # 空行分隔

        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def save_history(self, path: str | Path | None = None):
        """保存任务历史到JSON文件"""
        if path is None:
            path = self.task_dir / "history.json"

        # 从全局TokenTracker获取统计
        from uiautoagent.ai import TokenTracker

        total_tokens = TokenTracker.get_total()
        stats_by_category = TokenTracker.get_stats()

        data = {
            "total_steps": len(self.history),
            "total_tokens": {
                "prompt_tokens": total_tokens.prompt,
                "completion_tokens": total_tokens.completion,
                "total_tokens": total_tokens.total,
            },
            "tokens_by_category": {
                k: {
                    "prompt_tokens": v.prompt,
                    "completion_tokens": v.completion,
                    "total_tokens": v.total,
                }
                for k, v in stats_by_category.items()
            },
            "steps": [step.model_dump() for step in self.history],
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._log(f"\n📝 任务历史已保存至: {path}")

        # 同时保存可读的文本摘要
        self._save_text_summary()

        # 生成HTML可视化报告
        self._generate_html_report()

        # 创建 latest 软链接指向当前任务目录
        self._update_latest_symlink()

    def _generate_html_report(self):
        """生成HTML可视化报告"""
        from uiautoagent.agent.report import generate_html_report

        report_path = generate_html_report(self.history, self.task_dir, self.task)
        self._log(f"📊 HTML报告已保存至: {report_path}")

    def _update_latest_symlink(self):
        """创建/更新 latest 软链接指向当前任务目录"""
        import os

        tasks_dir = Path(self.config.tasks_dir)
        latest_link = tasks_dir / "latest"

        try:
            # 如果已存在 latest 软链接，先删除
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()

            # 创建相对路径的软链接（更便携）
            # 获取任务目录名称（如 "task_20240121_123456"）
            task_dir_name = self.task_dir.name
            os.symlink(task_dir_name, latest_link)

            self._log(f"🔗 软链接已更新: {latest_link} -> {task_dir_name}")
        except OSError as e:
            # 软链接创建失败（如权限问题、Windows 不支持等）
            self._log(f"⚠️  无法创建软链接: {e}")

    def _save_text_summary(self):
        """保存可读的文本摘要"""
        summary_path = self.task_dir / "summary.txt"

        # 从全局TokenTracker获取统计
        from uiautoagent.ai import TokenTracker

        total_tokens = TokenTracker.get_total()
        stats_by_category = TokenTracker.get_stats()

        lines = [
            "=" * 60,
            f"任务执行摘要 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            f"总步骤数: {len(self.history)}",
            "截图目录: screenshots/",
            "",
            "Token使用统计:",
            f"  输入Token: {total_tokens.prompt:,}",
            f"  输出Token: {total_tokens.completion:,}",
            f"  总计Token: {total_tokens.total:,}",
        ]

        # 按分类显示token统计
        if stats_by_category:
            # 分类名称映射
            category_names = {
                "plan": "AI计划思考",
                "clarify": "任务澄清",
                "summarize": "任务总结",
            }

            lines.append("")
            lines.append("按用途分类:")
            for category, stats in stats_by_category.items():
                name = category_names.get(category, category)
                lines.append(
                    f"  [{name}] {stats.total:,} tokens - 输入:{stats.prompt:,}, 输出:{stats.completion:,}"
                )

        lines.extend(
            [
                "",
                "步骤详情:",
                "-" * 60,
            ]
        )

        for step in self.history:
            status = "✅ 成功" if step.success else "❌ 失败"
            lines.append(f"\n[步骤 {step.step_number}] {status}")
            lines.append(f"  动作: {step.action}")
            if step.action.thought:
                lines.append(f"  思考: {step.action.thought}")
            lines.append(f"  观察: {step.observation}")
            lines.append(f"  截图: screenshots/step_{step.step_number:03d}.png")

        lines.append("\n" + "=" * 60)

        summary_path.write_text("\n".join(lines), encoding="utf-8")
        self._log(f"📄 文本摘要已保存至: {summary_path}")

    def print_summary(self):
        """打印任务执行摘要"""
        # 从全局TokenTracker获取统计
        from uiautoagent.ai import TokenTracker

        total_tokens = TokenTracker.get_total()
        stats_by_category = TokenTracker.get_stats()

        print("\n" + "=" * 50)
        print("📋 任务执行摘要")
        print("=" * 50)
        for step in self.history:
            status = "✅" if step.success else "❌"
            print(f"[{step.step_number}] {status} {step.action}")
        print("=" * 50)

        # 打印 token 使用统计
        if total_tokens.total > 0:
            print(
                f"📊 Token: {total_tokens.total:,} (输入:{total_tokens.prompt:,}, 输出:{total_tokens.completion:,})"
            )

            # 按分类详细统计
            if stats_by_category:
                # 分类名称映射
                category_names = {
                    "plan": "AI计划思考",
                    "clarify": "任务澄清",
                    "summarize": "任务总结",
                }

                print("\n按用途分类:")
                for category, stats in stats_by_category.items():
                    name = category_names.get(category, category)
                    print(
                        f"  [{name}] {stats.total:,} tokens (输入:{stats.prompt:,}, 输出:{stats.completion:,})"
                    )
        print("=" * 50)

    def get_context_for_ai(self) -> dict[str, Any]:
        """
        获取当前上下文信息，供AI决策使用

        Returns:
            包含所有历史步骤和当前截图的上下文字典
        """
        return {
            "step_count": self.step_count,
            "history": [
                {
                    "step": s.step_number,
                    "action": s.action.model_dump(),
                    "observation": s.observation,
                    "success": s.success,
                    "image_similarity": s.image_similarity,
                }
                for s in self.history
            ],
            "device_info": self.controller.get_device_info(),
        }
