"""通用设备AI Agent - 支持自主决策和执行任务"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import dictlog

from pydantic import BaseModel, ConfigDict, Field

from uiautoagent.controller.base import DeviceController
from uiautoagent.types import TokenUsage
from uiautoagent.detector import BBox
from uiautoagent.agent.plan import HistoryEntry
from uiautoagent.agent.plan import (
    Action,
    ActionType,
    AppIdParams,
    InputParams,
    LongPressParams,
    SwipeParams,
    TapParams,
    TaskProposal,
    WaitParams,
)
from uiautoagent.env import env

# 模块级 logger
log = dictlog.get_logger(__name__)


class TaskStep(BaseModel):
    """任务执行步骤记录"""

    model_config = ConfigDict(use_enum_values=True)

    step_number: int
    screenshot_path: str
    action: Action
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
    report_dir: str | None = Field(default_factory=lambda: env.report_dir)
    save_screenshots: bool = True
    verbose: bool = True
    step_wait_ms: int = Field(default_factory=lambda: env.step_wait_ms)


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
        self.controller = controller
        self.config = config or AgentConfig()
        self.history: list[TaskStep] = []
        self.step_count = 0
        self.task = task
        self.proposal: TaskProposal | None = None
        self._last_screenshot_path: Path | None = None  # 上一步操作后的截图路径
        self._last_screenshot_time: float = 0  # 上一步操作后的截图时间戳
        self._screenshot_ttl: float = 1  # 截图有效期（秒）

        # 创建任务输出目录
        if self.config.report_dir:
            self.report_dir = Path(self.config.report_dir)
            if self.report_dir.exists():
                self._log(f"⚠️  报告目录已存在，输出文件将被覆盖: {self.report_dir}")
        else:
            import shutil

            self.report_dir = Path("uiautoagent_report")
            if self.report_dir.exists():
                shutil.rmtree(self.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 截图子目录
        self.screenshot_dir = self.report_dir / "screenshots"
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

    def _log(self, message: str, **kwargs: Any):
        """打印日志"""
        if self.config.verbose:
            log.debug(message.strip(), **kwargs, stacklevel=2)

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
        success: bool,
        elapsed: float,
        screenshot_after_path: str | None = None,
        image_similarity: float | None = None,
    ) -> TaskStep:
        """创建任务步骤记录"""
        return TaskStep(
            step_number=self.step_count,
            screenshot_path=str(screenshot_path),
            action=action,
            success=success,
            timestamp=time.time(),
            elapsed=round(elapsed, 3),
            screenshot_after_path=screenshot_after_path,
            image_similarity=image_similarity,
        )

    def _normalized_bbox_to_actual(
        self, bbox: list[int], screenshot_path: Path
    ) -> BBox:
        """将1000x1000归一化坐标转换为实际截图坐标"""
        from PIL import Image

        img = Image.open(screenshot_path)
        w, h = img.size
        x1, y1, x2, y2 = bbox
        return BBox(
            x1=max(0, int(x1 * w / 1000)),
            y1=max(0, int(y1 * h / 1000)),
            x2=min(w, int(x2 * w / 1000)),
            y2=min(h, int(y2 * h / 1000)),
        )

    def _normalized_xy_to_actual(
        self, xy: tuple[int, int], screenshot_path: Path
    ) -> tuple[int, int]:
        """将1000x1000归一化坐标转换为实际截图坐标"""
        from PIL import Image

        img = Image.open(screenshot_path)
        w, h = img.size
        x, y = xy
        return int(x * w / 1000), int(y * h / 1000)

    def _execute_action(self, action: Action, screenshot_path: Path) -> None:
        """执行动作，失败时抛出异常"""
        if action.type == ActionType.TAP:
            assert isinstance(action.params, TapParams)
            if not action.params.bbox:
                raise ValueError(f"未提供点击坐标: {action.params.target}")
            actual_bbox = self._normalized_bbox_to_actual(
                action.params.bbox, screenshot_path
            )
            self.controller.tap_bbox(actual_bbox)

        elif action.type == ActionType.LONG_PRESS:
            assert isinstance(action.params, LongPressParams)
            if not action.params.bbox:
                raise ValueError(f"未提供长按坐标: {action.params.target}")
            actual_bbox = self._normalized_bbox_to_actual(
                action.params.bbox, screenshot_path
            )
            x, y = actual_bbox.center
            self.controller.long_press(x, y, action.params.long_press_ms)

        elif action.type == ActionType.INPUT:
            assert isinstance(action.params, InputParams)
            self.controller.input_text(action.params.text)

        elif action.type == ActionType.SWIPE:
            assert isinstance(action.params, SwipeParams)
            if action.params.swipe_start_xy and action.params.swipe_end_xy:
                x1, y1 = self._normalized_xy_to_actual(
                    action.params.swipe_start_xy, screenshot_path
                )
                x2, y2 = self._normalized_xy_to_actual(
                    action.params.swipe_end_xy, screenshot_path
                )
                self.controller.swipe(x1, y1, x2, y2)
            elif action.params.direction:
                self.controller.swipe_direction(action.params.direction)
            else:
                raise ValueError("未提供滑动参数（方向或坐标）")

        elif action.type == ActionType.BACK:
            self.controller.back()

        elif action.type == ActionType.WAIT:
            assert isinstance(action.params, WaitParams)
            time.sleep(action.params.wait_ms / 1000)

        elif action.type == ActionType.APP_LAUNCH:
            assert isinstance(action.params, AppIdParams)
            self.controller.app_launch(action.params.app_id)

        elif action.type == ActionType.APP_STOP:
            assert isinstance(action.params, AppIdParams)
            self.controller.app_stop(action.params.app_id)

        elif action.type == ActionType.APP_REBOOT:
            assert isinstance(action.params, AppIdParams)
            self.controller.app_reboot(action.params.app_id)

        elif action.type in (ActionType.DONE, ActionType.FAIL):
            pass

        else:
            raise ValueError(f"未知动作类型: {action.type}")

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

        # 判断是否成功（_execute_action 抛异常说明失败，DONE/FAIL 由调用方处理）
        success = action.type != ActionType.FAIL
        if success:
            # 执行动作
            log.info(
                f"步骤 {self.step_count}: {action}",
                screenshot=screenshot_path,
                thought=action.thought,
            )
            self._execute_action(action, screenshot_path)
            # 等待界面加载
            if self.config.step_wait_ms > 0:
                time.sleep(self.config.step_wait_ms / 1000)
        else:
            log.error(
                f"步骤 {self.step_count}: {action} 失败",
                screenshot=screenshot_path,
                thought=action.thought,
            )

        # 操作后截图和相似度计算（仅对会产生界面变化的操作）
        screenshot_after_path: str | None = None
        image_similarity: float | None = None

        if self._should_compare_screenshots(action) and success:
            screenshot_after_path, image_similarity = self._compare_screenshots(
                screenshot_path
            )
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
            success=success,
            elapsed=elapsed,
            screenshot_after_path=screenshot_after_path,
            image_similarity=image_similarity,
        )
        self.history.append(step)
        return step

    def get_current_screenshot(self) -> Path:
        """获取当前屏幕截图（用于AI决策）"""
        return self._take_screenshot()

    def _append_step_log(self, step: TaskStep) -> None:
        """将单步执行记录实时追加到 log.txt"""
        log_path = self.report_dir / "log.txt"

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
            path = self.report_dir / "history.json"

        # 从全局TokenTracker获取统计
        from uiautoagent.ai import TokenTracker

        total_tokens = TokenTracker.get_total()
        stats_by_category = TokenTracker.get_stats()

        data = {}
        if self.proposal:
            data["proposal"] = self.proposal.model_dump()
        data["total_steps"] = len(self.history)
        data["total_tokens"] = {
            "prompt_tokens": total_tokens.prompt,
            "completion_tokens": total_tokens.completion,
            "total_tokens": total_tokens.total,
        }
        data["tokens_by_category"] = {
            k: {
                "prompt_tokens": v.prompt,
                "completion_tokens": v.completion,
                "total_tokens": v.total,
            }
            for k, v in stats_by_category.items()
        }
        data["steps"] = [step.model_dump() for step in self.history]
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._log(f"📝 任务历史已保存至: {path}")

        # 同时保存可读的文本摘要
        self._save_text_summary()

        # 生成HTML可视化报告
        self._generate_html_report()

        # 创建 latest 软链接指向当前任务目录
        self._update_latest_symlink()

    def _generate_html_report(self):
        """生成HTML可视化报告"""
        from uiautoagent.agent.report import generate_html_report

        report_path = generate_html_report(self.history, self.report_dir, self.task)
        self._log(f"📊 HTML报告已保存至: {report_path}")

    def _update_latest_symlink(self):
        """创建/更新 latest 软链接"""
        # uiautoagent_report 为扁平目录，无需软链接
        pass

    def _save_text_summary(self):
        """保存可读的文本摘要"""
        summary_path = self.report_dir / "summary.txt"

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

        task_log = log.bind()

        for step in self.history:
            status = "✅" if step.success else "❌"
            task_log.trace(
                f"[{step.step_number}] {status} {step.action}",
                step_number=step.step_number,
                success=step.success,
            )

        # 打印 token 使用统计
        if total_tokens.total > 0:
            task_log.info(
                "📊 Token统计",
                total=total_tokens.total,
                prompt=total_tokens.prompt,
                completion=total_tokens.completion,
            )

            # 按分类详细统计
            if stats_by_category:
                # 分类名称映射
                category_names = {
                    "plan": "AI计划思考",
                    "clarify": "任务澄清",
                    "summarize": "任务总结",
                }

                task_log.info("按用途分类:")
                for category, stats in stats_by_category.items():
                    name = category_names.get(category, category)
                    task_log.info(
                        f"[{name}]",
                        category=name,
                        total=stats.total,
                        prompt=stats.prompt,
                        completion=stats.completion,
                    )

    def get_context_for_ai(self) -> dict[str, Any]:
        """
        获取当前上下文信息，供AI决策使用

        Returns:
            包含所有历史步骤和当前截图的上下文字典
        """
        return {
            "step_count": self.step_count,
            "history": [
                HistoryEntry(
                    step_number=s.step_number,
                    action=s.action,
                    success=s.success,
                    image_similarity=s.image_similarity,
                )
                for s in self.history
            ],
            "device_info": self.controller.get_device_info(),
        }
