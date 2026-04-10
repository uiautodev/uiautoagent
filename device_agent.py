"""通用设备AI Agent - 支持自主决策和执行任务"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from device_controller import DeviceController, SwipeDirection


class ActionType(str, Enum):
    """动作类型"""
    TAP = "tap"               # 点击元素
    INPUT = "input"           # 输入文本
    SWIPE = "swipe"           # 滑动
    BACK = "back"             # 返回
    WAIT = "wait"             # 等待
    DONE = "done"             # 任务完成
    FAIL = "fail"             # 任务失败


class Action(BaseModel):
    """执行的动作"""
    type: ActionType
    thought: str              # 为什么要执行这个动作
    target: str | None = None # 目标元素描述（tap用）
    position: tuple[int, int] | None = None  # 具体坐标
    text: str | None = None   # 输入的文本
    direction: SwipeDirection | None = None  # 滑动方向
    wait_ms: int = 1000       # 等待时间

    class Config:
        use_enum_values = True

    def __str__(self) -> str:
        if self.type == ActionType.TAP:
            pos = f"@{self.position}" if self.position else ""
            return f"点击: {self.target}{pos}"
        elif self.type == ActionType.INPUT:
            return f"输入: {self.text}"
        elif self.type == ActionType.SWIPE:
            return f"滑动: {self.direction}"
        elif self.type == ActionType.BACK:
            return "返回"
        elif self.type == ActionType.WAIT:
            return f"等待 {self.wait_ms}ms"
        elif self.type == ActionType.DONE:
            return f"✅ 完成: {self.thought}"
        elif self.type == ActionType.FAIL:
            return f"❌ 失败: {self.thought}"
        return self.type


class TaskStep(BaseModel):
    """任务执行步骤记录"""
    step_number: int
    screenshot_path: str
    action: Action
    observation: str      # 执行后的观察结果
    success: bool
    timestamp: float

    class Config:
        use_enum_values = True


class AgentConfig(BaseModel):
    """Agent配置"""
    max_steps: int = 20          # 最大执行步数
    screenshot_dir: str = "screenshots"
    save_screenshots: bool = True
    verbose: bool = True


class DeviceAgent:
    """通用设备自动化AI代理（支持Android/iOS等）"""

    def __init__(
        self,
        controller: DeviceController,
        config: AgentConfig | None = None,
    ):
        """
        初始化Agent

        Args:
            controller: 设备控制器（Android/iOS等）
            config: Agent配置
        """
        self.controller = controller
        self.config = config or AgentConfig()
        self.history: list[TaskStep] = []
        self.step_count = 0
        self.screenshot_dir = Path(self.config.screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)

    def _take_screenshot(self) -> Path:
        """截取屏幕并保存"""
        if self.config.save_screenshots:
            path = self.screenshot_dir / f"step_{self.step_count:03d}.png"
        else:
            path = Path("temp_screenshot.png")
        return self.controller.screenshot(path)

    def _log(self, message: str):
        """打印日志"""
        if self.config.verbose:
            print(message)

    def _detect_and_tap(self, screenshot_path: Path, target: str) -> tuple[bool, str]:
        """检测并点击元素"""
        from bbox_detector import detect_element

        result = detect_element(screenshot_path, target)
        if result.found and result.bbox:
            self.controller.tap_bbox(result.bbox)
            return True, f"已点击: {result.description or target}"
        return False, f"未找到元素: {target}"

    def _execute_action(self, action: Action, screenshot_path: Path) -> str:
        """执行动作并返回观察结果"""
        try:
            if action.type == ActionType.TAP:
                if action.position:
                    x, y = action.position
                    self.controller.tap(x, y)
                    return f"已点击坐标 ({x}, {y})"
                elif action.target:
                    success, msg = self._detect_and_tap(screenshot_path, action.target)
                    return msg

            elif action.type == ActionType.INPUT:
                if action.text:
                    self.controller.input_text(action.text)
                    return f"已输入: {action.text}"
                return "未提供输入文本"

            elif action.type == ActionType.SWIPE:
                if action.direction:
                    self.controller.swipe_direction(action.direction)
                    return f"已向{action.direction}滑动"
                return "未提供滑动方向"

            elif action.type == ActionType.BACK:
                self.controller.back()
                return "已点击返回键"

            elif action.type == ActionType.WAIT:
                time.sleep(action.wait_ms / 1000)
                return f"已等待 {action.wait_ms}ms"

            elif action.type in (ActionType.DONE, ActionType.FAIL):
                return action.thought or ""

            return f"未知动作类型: {action.type}"

        except Exception as e:
            return f"执行出错: {e}"

    def step(self, action: Action) -> TaskStep:
        """
        执行一步操作

        Args:
            action: 要执行的动作

        Returns:
            执行的步骤记录
        """
        self.step_count += 1

        # 截图（记录操作前的屏幕状态）
        screenshot_path = self._take_screenshot()

        # 执行动作
        observation = self._execute_action(action, screenshot_path)

        # 判断是否成功
        success = (
            not observation.startswith("未找到")
            and not observation.startswith("执行出错")
            and action.type != ActionType.FAIL
        )

        # 记录步骤
        step = TaskStep(
            step_number=self.step_count,
            screenshot_path=str(screenshot_path),
            action=action,
            observation=observation,
            success=success,
            timestamp=time.time(),
        )
        self.history.append(step)

        # 日志输出
        status = "✅" if success else "❌"
        self._log(f"\n[步骤 {self.step_count}] {status}")
        self._log(f"  动作: {action}")
        if action.thought:
            self._log(f"  思考: {action.thought}")
        self._log(f"  观察: {observation}")

        return step

    def get_current_screenshot(self) -> Path:
        """获取当前屏幕截图（用于AI决策）"""
        return self._take_screenshot()

    def save_history(self, path: str | Path = "task_history.json"):
        """保存任务历史到JSON文件"""
        data = {
            "total_steps": len(self.history),
            "steps": [step.model_dump() for step in self.history],
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self._log(f"\n📝 任务历史已保存至: {path}")

    def print_summary(self):
        """打印任务执行摘要"""
        print("\n" + "=" * 50)
        print("📋 任务执行摘要")
        print("=" * 50)
        for step in self.history:
            status = "✅" if step.success else "❌"
            print(f"[{step.step_number}] {status} {step.action}")
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
                }
                for s in self.history  # 所有步骤
            ],
            "current_screenshot": str(self.get_current_screenshot()),
            "device_info": self.controller.get_device_info(),
        }
