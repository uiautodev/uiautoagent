"""设备控制器抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal

from uiautoagent.detector import BBox, DetectionResult


SwipeDirection = Literal["up", "down", "left", "right"]


def direction_to_swipe_coords(
    w: int, h: int, direction: SwipeDirection, ratio: float = 0.25
) -> tuple[int, int, int, int]:
    """方向 → 滑动起止坐标"""
    cx, cy = w // 2, h // 2
    dist_x = int(w * ratio)
    dist_y = int(h * ratio)
    return {
        "up": (cx, cy + dist_y // 2, cx, cy - dist_y // 2),
        "down": (cx, cy - dist_y // 2, cx, cy + dist_y // 2),
        "left": (cx + dist_x // 2, cy, cx - dist_x // 2, cy),
        "right": (cx - dist_x // 2, cy, cx + dist_x // 2, cy),
    }[direction]


class DeviceController(ABC):
    """设备控制器抽象基类"""

    @abstractmethod
    def get_device_info(self) -> dict:
        """获取设备信息"""
        pass

    @abstractmethod
    def tap(self, x: int, y: int) -> None:
        """点击屏幕指定坐标"""
        pass

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """滑动屏幕"""
        pass

    def swipe_direction(
        self, direction: SwipeDirection, ratio: float = 0.25, duration_ms: int = 300
    ) -> None:
        """向指定方向滑动"""
        info = self.get_device_info()
        w, h = info["width"], info["height"]
        x1, y1, x2, y2 = direction_to_swipe_coords(w, h, direction, ratio)
        self.swipe(x1, y1, x2, y2, duration_ms)

    @abstractmethod
    def input_text(self, text: str) -> None:
        """输入文本"""
        pass

    @abstractmethod
    def clear_text(self, length: int = 100) -> None:
        """清除文本"""
        pass

    @abstractmethod
    def press_key(self, keycode: int) -> None:
        """按下按键"""
        pass

    @abstractmethod
    def back(self) -> None:
        """返回键"""
        pass

    @abstractmethod
    def home(self) -> None:
        """Home键"""
        pass

    @abstractmethod
    def screenshot(self, output_path: str | Path) -> Path:
        """截取屏幕"""
        pass

    @staticmethod
    @abstractmethod
    def list_devices() -> List[str]:
        """列出所有已连接的设备"""
        pass

    @abstractmethod
    def app_launch(self, app_id: str) -> None:
        """启动应用

        Args:
            app_id: Android 为包名（如 com.tencent.mm），iOS 为 Bundle ID（如 com.tencent.xin）
        """
        pass

    @abstractmethod
    def app_stop(self, app_id: str) -> None:
        """停止应用

        Args:
            app_id: Android 为包名（如 com.tencent.mm），iOS 为 Bundle ID（如 com.tencent.xin）
        """
        pass

    # 便捷方法
    def long_press(self, x: int, y: int, duration_ms: int = 800) -> None:
        """长按指定坐标"""
        self.swipe(x, y, x, y, duration_ms)

    def app_reboot(self, app_id: str) -> None:
        """重启应用（先停止再启动）"""
        self.app_stop(app_id)
        self.app_launch(app_id)

    def tap_bbox(self, bbox: BBox) -> None:
        """点击bbox的中心点"""
        x, y = bbox.center
        self.tap(x, y)

    def tap_result(self, result: DetectionResult) -> bool:
        """点击检测结果中的元素，返回是否成功"""
        if not result.found or not result.bbox:
            return False
        self.tap_bbox(result.bbox)
        return True
