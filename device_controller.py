"""设备控制器抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from bbox_detector import BBox, DetectionResult


SwipeDirection = Literal["up", "down", "left", "right"]


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

    @abstractmethod
    def swipe_direction(self, direction: SwipeDirection, ratio: float = 0.5, duration_ms: int = 300) -> None:
        """向指定方向滑动"""
        pass

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
    def list_devices() -> list[str]:
        """列出所有已连接的设备"""
        pass

    # 便捷方法
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
