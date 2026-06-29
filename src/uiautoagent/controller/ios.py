"""基于wdapy的iOS设备控制器"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import wdapy

from uiautoagent.controller.base import (
    DeviceController,
    direction_to_swipe_coords,
    SwipeDirection,
)


class IOSController(DeviceController):
    """iOS设备控制器（基于wdapy/WebDriverAgent）"""

    def __init__(
        self,
        udid: str | None = None,
        url: str | None = None,
    ):
        """
        初始化控制器

        Args:
            udid: 设备UDID，使用USB连接（优先）
            url: WebDriverAgent的URL，使用HTTP连接

        如果udid和url都未提供，自动检测第一个USB设备。
        """
        self.udid = udid
        self.url = url
        self._client: wdapy.AppiumClient | None = None
        self._device_info: dict | None = None
        self._window_size: tuple[int, int] | None = None
        self._scale: float | None = None

    @property
    def client(self) -> wdapy.AppiumClient:
        """懒加载wdapy客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> wdapy.AppiumClient:
        """创建wdapy客户端"""
        if self.udid:
            return wdapy.AppiumUSBClient(self.udid)
        elif self.url:
            return wdapy.AppiumClient(self.url)
        else:
            # 自动检测，使用第一个USB设备
            return wdapy.AppiumUSBClient()

    def _get_window_size(self) -> tuple[int, int]:
        """获取窗口尺寸（UIKit点，缓存结果）"""
        if self._window_size is None:
            self._window_size = self.client.window_size()
        return self._window_size

    def _get_scale(self) -> float:
        """获取屏幕缩放比例（像素/点），用于坐标转换"""
        if self._scale is None:
            self._scale = self.client.scale
        return self._scale

    def get_device_info(self) -> dict:
        """获取设备信息（缓存结果）"""
        if self._device_info:
            return self._device_info

        info = self.client.device_info()
        w, h = self._get_window_size()

        self._device_info = {
            "udid": info.uuid,
            "model": info.model,
            "name": info.name,
            "width": w,
            "height": h,
        }
        return self._device_info

    def tap(self, x: int, y: int) -> None:
        """点击屏幕指定坐标（像素坐标会自动转换为UIKit点坐标）"""
        scale = self._get_scale()
        self.client.tap(int(x / scale), int(y / scale))

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        """
        滑动屏幕

        Args:
            x1, y1: 起始坐标（像素）
            x2, y2: 结束坐标（像素）
            duration_ms: 滑动持续时间（毫秒）
        """
        scale = self._get_scale()
        self.client.swipe(
            int(x1 / scale),
            int(y1 / scale),
            int(x2 / scale),
            int(y2 / scale),
            duration=duration_ms / 1000,
        )

    def input_text(self, text: str) -> None:
        """
        输入文本

        Args:
            text: 要输入的文本
        """
        self.client.send_keys(text)

    def clear_text(self, length: int = 100) -> None:
        """
        清除文本框内容（通过模拟删除键）

        Args:
            length: 删除的字符数量
        """
        self.client.send_keys("\b" * length)

    def press_key(self, keycode: int) -> None:
        """
        按下按键

        常用映射:
            3: HOME
            24: VOLUME_UP
            25: VOLUME_DOWN
            26: POWER
        """
        key_map = {
            3: wdapy.Keycode.HOME,
            24: wdapy.Keycode.VOLUME_UP,
            25: wdapy.Keycode.VOLUME_DOWN,
            26: wdapy.Keycode.POWER,
        }
        key = key_map.get(keycode)
        if key:
            self.client.press(key)
        else:
            raise ValueError(f"不支持的iOS按键: {keycode}")

    def back(self) -> None:
        """返回（iOS通过从左边缘向右滑动模拟返回手势）"""
        w, h = self._get_window_size()
        # 从屏幕左侧边缘向右滑动，模拟iOS返回手势（坐标已是 UIKit 点）
        self.client.swipe(10, h // 2, w // 3, h // 2, duration=0.3)

    def home(self) -> None:
        """Home键"""
        self.client.homescreen()

    def screenshot(self, output_path: str | Path) -> Path:
        """
        截取屏幕

        Args:
            output_path: 输出文件路径

        Returns:
            实际保存的文件路径
        """
        output = Path(output_path)
        img = self.client.screenshot()
        img.save(str(output))
        return output

    @staticmethod
    def list_devices() -> List[str]:
        """列出所有已连接的iOS设备UDID"""
        try:
            result = subprocess.run(
                ["idevice_id", "-l"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                devices = [
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                ]
                return devices
        except FileNotFoundError:
            pass

        # 回退到 tidevice
        try:
            result = subprocess.run(
                ["tidevice", "list"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                devices = []
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if parts:
                        devices.append(parts[0])
                return devices
        except FileNotFoundError:
            pass

        return []

    def app_launch(self, app_id: str) -> None:
        """启动应用

        Args:
            app_id: Bundle ID，如 com.tencent.xin
        """
        self.client.app_start(app_id)

    def app_stop(self, app_id: str) -> None:
        """停止应用

        Args:
            app_id: Bundle ID，如 com.tencent.xin
        """
        self.client.app_terminate(app_id)
