"""基于ADB命令的Android设备控制器"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
from typing import List

from PIL import Image

from uiautoagent.controller.base import DeviceController, SwipeDirection


class DeviceInfo:
    """设备信息"""

    def __init__(self, serial: str, model: str, width: int, height: int):
        self.serial = serial
        self.model = model
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"DeviceInfo(serial={self.serial}, model={self.model}, {self.width}x{self.height})"


class AndroidController(DeviceController):
    """Android设备控制器（基于ADB）"""

    def __init__(self, serial: str | None = None):
        """
        初始化控制器

        Args:
            serial: 设备序列号，None表示使用默认设备
        """
        self.serial = serial
        self._device_info: DeviceInfo | None = None

    @property
    def device_arg(self) -> List[str]:
        """返回adb命令的设备参数"""
        return ["-s", self.serial] if self.serial else []

    def _run_adb(self, *args: str) -> str:
        """执行adb命令并返回输出"""
        cmd = ["adb", *self.device_arg, *args]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ADB命令失败: {' '.join(cmd)}\n{result.stderr}")
        return result.stdout.strip()

    def get_device_info(self) -> dict:
        """获取设备信息（缓存结果）"""
        if self._device_info:
            return {
                "serial": self._device_info.serial,
                "model": self._device_info.model,
                "width": self._device_info.width,
                "height": self._device_info.height,
            }

        # 获取设备序列号
        serial = self.serial or self._run_adb("get-serialno")

        # 获取设备型号
        model = self._run_adb("shell", "getprop", "ro.product.model")

        # 获取屏幕尺寸
        wm_size = self._run_adb("shell", "wm", "size")
        match = re.search(r"Physical size: (\d+)x(\d+)", wm_size)
        if not match:
            raise RuntimeError(f"无法获取屏幕尺寸: {wm_size}")
        width, height = int(match.group(1)), int(match.group(2))

        self._device_info = DeviceInfo(
            serial=serial, model=model, width=width, height=height
        )
        return {
            "serial": serial,
            "model": model,
            "width": width,
            "height": height,
        }

    def tap(self, x: int, y: int) -> None:
        """点击屏幕指定坐标"""
        self._run_adb("shell", "input", "tap", str(x), str(y))

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 500,
    ) -> None:
        """
        滑动屏幕

        Args:
            x1, y1: 起始坐标
            x2, y2: 结束坐标
            duration_ms: 滑动持续时间（毫秒）
        """
        self._run_adb(
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def swipe_direction(
        self,
        direction: SwipeDirection,
        ratio: float = 0.25,
        duration_ms: int = 500,
    ) -> None:
        """
        向指定方向滑动

        Args:
            direction: 滑动方向
            ratio: 滑动距离占屏幕的比例（0-1）
            duration_ms: 滑动持续时间
        """
        info = self.get_device_info()
        w, h = info["width"], info["height"]
        cx, cy = w // 2, h // 2

        # 根据方向计算滑动距离
        dist_x = int(w * ratio)
        dist_y = int(h * ratio)

        moves = {
            "up": (cx, cy + dist_y // 2, cx, cy - dist_y // 2),
            "down": (cx, cy - dist_y // 2, cx, cy + dist_y // 2),
            "left": (cx + dist_x // 2, cy, cx - dist_x // 2, cy),
            "right": (cx - dist_x // 2, cy, cx + dist_x // 2, cy),
        }

        x1, y1, x2, y2 = moves[direction]
        self.swipe(x1, y1, x2, y2, duration_ms)

    def input_text(self, text: str) -> None:
        """
        输入文本（不支持中文，需要特殊字符转换）

        Args:
            text: 要输入的文本（空格用 %s 替代）
        """
        # 替换空格为 %s
        text = text.replace(" ", "%s")
        self._run_adb("shell", "input", "text", text)

    def clear_text(self, length: int = 100) -> None:
        """
        清除文本框内容（通过模拟删除键）

        Args:
            length: 删除的字符数量
        """
        # Android keycode: KEYCODE_DEL = 67
        for _ in range(length):
            self._run_adb("shell", "input", "keyevent", "67")

    def press_key(self, keycode: int) -> None:
        """
        按下按键

        常用keycode:
            3: HOME
            4: BACK
            24: VOLUME_UP
            25: VOLUME_DOWN
            26: POWER
            67: DEL
            66: ENTER
        """
        self._run_adb("shell", "input", "keyevent", str(keycode))

    def back(self) -> None:
        """返回键"""
        self.press_key(4)

    def home(self) -> None:
        """Home键"""
        self.press_key(3)

    def screenshot(self, output_path: str | Path) -> Path:
        """
        截取屏幕

        Args:
            output_path: 输出文件路径

        Returns:
            实际保存的文件路径
        """
        output = Path(output_path)
        temp_path = "/sdcard/screenshot.png"
        try:
            self._run_adb("shell", "screencap", "-p", temp_path)
            self._run_adb("pull", temp_path, str(output))
            self._run_adb("shell", "rm", temp_path)
        except RuntimeError:
            # 安全验证等场景下 screencap 会失败，返回纯黑图片
            info = self.get_device_info()
            img = Image.new("RGB", (info["width"], info["height"]), (0, 0, 0))
            img.save(output)
        return output

    @staticmethod
    def list_devices() -> List[str]:
        """列出所有已连接的设备序列号"""
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().split("\n")[1:]  # 跳过标题行
        devices = []
        for line in lines:
            if "\tdevice" in line:
                serial = line.split("\t")[0]
                devices.append(serial)
        return devices

    def app_launch(self, app_id: str) -> None:
        """启动应用

        Args:
            app_id: 应用包名，如 com.tencent.mm
        """
        output = self._run_adb(
            "shell",
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            "-c",
            "android.intent.category.LAUNCHER",
            app_id,
        )
        for line in output.splitlines():
            line = line.strip()
            if "/" in line and app_id in line:
                self._run_adb("shell", "am", "start", "-n", line)
                return
        raise RuntimeError(f"无法解析应用 {app_id} 的主Activity: {output}")

    def app_stop(self, app_id: str) -> None:
        """停止应用

        Args:
            app_id: 应用包名，如 com.tencent.mm
        """
        self._run_adb("shell", "am", "force-stop", app_id)


def find_and_tap(
    controller: AndroidController,
    image_path: str | Path,
    query: str,
    **detect_kwargs,
) -> bool:
    """
    截图 -> 检测元素 -> 点击

    Args:
        controller: Android控制器
        image_path: 截图保存路径
        query: 要查找的元素描述
        **detect_kwargs: 传递给detect_element的额外参数

    Returns:
        是否成功找到并点击
    """
    from uiautoagent.detector import detect_element

    # 截图
    controller.screenshot(image_path)

    # 检测
    result = detect_element(image_path, query, **detect_kwargs)

    # 点击
    return controller.tap_result(result)
