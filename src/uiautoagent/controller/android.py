"""基于 adbutils 的 Android 设备控制器"""

from __future__ import annotations

from pathlib import Path
from typing import List

from adbutils import AdbClient, AdbDevice
from PIL import Image

from uiautoagent.controller.base import DeviceController


class AndroidController(DeviceController):
    """Android 设备控制器（基于 adbutils）"""

    def __init__(self, serial: str | None = None):
        self._client = AdbClient()
        self._device: AdbDevice = self._client.device(serial)
        self._device_info: dict | None = None

    def get_device_info(self) -> dict:
        if self._device_info:
            return self._device_info

        serial = self._device.serial
        model = self._device.shell("getprop ro.product.model")

        size = self._device.window_size()
        width, height = size.width, size.height

        self._device_info = {
            "serial": serial,
            "model": model,
            "width": width,
            "height": height,
        }
        return self._device_info

    def tap(self, x: int, y: int) -> None:
        self._device.shell(f"input tap {x} {y}")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 500,
    ) -> None:
        self._device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, text: str) -> None:
        self._device.send_keys(text)

    def clear_text(self, length: int = 100) -> None:
        for _ in range(length):
            self._device.shell("input keyevent 67")

    def press_key(self, keycode: int) -> None:
        self._device.shell(f"input keyevent {keycode}")

    def back(self) -> None:
        self.press_key(4)

    def home(self) -> None:
        self.press_key(3)

    def screenshot(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        try:
            img = self._device.screenshot()
            img.save(str(output))
        except Exception:
            size = self._device.window_size()  # 设备离线则直接抛异常
            img = Image.new("RGB", (size.width, size.height), (0, 0, 0))
            img.save(output)
        return output

    @staticmethod
    def list_devices() -> List[str]:
        client = AdbClient()
        return [d.serial for d in client.device_list()]  # type: ignore[misc]

    def app_launch(self, app_id: str) -> None:
        output = self._device.shell(
            f"cmd package resolve-activity --brief -c android.intent.category.LAUNCHER {app_id}"
        )
        for line in output.splitlines():
            line = line.strip()
            if "/" in line and app_id in line:
                self._device.shell(f"am start -n {line}")
                return
        raise RuntimeError(f"无法解析应用 {app_id} 的主Activity: {output}")

    def app_stop(self, app_id: str) -> None:
        self._device.shell(f"am force-stop {app_id}")


def find_and_tap(
    controller: AndroidController,
    image_path: str | Path,
    query: str,
    **detect_kwargs,
) -> bool:
    """截图 -> 检测元素 -> 点击"""
    from uiautoagent.detector import detect_element

    controller.screenshot(image_path)
    result = detect_element(image_path, query, **detect_kwargs)
    return controller.tap_result(result)
