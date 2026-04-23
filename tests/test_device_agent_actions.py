"""Tests for new device actions."""

from pathlib import Path

from uiautoagent.agent.device_agent import Action, ActionType, AgentConfig, DeviceAgent
from uiautoagent.agent.plan import LongPressParams, AppIdParams
from uiautoagent.controller.base import DeviceController


class DummyController(DeviceController):
    """Minimal controller for action tests."""

    def __init__(self):
        self.calls: list[tuple] = []

    def get_device_info(self) -> dict:
        return {"model": "dummy", "width": 1080, "height": 1920}

    def tap(self, x: int, y: int) -> None:
        self.calls.append(("tap", x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))

    def swipe_direction(
        self, direction, ratio: float = 0.25, duration_ms: int = 300
    ) -> None:
        self.calls.append(("swipe_direction", direction, ratio, duration_ms))

    def input_text(self, text: str) -> None:
        self.calls.append(("input_text", text))

    def clear_text(self, length: int = 100) -> None:
        self.calls.append(("clear_text", length))

    def press_key(self, keycode: int) -> None:
        self.calls.append(("press_key", keycode))

    def back(self) -> None:
        self.calls.append(("back",))

    def home(self) -> None:
        self.calls.append(("home",))

    def screenshot(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.write_bytes(b"")
        self.calls.append(("screenshot", str(output)))
        return output

    @staticmethod
    def list_devices() -> list[str]:
        return []

    def app_launch(self, app_id: str) -> None:
        self.calls.append(("app_launch", app_id))

    def app_stop(self, app_id: str) -> None:
        self.calls.append(("app_stop", app_id))

    def app_reboot(self, app_id: str) -> None:
        self.calls.append(("app_stop", app_id))
        self.calls.append(("app_launch", app_id))

    def long_press(self, x: int, y: int, duration_ms: int = 800) -> None:
        self.calls.append(("swipe", x, y, x, y, duration_ms))


def test_long_press_by_bbox(tmp_path):
    controller = DummyController()
    agent = DeviceAgent(
        controller,
        config=AgentConfig(
            tasks_dir=str(tmp_path), save_screenshots=False, verbose=False
        ),
    )

    # 创建一个真实的截图文件（100x100）
    from PIL import Image

    img_path = tmp_path / "screen.png"
    Image.new("RGB", (100, 100)).save(img_path)

    step = agent.step(
        Action(
            type=ActionType.LONG_PRESS,
            thought="长按目标",
            params=LongPressParams(
                target="确定按钮", long_press_ms=900, bbox=[100, 200, 300, 400]
            ),
        ),
        screenshot_path=img_path,
    )

    assert step.success is True
    # bbox [100,200,300,400] -> 实际坐标: (10,20,30,40), center=(20,30)
    assert ("swipe", 20, 30, 20, 30, 900) in controller.calls
    assert step.observation == "已长按: 确定按钮 (900ms)"


def test_app_reboot_action(tmp_path):
    controller = DummyController()
    agent = DeviceAgent(
        controller,
        config=AgentConfig(
            tasks_dir=str(tmp_path), save_screenshots=False, verbose=False
        ),
    )

    step = agent.step(
        Action(
            type=ActionType.APP_REBOOT,
            thought="重启微信",
            params=AppIdParams(app_id="com.tencent.mm"),
        ),
        screenshot_path=tmp_path / "screen.png",
    )

    assert step.success is True
    # app_reboot 会先调用 app_stop 再调用 app_launch
    assert ("app_stop", "com.tencent.mm") in controller.calls
    assert ("app_launch", "com.tencent.mm") in controller.calls
    # 确保顺序正确（app_stop 在 app_launch 之前）
    stop_idx = controller.calls.index(("app_stop", "com.tencent.mm"))
    launch_idx = controller.calls.index(("app_launch", "com.tencent.mm"))
    assert stop_idx < launch_idx, "app_stop should be called before app_launch"
    assert step.observation == "已重启应用: com.tencent.mm"
