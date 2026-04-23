"""测试 Action 的解析和验证"""

import pytest
from pydantic import ValidationError

from uiautoagent.agent.plan import (
    parse_plan_response,
    Action,
    ActionType,
    TapParams,
    LongPressParams,
    InputParams,
    SwipeParams,
    WaitParams,
    AppIdParams,
    DoneParams,
    EmptyParams,
)


class TestActionValidate:
    """测试 Action 的解析和验证"""

    def test_validate_tap_action(self):
        """测试解析 tap 操作"""
        data = {
            "type": "tap",
            "thought": "点击搜索按钮",
            "log": "点击搜索",
            "params": {"target": "搜索按钮"},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.TAP
        assert isinstance(result.params, TapParams)
        assert result.params.target == "搜索按钮"

    def test_validate_tap_action_with_bbox(self):
        """测试解析带 bbox 的 tap 操作"""
        data = {
            "type": "tap",
            "thought": "点击搜索按钮",
            "log": "点击搜索",
            "params": {"target": "搜索按钮", "bbox": [100, 200, 300, 250]},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.TAP
        assert isinstance(result.params, TapParams)
        assert result.params.target == "搜索按钮"
        assert result.params.bbox == [100, 200, 300, 250]

    def test_validate_long_press_action(self):
        """测试解析 long_press 操作"""
        data = {
            "type": "long_press",
            "thought": "长按消息",
            "log": "长按",
            "params": {
                "target": "消息内容",
                "long_press_ms": 1000,
            },
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.LONG_PRESS
        assert isinstance(result.params, LongPressParams)
        assert result.params.target == "消息内容"
        assert result.params.long_press_ms == 1000

    def test_validate_long_press_action_with_bbox(self):
        """测试解析带 bbox 的 long_press 操作"""
        data = {
            "type": "long_press",
            "thought": "长按消息",
            "log": "长按",
            "params": {
                "target": "消息内容",
                "long_press_ms": 1000,
                "bbox": [50, 100, 200, 150],
            },
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.LONG_PRESS
        assert isinstance(result.params, LongPressParams)
        assert result.params.bbox == [50, 100, 200, 150]

    def test_validate_long_press_with_default_ms(self):
        """测试 long_press 使用默认时长"""
        data = {
            "type": "long_press",
            "params": {"target": "消息"},
        }
        result = Action.model_validate(data)
        assert isinstance(result.params, LongPressParams)
        assert result.params.long_press_ms == 800  # 默认值

    def test_validate_input_action(self):
        """测试解析 input 操作"""
        data = {
            "type": "input",
            "thought": "输入搜索内容",
            "log": "输入",
            "params": {"text": "python"},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.INPUT
        assert isinstance(result.params, InputParams)
        assert result.params.text == "python"

    def test_validate_swipe_with_direction(self):
        """测试解析 swipe 操作（方向）"""
        data = {
            "type": "swipe",
            "thought": "向上滑动",
            "log": "上滑",
            "params": {"direction": "up"},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.SWIPE
        assert isinstance(result.params, SwipeParams)
        assert result.params.direction == "up"
        assert result.params.swipe_start_xy is None
        assert result.params.swipe_end_xy is None

    def test_validate_swipe_with_xy(self):
        """测试解析 swipe 操作（坐标）"""
        data = {
            "type": "swipe",
            "thought": "滑动",
            "log": "滑动",
            "params": {
                "swipe_start_xy": [100, 200],
                "swipe_end_xy": [300, 400],
            },
        }
        result = Action.model_validate(data)
        assert isinstance(result.params, SwipeParams)
        assert result.params.swipe_start_xy == (100, 200)
        assert result.params.swipe_end_xy == (300, 400)
        assert result.params.direction is None

    def test_validate_back_action(self):
        """测试解析 back 操作"""
        data = {
            "type": "back",
            "thought": "返回",
            "log": "返回",
            "params": {},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.BACK
        assert isinstance(result.params, EmptyParams)

    def test_validate_wait_action(self):
        """测试解析 wait 操作"""
        data = {
            "type": "wait",
            "thought": "等待",
            "log": "等待",
            "params": {"wait_ms": 2000},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.WAIT
        assert isinstance(result.params, WaitParams)
        assert result.params.wait_ms == 2000

    def test_validate_wait_with_default_ms(self):
        """测试 wait 使用默认时长"""
        data = {
            "type": "wait",
            "params": {},
        }
        result = Action.model_validate(data)
        assert isinstance(result.params, WaitParams)
        assert result.params.wait_ms == 1000  # 默认值

    def test_validate_app_launch_action(self):
        """测试解析 app_launch 操作"""
        data = {
            "type": "app_launch",
            "thought": "启动微信",
            "log": "启动",
            "params": {"app_id": "com.tencent.mm"},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.APP_LAUNCH
        assert isinstance(result.params, AppIdParams)
        assert result.params.app_id == "com.tencent.mm"

    def test_validate_app_stop_action(self):
        """测试解析 app_stop 操作"""
        data = {
            "type": "app_stop",
            "params": {"app_id": "com.tencent.mm"},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.APP_STOP

    def test_validate_app_reboot_action(self):
        """测试解析 app_reboot 操作"""
        data = {
            "type": "app_reboot",
            "thought": "重启微信",
            "log": "重启",
            "params": {"app_id": "com.tencent.mm"},
        }
        result = Action.model_validate(data)
        assert isinstance(result.params, AppIdParams)
        assert result.type == ActionType.APP_REBOOT
        assert result.params.app_id == "com.tencent.mm"

    def test_validate_done_action(self):
        """测试解析 done 操作"""
        data = {
            "type": "done",
            "thought": "完成",
            "log": "完成",
            "params": {},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.DONE
        assert isinstance(result.params, DoneParams)
        assert result.params.return_result is False

    def test_validate_done_with_result(self):
        """测试解析 done 操作（带结果）"""
        data = {
            "type": "done",
            "thought": "完成",
            "log": "完成",
            "params": {"return_result": True, "result": "任务完成，共找到15个好友"},
        }
        result = Action.model_validate(data)
        assert isinstance(result.params, DoneParams)
        assert result.params.return_result is True
        assert result.params.result == "任务完成，共找到15个好友"

    def test_validate_fail_action(self):
        """测试解析 fail 操作"""
        data = {
            "type": "fail",
            "thought": "失败",
            "log": "失败",
            "params": {},
        }
        result = Action.model_validate(data)
        assert result.type == ActionType.FAIL
        assert isinstance(result.params, EmptyParams)

    def test_validate_missing_required_field(self):
        """测试缺少必需字段时抛出 ValidationError"""
        # tap 操作缺少必需的 target 字段
        data = {
            "type": "tap",
            "thought": "点击",
            "params": {},
        }
        with pytest.raises(ValidationError):
            Action.model_validate(data)

    def test_validate_invalid_direction(self):
        """测试无效的 direction 值"""
        data = {
            "type": "swipe",
            "params": {
                "direction": "diagonal"  # 无效值
            },
        }
        with pytest.raises(ValidationError):
            Action.model_validate(data)

    def test_validate_invalid_type(self):
        """测试无效的 type 值"""
        data = {
            "type": "invalid_type",
            "params": {},
        }
        with pytest.raises(ValidationError):
            Action.model_validate(data)

    def test_validate_extra_fields_ignored(self):
        """测试额外字段被忽略（pydantic 默认行为）"""
        data = {
            "type": "tap",
            "params": {"target": "按钮"},
            "extra_field": "应该被忽略",  # 额外字段
        }
        result = Action.model_validate(data)
        # 额外字段应该被忽略，不会出现在结果中
        assert not hasattr(result, "extra_field")

    def test_validate_min_wait_ms(self):
        """测试 wait_ms 最小值约束"""
        data = {
            "type": "wait",
            "params": {
                "wait_ms": -100  # 负数，应该失败
            },
        }
        with pytest.raises(ValidationError):
            Action.model_validate(data)

    def test_validate_min_long_press_ms(self):
        """测试 long_press_ms 最小值约束"""
        data = {
            "type": "long_press",
            "params": {
                "target": "消息",
                "long_press_ms": -50,  # 负数，应该失败
            },
        }
        with pytest.raises(ValidationError):
            Action.model_validate(data)


class TestParsePlanResponse:
    """测试 parse_plan_response 函数"""

    def test_parse_single_json(self):
        """测试解析单个 JSON"""
        raw = '{"type": "tap", "thought": "test", "log": "test", "params": {"target": "按钮"}}'
        result = parse_plan_response(raw)
        assert isinstance(result, Action)
        assert result.type == ActionType.TAP
        assert result.params.target == "按钮"

    def test_parse_json_with_markdown(self):
        """测试解析被 markdown 代码块包裹的 JSON"""
        raw = """```json
        {
            "type": "tap",
            "thought": "test",
            "params": {"target": "按钮"}
        }
        ```"""
        result = parse_plan_response(raw)
        assert isinstance(result, Action)
        assert result.params.target == "按钮"

    def test_parse_json_array(self):
        """测试解析 JSON 数组（取第一个）"""
        raw = '[{"type": "tap", "params": {"target": "按钮"}}, {"type": "back", "params": {}}]'
        result = parse_plan_response(raw)
        assert isinstance(result, Action)
        assert result.type == ActionType.TAP

    def test_parse_empty_string(self):
        """测试解析空字符串"""
        with pytest.raises(ValueError, match="未找到有效的 JSON 内容"):
            parse_plan_response("")

    def test_parse_invalid_json(self):
        """测试解析无效的 JSON"""
        with pytest.raises(ValueError, match="无法解析 AI 返回的 JSON"):
            parse_plan_response("{invalid json}")
