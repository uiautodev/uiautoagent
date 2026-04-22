"""AI 规划响应模型 - 统一结构，params 使用联合类型"""

from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field, model_validator
import logging

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """动作类型"""

    TAP = "tap"  # 点击元素
    LONG_PRESS = "long_press"  # 长按元素
    INPUT = "input"  # 输入文本
    SWIPE = "swipe"  # 滑动
    BACK = "back"  # 返回
    WAIT = "wait"  # 等待
    DONE = "done"  # 任务完成
    FAIL = "fail"  # 任务失败
    APP_LAUNCH = "app_launch"  # 启动应用
    APP_STOP = "app_stop"  # 停止应用
    APP_REBOOT = "app_reboot"  # 重启应用


# ========== 各种动作的 Params 类型 ==========


class EmptyParams(BaseModel):
    """空参数 - 用于 back、fail 等无额外参数的操作"""

    pass


class TapParams(BaseModel):
    """点击操作参数"""

    target: str = Field(..., description="目标元素描述，如'搜索按钮'")


class LongPressParams(BaseModel):
    """长按操作参数"""

    target: str = Field(..., description="目标元素描述")
    long_press_ms: int = Field(default=800, ge=0, description="长按毫秒数，默认800")


class InputParams(BaseModel):
    """输入文本操作参数"""

    text: str = Field(..., description="要输入的文本内容")


class SwipeParams(BaseModel):
    """滑动操作参数"""

    # 方式1: 按方向滑动
    direction: Literal["up", "down", "left", "right"] | None = Field(
        default=None, description="滑动方向（up/down/left/right）"
    )
    # 方式2: 按位置描述滑动
    swipe_start: str | None = Field(
        default=None, description="滑动起始位置描述，如'头像图标'"
    )
    swipe_end: str | None = Field(
        default=None, description="滑动结束位置描述，如'设置按钮'"
    )


class WaitParams(BaseModel):
    """等待操作参数"""

    wait_ms: int = Field(default=1000, ge=0, description="等待毫秒数，默认1000")


class AppIdParams(BaseModel):
    """应用ID参数"""

    app_id: str = Field(
        ...,
        description="应用包名（Android）或 Bundle ID（iOS），如 com.tencent.mm",
    )


class DoneParams(BaseModel):
    """任务完成操作参数"""

    return_result: bool = Field(default=False, description="是否返回观察结果")
    result: str | None = Field(default=None, description="任务返回的结果或答案")


# ========== 联合类型 ==========

ActionParams = Union[
    EmptyParams,
    TapParams,
    LongPressParams,
    InputParams,
    SwipeParams,
    WaitParams,
    AppIdParams,
    DoneParams,
]


# ========== 统一的 PlanAction ==========

# 定义 type 到 params 类型的映射
_ACTION_TYPE_TO_PARAMS = {
    ActionType.TAP: TapParams,
    ActionType.LONG_PRESS: LongPressParams,
    ActionType.INPUT: InputParams,
    ActionType.SWIPE: SwipeParams,
    ActionType.WAIT: WaitParams,
    ActionType.APP_LAUNCH: AppIdParams,
    ActionType.APP_STOP: AppIdParams,
    ActionType.APP_REBOOT: AppIdParams,
    ActionType.DONE: DoneParams,
    ActionType.BACK: EmptyParams,
    ActionType.FAIL: EmptyParams,
}


class Action(BaseModel):
    """统一的动作模型 - 所有动作都有 type, thought, log, params 四个字段"""

    type: ActionType
    thought: str = ""
    log: str = ""
    params: ActionParams = Field(default_factory=EmptyParams)

    model_config = {"validate_assignment": True}

    @model_validator(mode="before")
    @classmethod
    def validate_params_by_type(cls, data):
        """根据 type 字段验证 params 的类型"""
        if isinstance(data, dict):
            action_type = data.get("type")
            if action_type:
                # 获取预期的 params 类
                expected_params_cls = _ACTION_TYPE_TO_PARAMS.get(action_type)
                if expected_params_cls:
                    params_data = data.get("params", {})
                    # 如果 params 不是 dict，保持原样
                    if isinstance(params_data, dict):
                        # 使用预期的 params 类来验证数据
                        validated_params = expected_params_cls.model_validate(
                            params_data
                        )
                        data["params"] = validated_params
        return data

    def __str__(self) -> str:
        """返回友好的字符串表示"""
        if self.type == ActionType.TAP:
            assert isinstance(self.params, TapParams)
            return f"点击: {self.params.target}"
        elif self.type == ActionType.LONG_PRESS:
            assert isinstance(self.params, LongPressParams)
            target = self.params.target or "坐标"
            return f"长按: {target} ({self.params.long_press_ms}ms)"
        elif self.type == ActionType.INPUT:
            assert isinstance(self.params, InputParams)
            return f"输入: {self.params.text}"
        elif self.type == ActionType.SWIPE:
            assert isinstance(self.params, SwipeParams)
            if self.params.swipe_start and self.params.swipe_end:
                return f"滑动: {self.params.swipe_start} → {self.params.swipe_end}"
            if self.params.direction:
                return f"滑动: {self.params.direction}"
            return "滑动"
        elif self.type == ActionType.BACK:
            return "返回"
        elif self.type == ActionType.WAIT:
            assert isinstance(self.params, WaitParams)
            return f"等待 {self.params.wait_ms}ms"
        elif self.type == ActionType.DONE:
            return f"✅ 完成: {self.thought}" if self.thought else "✅ 完成"
        elif self.type == ActionType.FAIL:
            return f"❌ 失败: {self.thought}" if self.thought else "❌ 失败"
        elif self.type == ActionType.APP_LAUNCH:
            assert isinstance(self.params, AppIdParams)
            return f"启动应用: {self.params.app_id}"
        elif self.type == ActionType.APP_STOP:
            assert isinstance(self.params, AppIdParams)
            return f"停止应用: {self.params.app_id}"
        elif self.type == ActionType.APP_REBOOT:
            assert isinstance(self.params, AppIdParams)
            return f"重启应用: {self.params.app_id}"
        return str(self.type)


# 保持向后兼容的别名
PlanAction = Action
PlanResponse = Action


def get_action_examples_prompt() -> str:
    """获取操作类型说明和示例的 Markdown 文本"""

    return """## 操作类型说明

**所有操作都包含这四个字段：**
- `type`: 操作类型（如 "tap", "swipe" 等）
- `thought`: 为什么执行这个操作
- `log`: 简洁说明要做的事情
- `params`: 操作参数（不同类型有不同参数）

---

### 1. tap - 点击元素

**params 字段：**
- `target`: 目标元素描述，如"搜索按钮"

---

### 2. long_press - 长按元素

**params 字段：**
- `target`: 目标元素描述
- `long_press_ms`: 长按毫秒数（可选，默认800）

---

### 3. input - 输入文本

**params 字段：**
- `text`: 要输入的文本内容

---

### 4. swipe - 滑动屏幕

**方式1：按方向滑动**
- `direction`: "up" / "down" / "left" / "right"

**方式2：按位置描述滑动**
- `swipe_start`: 起始位置描述，如"个人资料图标"
- `swipe_end`: 结束位置描述，如"设置按钮"

*direction 和 swipe_start/swipe_end 二选一*

---

### 5. back - 返回上一页

**params 字段：** 无（空对象 `{}`）

---

### 6. wait - 等待

**params 字段：**
- `wait_ms`: 等待毫秒数（可选，默认1000）

---

### 7. app_launch - 启动应用

**params 字段：**
- `app_id`: 应用包名，如 `com.tencent.mm`

---

### 8. app_stop - 停止应用

**params 字段：**
- `app_id`: 应用包名

---

### 9. app_reboot - 重启应用

**params 字段：**
- `app_id`: 应用包名

---

### 10. done - 任务完成

**params 字段：**
- `return_result`: 是否返回观察结果（可选，默认false）
- `result`: 任务返回的结果或答案（当 return_result=true 时）

---

### 11. fail - 任务失败

**params 字段：** 无（空对象 `{}`）

---

## 使用说明

1. **只包含必需字段**：每个操作只需要包含它特有的字段
2. **省略默认值**：如 `long_press_ms` 默认800，`wait_ms` 默认1000，无需指定
3. **input 前置**：输入前需要先用 tap 点击输入框
"""


def parse_plan_response(raw: str) -> PlanAction:
    """
    解析 AI 返回的 JSON 为 PlanAction

    使用 json_repair 自动处理 markdown 代码块和常见的 LLM 格式错误。

    Args:
        raw: JSON 字符串（可能被 markdown 代码块包裹）

    Returns:
        解析后的 PlanAction 实例

    Raises:
        ValueError: 解析失败时
    """
    from json_repair import loads

    if not raw.strip():
        raise ValueError("未找到有效的 JSON 内容")

    try:
        data = loads(raw)
        # json_repair 遇到多个 JSON 对象时会返回列表，取第一个
        if isinstance(data, list):
            if not data:
                raise ValueError("未找到有效的 JSON 内容")
            data = data[0]
        return PlanAction.model_validate(data)
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to parse plan response: {e}")
        logger.warning(f"Extracted JSON: {raw[:200]}")
        raise ValueError(f"无法解析 AI 返回的 JSON: {e}") from e
