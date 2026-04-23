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
    """空参数"""

    pass


class TapParams(BaseModel):
    """点击元素"""

    target: str = Field(..., description="目标元素描述，如'搜索按钮'")
    bbox: list[int] | None = Field(
        default=None,
        description="目标元素的边界框坐标 [x1, y1, x2, y2]，基于1000x1000归一化坐标系。提供后可跳过元素检测直接点击",
    )


class LongPressParams(BaseModel):
    """长按元素"""

    target: str = Field(..., description="目标元素描述")
    long_press_ms: int = Field(default=800, ge=0, description="长按毫秒数，默认800")
    bbox: list[int] | None = Field(
        default=None,
        description="目标元素的边界框坐标 [x1, y1, x2, y2]，基于1000x1000归一化坐标系。提供后可跳过元素检测直接长按",
    )


class InputParams(BaseModel):
    """输入文本"""

    text: str = Field(..., description="要输入的文本内容")


class SwipeParams(BaseModel):
    """滑动屏幕

    方式1：按方向滑动
    direction: "up" / "down" / "left" / "right"

    方式2：按坐标滑动
    swipe_start_xy: [x, y] 起始坐标
    swipe_end_xy: [x, y] 结束坐标

    direction 和 swipe_start_xy/swipe_end_xy 二选一
    """

    direction: Literal["up", "down", "left", "right"] | None = Field(
        default=None, description="滑动方向（up/down/left/right）"
    )
    swipe_start_xy: tuple[int, int] | None = Field(
        default=None,
        description="滑动起始坐标 [x, y]，基于1000x1000归一化坐标系",
    )
    swipe_end_xy: tuple[int, int] | None = Field(
        default=None,
        description="滑动结束坐标 [x, y]，基于1000x1000归一化坐标系",
    )


class WaitParams(BaseModel):
    """等待"""

    wait_ms: int = Field(default=1000, ge=0, description="等待毫秒数，默认1000")


class AppIdParams(BaseModel):
    """启动/停止/重启应用

    常用包名参考：
    - 微信：Android com.tencent.mm，iOS com.tencent.xin
    - QQ：Android com.tencent.mobileqq，iOS com.tencent.mqq
    - 抖音：Android com.ss.android.ugc.aweme，iOS com.ss.iphone.ugc.Aweme
    - 小红书：Android com.xingin.xhs，iOS com.xingin.discover
    - 支付宝：Android com.eg.android.AlipayGphone，iOS com.alipay.iphoneclient
    - 淘宝：Android com.taobao.taobao，iOS com.taobao.taobao4iphone
    - 哔哩哔哩：Android tv.danmaku.bili，iOS com.bilibili.app
    """

    app_id: str = Field(
        ...,
        description="应用包名（Android）或 Bundle ID（iOS），如 com.tencent.mm",
    )


class DoneParams(BaseModel):
    """任务完成"""

    return_result: bool = Field(default=False, description="是否返回观察结果")
    result: str | None = Field(default=None, description="任务返回的结果或答案")


class TaskProposal(BaseModel):
    """任务提案 - 记录原始输入和澄清后的任务"""

    original_task: str = Field(..., description="用户原始输入的任务描述")
    clarified_task: str = Field(..., description="AI澄清后的任务描述")
    timestamp: str = Field(
        default_factory=lambda: __import__("datetime").datetime.now().isoformat(),
        description="提案创建时间",
    )


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


# ========== Action ==========

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
            if self.params.swipe_start_xy and self.params.swipe_end_xy:
                return (
                    f"滑动: {self.params.swipe_start_xy} -> {self.params.swipe_end_xy}"
                )
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


def _generate_action_doc(params_cls: type[BaseModel]) -> str:
    """从 Params 模型定义生成单个操作的 params 文档"""
    lines = []
    # 包含完整 docstring（多行时作为说明文字）
    full_doc = (params_cls.__doc__ or "").strip()
    doc_lines = full_doc.split("\n")
    if len(doc_lines) > 1:
        # 多行 docstring：第一行是标题（已在 title 中），其余作为说明
        extra = "\n".join(line.strip() for line in doc_lines[1:] if line.strip())
        if extra:
            lines.append(extra)

    required_fields = []
    optional_fields = []

    for name, field_info in params_cls.model_fields.items():
        desc = field_info.description or name
        if field_info.is_required():
            required_fields.append((name, desc))
        else:
            default = field_info.default
            optional_fields.append((name, desc, default))

    if required_fields or optional_fields:
        lines.append("params 字段：")
        for name, desc in required_fields:
            lines.append(f"- {name}: {desc}")
        for name, desc, default in optional_fields:
            lines.append(f"- {name}: {desc}（可选，默认{default}）")
    else:
        lines.append("params 字段：无")

    return "\n".join(lines)


def get_action_examples_prompt() -> str:
    """从 ActionType 和 Params 模型定义自动生成操作类型说明"""

    sections = []
    for i, action_type in enumerate(ActionType, 1):
        params_cls = _ACTION_TYPE_TO_PARAMS[action_type]
        docstring = (params_cls.__doc__ or "").strip().split("\n")[0]
        title = f"{action_type.value} - {docstring}"
        doc = _generate_action_doc(params_cls)
        sections.append(f"{i}. {title}\n{doc}")

    body = "\n\n".join(sections)

    return f"""操作类型说明

所有操作都包含这四个字段：
- type: 操作类型（如 "tap", "swipe" 等）
- thought: 为什么执行这个操作
- log: 简洁说明要做的事情
- params: 操作参数（不同类型有不同参数）

{body}

使用说明：
1. 只包含必需字段，每个操作只需要包含它特有的字段
2. 省略默认值（如 long_press_ms 默认800，wait_ms 默认1000）
3. input 前置：输入前需要先用 tap 点击输入框
"""


def parse_plan_response(raw: str) -> Action:
    """
    解析 AI 返回的 JSON 为 Action

    使用 json_repair 自动处理 markdown 代码块和常见的 LLM 格式错误。

    Args:
        raw: JSON 字符串（可能被 markdown 代码块包裹）

    Returns:
        解析后的 Action 实例

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
        return Action.model_validate(data)
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to parse plan response: {e}")
        logger.warning(f"Extracted JSON: {raw[:200]}")
        raise ValueError(f"无法解析 AI 返回的 JSON: {e}") from e
