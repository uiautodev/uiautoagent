"""基于AI视觉模型获取元素bbox位置的检测器"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Type, TypeVar

import json

from PIL import Image
from pydantic import BaseModel, ValidationError

from uiautoagent.ai import Category, chat_completion

_T = TypeVar("_T", bound=BaseModel)


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def __str__(self) -> str:
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, {self.width}x{self.height})"


class ElementLocation(BaseModel):
    thought: str | None = None
    found: bool
    bbox: list[int] | None
    description: str | None = None


class DetectionResult(BaseModel):
    found: bool
    bbox: BBox | None
    description: str | None = None
    thought: str | None = None


# --- 输出示例 ---
_example_found = ElementLocation(
    found=True,
    bbox=[100, 200, 300, 250],
    description="登录按钮",
    thought="在图片左上角发现了蓝色的登录按钮",
)
_example_not_found = ElementLocation(
    found=False, bbox=None, description=None, thought="图片中没有找到符合描述的元素"
)

_EXAMPLES = f"""
输出示例：
找到元素时：
{_example_found.model_dump_json(ensure_ascii=False)}

未找到元素时：
{_example_not_found.model_dump_json(ensure_ascii=False)}
"""

# 将示例加入系统提示
SYSTEM_PROMPT = f"""你是一个UI元素定位专家。用户会给你一张截图和需要查找的元素描述，你需要在图片中定位该元素并返回其边界框坐标。

假设图片尺寸统一为1000x1000，所有坐标均基于此尺寸给出。
请以JSON格式返回结果，包含你的思考过程(thought)。
如果找不到对应元素，found设为false，bbox设为null，并在thought中说明原因。

注：执行历史中也可能包含需要返回的结果，请仔细分析。

{_EXAMPLES}
"""


def safe_validate_json(
    raw: str | None,
    model_class: Type[_T],
    *,
    max_retries: int = 1,
) -> _T:
    """
    安全地解析并验证 JSON 模型，支持 AI 重新格式化。

    Args:
        raw: 原始 JSON 字符串
        model_class: Pydantic 模型类
        max_retries: AI 重新格式化的最大重试次数

    Returns:
        验证后的模型实例

    Raises:
        ValueError: raw 为空或 AI 格式化失败
    """
    if not raw or not raw.strip():
        raise ValueError("原始 JSON 字符串为空")

    # 尝试直接解析
    try:
        return model_class.model_validate_json(raw)
    except ValidationError:
        pass  # 继续尝试 AI 修复
    except json.JSONDecodeError:
        pass  # 继续尝试 AI 修复

    # AI 重新格式化
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    json_example = json.dumps(
        {
            k: (v.get("default") if "default" in v else None)
            for k, v in properties.items()
        },
        indent=2,
    )

    for attempt in range(max_retries + 1):
        try:
            response = chat_completion(
                category=Category.TEXT,
                messages=[
                    {
                        "role": "system",
                        "content": f"""你是一个 JSON 修复专家。用户会给你一个格式错误的 JSON 字符串，你需要将其修复为符合指定 schema 的有效 JSON。

目标 schema:
{json.dumps(schema, indent=2)}

示例格式:
{json_example}

要求:
1. 只返回修复后的 JSON 字符串，不要有任何额外说明
2. 确保所有必需字段都存在
3. 保持原有数据语义，只修复格式问题""",
                    },
                    {"role": "user", "content": f"请修复以下 JSON:\n\n{raw}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=2048,
                temperature=0.0,
            )
            fixed = response.choices[0].message.content
            if fixed:
                return model_class.model_validate_json(fixed)
        except (ValidationError, json.JSONDecodeError):
            if attempt < max_retries:
                continue
            raise

    raise ValueError(f"AI 格式化失败，无法解析为 {model_class.__name__}")


def _encode_image(image_source: str | Path) -> tuple[str, str]:
    """将图片编码为base64，返回 (base64_str, media_type)"""
    path = Path(image_source)
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")
    return base64.b64encode(path.read_bytes()).decode(), media_type


def detect_element(
    image_source: str | Path,
    query: str,
) -> DetectionResult:
    """
    在图片中检测指定元素并返回其bbox。

    Args:
        image_source: 图片路径
        query: 要查找的元素描述，如"登录按钮"、"搜索框"
    """
    b64, media_type = _encode_image(image_source)
    img = Image.open(image_source)
    w, h = img.size

    response = chat_completion(
        category=Category.DETECT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"请定位: {query}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=1024,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    print("Raw:", raw)
    loc = safe_validate_json(raw, ElementLocation)

    bbox = None
    if loc.found and loc.bbox:
        x1, y1, x2, y2 = loc.bbox
        bbox = BBox(
            x1=max(0, int(x1 * w / 1000)),
            y1=max(0, int(y1 * h / 1000)),
            x2=min(w, int(x2 * w / 1000)),
            y2=min(h, int(y2 * h / 1000)),
        )

    return DetectionResult(
        found=loc.found, bbox=bbox, description=loc.description, thought=loc.thought
    )


def draw_bbox(
    image_source: str | Path, result: DetectionResult, output: str | Path | None = None
) -> Image.Image:
    """在图片上绘制检测到的bbox"""
    from PIL import ImageDraw

    img = Image.open(image_source).convert("RGB")
    if result.bbox:
        draw = ImageDraw.Draw(img)
        b = result.bbox
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline="red", width=3)
        if result.description:
            draw.text((b.x1, b.y1 - 16), result.description, fill="red")

    if output:
        img.save(output)
    return img


# --- 多元素检测支持 ---
class MultiElementLocation(BaseModel):
    """多元素检测结果"""

    thought: str | None = None
    results: dict[str, ElementLocation]  # key: query, value: 检测结果


_multi_example = MultiElementLocation(
    thought="成功定位所有元素",
    results={
        "起始按钮": ElementLocation(
            found=True, bbox=[100, 200, 300, 250], description="起始按钮"
        ),
        "结束按钮": ElementLocation(
            found=True, bbox=[400, 500, 600, 550], description="结束按钮"
        ),
    },
)

_MULTI_EXAMPLES = f"""
输出示例（多元素检测）：
{_multi_example.model_dump_json(ensure_ascii=False)}
"""


def detect_elements(
    image_source: str | Path,
    queries: list[str],
) -> dict[str, DetectionResult]:
    """
    在图片中同时检测多个元素并返回其bbox。

    Args:
        image_source: 图片路径
        queries: 要查找的元素描述列表，如["起始按钮", "结束按钮"]

    Returns:
        dict[str, DetectionResult]: key是查询字符串，value是对应的检测结果
    """
    b64, media_type = _encode_image(image_source)
    img = Image.open(image_source)
    w, h = img.size

    # 构建查询文本
    query_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))

    response = chat_completion(
        category=Category.DETECT,
        messages=[
            {
                "role": "system",
                "content": f"""你是一个UI元素定位专家。用户会给你一张截图和多个需要查找的元素描述，你需要在图片中定位所有元素并返回各自的边界框坐标。

假设图片尺寸统一为1000x1000，所有坐标均基于此尺寸给出。
请以JSON格式返回结果，包含你的思考过程(thought)。
results字段中，key为元素描述，value为该元素的检测结果。
如果找不到对应元素，将对应元素的found设为false，bbox设为null。

{_MULTI_EXAMPLES}""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"请同时定位以下元素：\n{query_text}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=2048,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    print("Raw multi-detect:", raw)
    loc = safe_validate_json(raw, MultiElementLocation)

    # 转换结果
    final_results: dict[str, DetectionResult] = {}
    for query, element_loc in loc.results.items():
        bbox = None
        if element_loc.found and element_loc.bbox:
            x1, y1, x2, y2 = element_loc.bbox
            bbox = BBox(
                x1=max(0, int(x1 * w / 1000)),
                y1=max(0, int(y1 * h / 1000)),
                x2=min(w, int(x2 * w / 1000)),
                y2=min(h, int(y2 * h / 1000)),
            )

        final_results[query] = DetectionResult(
            found=element_loc.found,
            bbox=bbox,
            description=element_loc.description,
            thought=element_loc.thought,
        )

    return final_results
