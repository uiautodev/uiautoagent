"""基于AI视觉模型获取元素bbox位置的检测器"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))


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
    found: bool
    bbox: list[int] | None
    description: str | None = None
    thought: str | None = None


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

{_EXAMPLES}
"""


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
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> DetectionResult:
    """
    在图片中检测指定元素并返回其bbox。

    Args:
        image_source: 图片路径
        query: 要查找的元素描述，如"登录按钮"、"搜索框"
        base_url: 覆盖环境变量中的BASE_URL
        api_key: 覆盖环境变量中的API_KEY
        model: 覆盖环境变量中的MODEL_NAME
    """
    key = api_key or API_KEY
    if not key:
        raise ValueError("API_KEY未设置，请在.env中配置或通过参数传入")

    client = OpenAI(
        base_url=base_url or BASE_URL,
        api_key=key,
        timeout=REQUEST_TIMEOUT,
    )
    model_name = model or MODEL_NAME

    b64, media_type = _encode_image(image_source)
    img = Image.open(image_source)
    w, h = img.size

    response = client.chat.completions.create(
        model=model_name,
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
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "element_location",
                "schema": ElementLocation.model_json_schema(),
            },
        },
        max_tokens=1024,
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    loc = ElementLocation.model_validate_json(raw)

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
