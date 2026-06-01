"""基于AI视觉模型的图片内容提取器"""

from __future__ import annotations

import json
from pathlib import Path

import dictlog
from PIL import Image
from pydantic import BaseModel

from uiautoagent.ai import Category, chat_completion
from uiautoagent.detector.bbox_detector import _encode_image, safe_validate_json

log = dictlog.get_logger(__name__)


class ExtractionResult(BaseModel):
    """图片内容提取结果"""

    success: bool
    content: dict | list | None = None
    raw_response: str | None = None
    thought: str | None = None


class _ExtractionResponse(BaseModel):
    """AI 响应的内部模型，用于 safe_validate_json 解析"""

    thought: str | None = None
    content: dict | list


# --- 系统提示词 ---

_SYSTEM_FREE = """你是一个图片内容提取专家。用户会给你一张图片和一个查询请求，你需要根据查询从图片中提取相关内容，并以 JSON 格式返回结果。

请以如下 JSON 格式返回：
{
  "thought": "你的分析和推理过程",
  "content": { ... }
}

content 字段包含你提取的结构化内容，格式由你根据查询意图自主决定。
确保提取的信息准确、完整、结构清晰。
"""

_SYSTEM_WITH_EXAMPLE = """你是一个图片内容提取专家。用户会给你一张图片、一个查询请求和一个输出格式示例，你需要根据查询从图片中提取相关内容，并严格按照示例的格式输出。

请以如下 JSON 格式返回：
{
  "thought": "你的分析和推理过程",
  "content": { ... }
}

content 字段必须严格遵循用户提供的示例格式，保持相同的字段名和嵌套结构，用实际提取的内容填充示例中的占位值。

输出格式示例：
"""


def extract_content(
    image_source: str | Path | Image.Image,
    query: str,
    example: str | dict | list | None = None,
) -> ExtractionResult:
    """
    从图片中提取指定内容并返回结构化 JSON。

    Args:
        image_source: 图片文件路径或 PIL Image 对象
        query: 查询提示词，如 "提取所有价格信息"、"识别图中的表格数据"
        example: 可选的输出格式示例，支持 JSON 字符串/对象/数组。
                 传入标准 JSON 时 AI 严格按格式输出；
                 传入非 JSON 字符串时 AI 将其作为格式描述参考。
                 例如 '{"name": "商品名", "price": 0}' 或 '商品名, 价格'

    Returns:
        ExtractionResult 包含提取结果

    Raises:
        FileNotFoundError: 图片文件不存在（仅当传入文件路径时）
        ValueError: example 无法解析
    """
    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {path}")

    b64, media_type = _encode_image(image_source)

    # 构建系统提示词
    if example is not None:
        if isinstance(example, str):
            example_text = example
        else:
            example_text = json.dumps(example, ensure_ascii=False, indent=2)
        system_prompt = _SYSTEM_WITH_EXAMPLE + example_text
    else:
        system_prompt = _SYSTEM_FREE

    raw: str | None = None
    try:
        response = chat_completion(
            category=Category.VISION,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.0,
        )

        raw = response.choices[0].message.content
        log.debug("Raw extract response", raw=raw[:200] if raw else None)

        parsed = safe_validate_json(raw, _ExtractionResponse)
        return ExtractionResult(
            success=True,
            content=parsed.content,
            raw_response=raw,
            thought=parsed.thought,
        )

    except (ValueError, json.JSONDecodeError) as e:
        log.error("图片内容提取失败", error=str(e))
        return ExtractionResult(
            success=False,
            content=None,
            raw_response=raw,
            thought=None,
        )
