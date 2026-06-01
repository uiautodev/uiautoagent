"""E2E integration tests for extract_content.

Requires a real VISION model API (configured via .env).
Run with: uv run pytest tests/e2e/test_extract_content_e2e.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from openai import InternalServerError, APIStatusError
from PIL import Image, ImageDraw, ImageFont

from uiautoagent.detector.content_extractor import ExtractionResult, extract_content

_has_api_key = any(
    os.getenv(k) for k in ("UIAUTO_API_KEY", "VISION_API_KEY", "OPENAI_API_KEY")
)

pytestmark = pytest.mark.skipif(
    not _has_api_key,
    reason="No API key configured — skipping e2e tests",
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def ensure_fixtures():
    """Create fixture images if they don't exist."""
    FIXTURES.mkdir(exist_ok=True)

    price_path = FIXTURES / "simple_price_tag.png"
    if not price_path.exists():
        img = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 32)
        except OSError:
            font = ImageFont.load_default()
        draw.text((30, 40), "苹果  ¥5.50", fill="black", font=font)
        draw.text((30, 100), "香蕉  ¥3.20", fill="black", font=font)
        draw.text((30, 160), "橙子  ¥4.80", fill="black", font=font)
        draw.rectangle([20, 25, 380, 220], outline="gray", width=2)
        img.save(price_path)


def test_extract_content_e2e():
    """E2E: 生成图片 → 调用真实 API → 验证结构化提取结果"""
    img = FIXTURES / "simple_price_tag.png"

    try:
        result = extract_content(img, "提取所有商品名称和价格")
    except (InternalServerError, APIStatusError) as e:
        pytest.xfail(f"Transient API error: {e}")

    assert isinstance(result, ExtractionResult)
    assert result.success is True
    assert result.content is not None
    assert result.thought is not None
    assert result.raw_response is not None

    content_str = json.dumps(result.content, ensure_ascii=False)
    print(
        f"\n[extract_content result]\nthought: {result.thought}\ncontent: {content_str}\n"
    )
    # AI 应该识别出图片中的文字内容
    assert result.thought is not None
    assert len(content_str) > 5
