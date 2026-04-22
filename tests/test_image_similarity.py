"""测试图片相似度计算功能"""

import tempfile
from pathlib import Path

from PIL import Image
import numpy as np
import pytest

from uiautoagent.agent.image_similarity import (
    calculate_image_similarity,
    format_similarity_change,
)


def test_identical_images():
    """测试完全相同的图片"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建两张相同的图片
        img_path1 = Path(tmpdir) / "img1.png"
        img_path2 = Path(tmpdir) / "img2.png"

        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_path1)
        img.save(img_path2)

        similarity = calculate_image_similarity(img_path1, img_path2)
        assert similarity == pytest.approx(1.0, rel=0.01)


def test_different_images():
    """测试完全不同的图片"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path1 = Path(tmpdir) / "img1.png"
        img_path2 = Path(tmpdir) / "img2.png"

        # 创建两张完全不同的图片（纯色）
        img1 = Image.new("RGB", (100, 100), color=(0, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(255, 255, 255))

        img1.save(img_path1)
        img2.save(img_path2)

        similarity = calculate_image_similarity(img_path1, img_path2)
        # 黑白图片相似度应该很低
        assert similarity < 0.5


def test_similar_images():
    """测试相似的图片"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path1 = Path(tmpdir) / "img1.png"
        img_path2 = Path(tmpdir) / "img2.png"

        # 创建相似的图片（随机噪声）
        np.random.seed(42)
        arr1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        arr2 = arr1 + np.random.randint(-20, 20, (100, 100, 3), dtype=np.int16)
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)

        img1 = Image.fromarray(arr1)
        img2 = Image.fromarray(arr2)

        img1.save(img_path1)
        img2.save(img_path2)

        similarity = calculate_image_similarity(img_path1, img_path2)
        # 应该有较高的相似度
        assert similarity > 0.7


def test_format_similarity_change():
    """测试相似度变化信息格式化"""
    # 测试各种相似度级别
    assert "几乎没有变化" in format_similarity_change(0.98, "tap")
    assert "操作可能未生效" in format_similarity_change(0.98, "tap")

    assert "有轻微变化" in format_similarity_change(0.90, "swipe")
    assert "操作似乎已生效" in format_similarity_change(0.90, "swipe")

    assert "有明显变化" in format_similarity_change(0.75, "tap")

    assert "有很大变化" in format_similarity_change(0.60, "tap")

    assert "界面完全不同" in format_similarity_change(0.40, "tap")

    # 测试非交互操作
    result = format_similarity_change(0.50, "wait")
    assert "操作已完成" in result


def test_action_type_str_handling():
    """测试 action.type 为字符串时的处理"""
    from uiautoagent.agent.device_agent import Action, ActionType

    # 模拟从 JSON 解析的 Action（type 是字符串）
    action_dict = {
        "type": "tap",
        "thought": "点击按钮",
        "params": {"target": "搜索按钮"},
    }
    action = Action(**action_dict)

    # 验证 type 可能是字符串（由于 use_enum_values = True）
    # 实际上 Pydantic v2 会保持枚举类型，但为了兼容性我们仍然处理这种情况
    if isinstance(action.type, ActionType):
        action_type_str = action.type.value
    else:
        action_type_str = action.type

    assert action_type_str == "tap"

    # 验证 format_similarity_change 可以正确处理
    result = format_similarity_change(0.96, action_type_str)
    assert "几乎没有变化" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
