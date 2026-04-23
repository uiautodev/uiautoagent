"""图片相似度计算模块 - 用于对比操作前后的屏幕变化"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np


def calculate_image_similarity(
    image_path1: str | Path, image_path2: str | Path
) -> float:
    """
    计算两张图片的相似度

    Args:
        image_path1: 第一张图片路径
        image_path2: 第二张图片路径

    Returns:
        相似度值，范围 0-1，1 表示完全相同，0 表示完全不同
    """
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    # 调整大小到较小图片的尺寸，确保尺寸一致
    min_size = (
        min(img1.width, img2.width),
        min(img1.height, img2.height),
    )
    img1 = img1.resize(min_size)
    img2 = img2.resize(min_size)

    return _calculate_similarity(img1, img2)


def _calculate_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """使用均方误差(MSE)计算相似度"""
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    # 计算均方误差
    mse = np.mean((arr1 - arr2) ** 2)

    # 将 MSE 转换为相似度 (0-1)
    # MSE=0 时相似度为1，MSE 越大相似度越低
    max_mse = 255**2  # 最大可能的 MSE
    similarity = np.exp(-mse / (max_mse * 0.1))  # 0.1 是调节系数

    return float(similarity)


def format_similarity_change(similarity: float, action_type: str) -> str:
    """
    格式化相似度变化信息，便于AI理解

    Args:
        similarity: 相似度值 (0-1)
        action_type: 操作类型

    Returns:
        格式化的描述文本
    """
    if similarity > 0.95:
        change_level = "几乎没有变化"
    elif similarity > 0.85:
        change_level = "有轻微变化"
    elif similarity > 0.7:
        change_level = "有明显变化"
    elif similarity > 0.5:
        change_level = "有很大变化"
    else:
        change_level = "界面完全不同"

    # 根据操作类型判断变化是否合理
    if action_type in ("tap", "long_press", "input", "swipe", "back"):
        if similarity > 0.95:
            assessment = "操作可能未生效或界面响应很慢"
        else:
            assessment = "操作似乎已生效"
    else:
        assessment = "操作已完成"

    return f"界面相似度: {similarity:.2%} ({change_level}) - {assessment}"
