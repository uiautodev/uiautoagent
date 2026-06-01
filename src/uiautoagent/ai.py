"""AI 客户端管理 - 统一 OpenAI 客户端初始化"""

from __future__ import annotations

import os
import uuid
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from threading import Lock
from typing import Any

import dictlog
import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

# 可重试的瞬态错误（换一个候选模型可能成功）
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# 模块级 logger
log = dictlog.get_logger(__name__)

# 当前进程的 session ID，用于 OpenRouter 等平台的请求追踪
SESSION_ID = os.getenv("SESSION_ID") or str(uuid.uuid4())


def _get_env(key: str, default: str | None = None) -> str | None:
    """
    获取环境变量，优先使用 UIAUTO_ 前缀版本

    Args:
        key: 环境变量名称（不含前缀）
        default: 默认值

    Returns:
        环境变量值，如果不存在则返回默认值

    Example:
        >>> _get_env("BASE_URL", "https://api.openai.com/v1")
        # 优先读取 UIAUTO_BASE_URL，如果不存在则读取 BASE_URL
    """
    return os.getenv(f"UIAUTO_{key}", os.getenv(key, default))


def _parse_model_list(value: str | None) -> list[str]:
    """解析逗号分隔的模型配置。"""
    if not value:
        return []
    return [model.strip() for model in value.split(",") if model.strip()]


class Category(str, Enum):
    """AI 调用场景分类

    用于区分不同用途的 AI 调用，便于统计 token 使用量和配置不同模型。
    """

    VISION = "vision"
    TEXT = "text"
    DEFAULT = "default"


# 不同场景的模型配置
_MODEL_CONFIG: dict[Category, list[str]] = {
    Category.VISION: _parse_model_list(_get_env("MODEL_VISION")),
    Category.TEXT: _parse_model_list(_get_env("MODEL_TEXT")),
}
_DEFAULT_MODELS = _parse_model_list(_get_env("MODEL_NAME")) or ["doubao-seed-2.0-pro"]


class TokenStats(BaseModel):
    """Token使用量统计"""

    prompt: int = Field(default=0, ge=0, description="输入token数量")
    completion: int = Field(default=0, ge=0, description="输出token数量")
    total: int = Field(default=0, ge=0, description="总token数量")

    def add(self, prompt: int, completion: int) -> None:
        """增加token数量"""
        self.prompt += prompt
        self.completion += completion
        self.total += prompt + completion


# 全局token统计（线程安全）
_token_stats: dict[str, TokenStats] = defaultdict(TokenStats)
_stats_lock = Lock()

# 最近一次记录的token信息
_last_record: TokenStats | None = None
_record_lock = Lock()


class TokenTracker:
    """Token使用量追踪器"""

    def __init__(self, category: Category | str = Category.DEFAULT):
        """
        初始化追踪器

        Args:
            category: 用途分类，如 Category.DECISION, Category.SUMMARIZE
        """
        self.category = category if isinstance(category, str) else category.value

    def record(self, response) -> TokenStats:
        """
        记录API响应的token使用量

        Args:
            response: OpenAI API响应对象

        Returns:
            token使用量TokenStats对象
        """
        usage = getattr(response, "usage", None)
        if usage:
            stats = TokenStats(
                prompt=usage.prompt_tokens,
                completion=usage.completion_tokens,
                total=usage.total_tokens,
            )

            with _stats_lock:
                _token_stats[self.category].add(stats.prompt, stats.completion)

            # 保存最近一次记录
            global _last_record
            with _record_lock:
                _last_record = stats

            return stats

        return TokenStats()

    @staticmethod
    def get_stats() -> dict[str, TokenStats]:
        """
        获取所有token统计

        Returns:
            按分类统计的token使用量
        """
        with _stats_lock:
            # 返回副本，避免外部修改
            return {k: TokenStats(**v.model_dump()) for k, v in _token_stats.items()}

    @staticmethod
    def get_total() -> TokenStats:
        """
        获取总token使用量

        Returns:
            总token使用量TokenStats对象
        """
        with _stats_lock:
            total = TokenStats()
            for stats in _token_stats.values():
                total.add(stats.prompt, stats.completion)
            return total

    @staticmethod
    def get_last_record() -> TokenStats | None:
        """
        获取最近一次记录的token信息

        Returns:
            最近一次的token使用量TokenStats对象，如果没有则返回None
        """
        with _record_lock:
            return _last_record

    @staticmethod
    def reset():
        """重置所有统计"""
        with _stats_lock:
            _token_stats.clear()
        global _last_record
        with _record_lock:
            _last_record = None


@lru_cache(maxsize=1)
def _get_ai_client() -> OpenAI:
    """
    获取 AI 客户端实例（单例模式，内部使用）

    注意：默认忽略系统代理，可通过 UIAUTO_MODEL_PROXY 环境变量指定代理

    Returns:
        OpenAI 客户端实例
    """
    timeout_str = _get_env("REQUEST_TIMEOUT", "60") or "60"
    timeout = float(timeout_str)
    proxy = _get_env("MODEL_PROXY")

    http_client = httpx.Client(trust_env=False, timeout=timeout, proxy=proxy)

    # OpenRouter 可选追踪头
    default_headers: dict[str, str] = {}
    if site_url := os.getenv("OPENROUTER_SITE_URL"):
        default_headers["HTTP-Referer"] = site_url
    if site_name := os.getenv("OPENROUTER_SITE_NAME"):
        default_headers["X-OpenRouter-Title"] = site_name

    return OpenAI(
        base_url=_get_env("BASE_URL", "https://api.openai.com/v1")
        or "https://api.openai.com/v1",
        api_key=_get_env("API_KEY"),
        http_client=http_client,
        timeout=timeout,
        default_headers=default_headers or None,
    )


def _normalize_model_candidates(model: str | list[str] | None) -> list[str]:
    """规范化候选模型列表。"""
    if model is None:
        return []
    if isinstance(model, str):
        return _parse_model_list(model)
    return _parse_model_list(",".join(model))


def get_ai_model(category: Category | str | None = None) -> list[str]:
    """
    获取 AI 候选模型列表

    Args:
        category: 可选的场景分类，用于获取特定场景的模型。
                  如果为 None，返回默认模型列表。
                  支持 Category 枚举值或对应的字符串值

    Returns:
        模型名称列表，如 ["gpt-4o", "gpt-4o-mini"]

    Example:
        >>> from uiautoagent.ai import get_ai_model, Category
        >>> get_ai_model()  # 获取默认模型列表
        ['gpt-4o']
        >>> get_ai_model(Category.VISION)  # 获取视觉场景的模型列表
        ['gpt-4o-mini', 'gpt-4o']
        >>> get_ai_model("vision")  # 也支持字符串
        ['gpt-4o-mini', 'gpt-4o']
    """
    if category:
        cat = category if isinstance(category, Category) else Category(category)
        models = _MODEL_CONFIG.get(cat)
        if models:
            return models.copy()
    return _DEFAULT_MODELS.copy()


def get_ai_config() -> dict:
    """
    获取 AI 配置信息

    Returns:
        包含 base_url, model, timeout 等配置的字典
    """
    return {
        "base_url": _get_env("BASE_URL", "https://api.openai.com/v1")
        or "https://api.openai.com/v1",
        "models": _DEFAULT_MODELS.copy(),
        "timeout": int(_get_env("REQUEST_TIMEOUT", "60") or "60"),
    }


def check_model_available(model: str) -> bool:
    """
    检查单个模型是否可用（发送最小请求实测，兼容不提供 /models 接口的代理）

    Args:
        model: 要检查的模型名称

    Returns:
        True 表示模型可用，False 表示不可用
    """
    try:
        client = _get_ai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
        )
        return bool(resp.choices and resp.choices[0].message is not None)
    except Exception as e:
        log.error(
            "Model unavailable", model=model, error=str(e), error_type=type(e).__name__
        )
        return False


def check_all_models_available() -> bool:
    """
    检查所有已配置场景是否至少存在一个可用模型

    Returns:
        True 表示所有场景都有可用模型，False 表示存在不可用场景
    """
    model_groups: dict[str, list[str]] = {"default": _DEFAULT_MODELS.copy()}
    for cat, models in _MODEL_CONFIG.items():
        if models:
            model_groups[cat.value] = models.copy()

    checked: dict[str, bool] = {}
    total_candidates = sum(len(models) for models in model_groups.values())
    log.info(
        f"检查模型可用性（共 {total_candidates} 个候选）...", total=total_candidates
    )

    all_ok = True
    for label, candidates in model_groups.items():
        group_ok = False
        for index, candidate in enumerate(candidates, start=1):
            if candidate not in checked:
                checked[candidate] = check_model_available(candidate)

            ok = checked[candidate]
            status = "✅" if ok else "❌"
            log.info(
                f"  {status} {candidate!r} [{label} #{index}]",
                model=candidate,
                label=label,
                index=index,
                status=status,
                available=ok,
            )
            if ok:
                group_ok = True
                break

        if not group_ok:
            all_ok = False
            log.error("当前场景没有可用模型", label=label, candidates=candidates)

    return all_ok


def chat_completion(
    category: Category,
    model: str | list[str] | None = None,
    **kwargs: Any,
) -> ChatCompletion:
    """
    调用 OpenAI Chat Completions API 并自动统计 token 使用量

    这是一个统一的 AI 调用入口，封装了 chat.completions.create 和 token 统计。
    所有需要调用 AI 的地方都应该使用这个函数。

    Args:
        category: 用途分类，用于 token 统计和模型选择。
                  如果配置了对应的环境变量（如 MODEL_VISION），将使用该场景的候选模型，
                  否则使用默认模型列表（MODEL_NAME）。
        model: 可选，显式指定模型。如果提供，将覆盖 category 的模型选择；
               支持单个模型、逗号分隔字符串或模型列表。
        **kwargs: 传递给 chat.completions.create 的所有参数，包括：
            - messages: 消息列表
            - max_tokens: 最大生成 token 数
            - temperature: 温度参数
            - response_format: 响应格式
            - 等等...

    Returns:
        ChatCompletion: API 响应对象
    """
    client = _get_ai_client()
    tracker = TokenTracker(category)
    model_candidates = (
        get_ai_model(category) if model is None else _normalize_model_candidates(model)
    )
    if not model_candidates:
        raise ValueError("未配置可用的模型")

    extra_body = kwargs.pop("extra_body", {}) or {}
    extra_body.setdefault("session_id", SESSION_ID)

    last_error: Exception | None = None
    for index, candidate in enumerate(model_candidates, start=1):
        try:
            response = client.chat.completions.create(
                model=candidate, extra_body=extra_body, **kwargs
            )
            tracker.record(response)
            return response
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if index < len(model_candidates):
                log.warning(
                    "Model call failed, trying next model",
                    model=model_candidates[index],
                    prev_model=candidate,
                    category=tracker.category,
                )

    raise last_error  # type: ignore[misc]
