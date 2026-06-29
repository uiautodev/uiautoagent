"""环境变量管理 - 统一入口"""

from __future__ import annotations

import os
import uuid
from functools import cached_property


def _get_env(key: str, default: str | None = None) -> str | None:
    """获取环境变量，优先使用 UIAUTO_ 前缀版本"""
    return os.getenv(f"UIAUTO_{key}", os.getenv(key, default))


class EnvConfig:
    """环境变量配置，通过 property 动态读取，支持运行时修改"""

    # ---- 通用 ----

    @cached_property
    def session_id(self) -> str:
        """会话 ID（首次访问后缓存）"""
        return os.getenv("SESSION_ID") or str(uuid.uuid4())

    # ---- AI 配置 ----

    @property
    def base_url(self) -> str:
        return (
            _get_env("BASE_URL", "https://api.openai.com/v1")
            or "https://api.openai.com/v1"
        )

    @property
    def api_key(self) -> str | None:
        return _get_env("API_KEY")

    @property
    def model_name(self) -> str | None:
        return _get_env("MODEL_NAME")

    @property
    def model_vision(self) -> str | None:
        return _get_env("MODEL_VISION")

    @property
    def model_text(self) -> str | None:
        return _get_env("MODEL_TEXT")

    @property
    def model_proxy(self) -> str | None:
        return _get_env("MODEL_PROXY")

    @property
    def request_timeout(self) -> int:
        return int(_get_env("REQUEST_TIMEOUT", "60") or "60")

    # ---- OpenRouter 配置 ----

    @property
    def openrouter_site_url(self) -> str | None:
        return os.getenv("OPENROUTER_SITE_URL")

    @property
    def openrouter_site_name(self) -> str | None:
        return os.getenv("OPENROUTER_SITE_NAME")

    # ---- Agent 配置 ----

    @property
    def report_dir(self) -> str | None:
        return _get_env("REPORT_DIR")

    @property
    def step_wait_ms(self) -> int:
        return int(_get_env("STEP_WAIT_MS", "1000") or "1000")

    @property
    def recordings_dir(self) -> str | None:
        return _get_env("RECORDINGS_DIR")


# 全局单例
env = EnvConfig()
