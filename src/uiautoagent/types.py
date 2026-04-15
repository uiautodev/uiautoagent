"""通用类型定义"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """AI 调用 token 消耗"""

    prompt: int = Field(ge=0, description="输入 token 数")
    completion: int = Field(ge=0, description="输出 token 数")
    total: int = Field(ge=0, description="总 token 数")
