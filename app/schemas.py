from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    Field,
)


class AgentChatRequest(
    BaseModel
):
    message: str = Field(
        ...,
        min_length=1,
        description="用户自然语言问题",
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "会话ID，为空时自动创建"
        ),
    )


class AgentChatResponse(
    BaseModel
):
    ok: bool

    session_id: str

    answer: str

    route: str | None = None

    route_reason: str | None = None

    tools_used: list[str] = Field(
        default_factory=list
    )

    tool_plan: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    trace_id: str | None = None

    latency_ms: float | None = None


class AgentToolsResponse(
    BaseModel
):
    ok: bool

    count: int

    tools: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )


class AgentHealthResponse(
    BaseModel
):
    ok: bool

    mysql: bool

    redis: bool

    mcp: bool

    graph: bool

    tool_count: int


class AgentSessionResponse(
    BaseModel
):
    id: int

    session_id: str

    title: str | None

    created_at: datetime

    updated_at: datetime


class AgentSessionUpdateRequest(
    BaseModel
):
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )


class SessionDeleteResponse(
    BaseModel
):
    ok: bool

    session_id: str

    message: str