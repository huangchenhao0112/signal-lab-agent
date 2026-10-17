from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.agent_runtime import AgentRuntime
import pytest

pytestmark = pytest.mark.integration

def test_chat_without_token(
    client: TestClient,
):
    """
    未携带 JWT 时，
    受保护的 /agent/chat 应拒绝访问。
    """

    response = client.post(
        "/agent/chat",
        json={
            "message": "测试"
        },
    )

    assert response.status_code in {
        401,
        403,
    }


def test_chat_with_invalid_token(
    client: TestClient,
):
    """
    非法 JWT 应被拒绝。
    """

    response = client.post(
        "/agent/chat",
        json={
            "message": "测试"
        },
        headers={
            "Authorization":
            "Bearer invalid-token"
        },
    )

    assert response.status_code == 401


def test_chat_with_valid_token(
    client: TestClient,
    auth_headers: dict,
    monkeypatch,
):
    """
    合法 JWT 应通过鉴权。

    这里 Mock AgentRuntime.chat，
    避免真正调用 LangGraph、MCP、
    LLM、Redis 等完整业务链路。
    """

    mock_chat = AsyncMock(
        return_value={
            "ok": True,
            "session_id": "pytest-session",
            "route": "capability",
            "answer": "mock answer",
            "tools_used": [],
            "tool_plan": [],
            "trace_id": "pytest-trace",
            "latency_ms": 1,
        }
    )

    monkeypatch.setattr(
        AgentRuntime,
        "chat",
        mock_chat,
    )

    response = client.post(
        "/agent/chat",
        json={
            "message":
            "介绍一下你的能力"
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ok"] is True

    assert (
        data["answer"]
        == "mock answer"
    )

    mock_chat.assert_awaited_once()