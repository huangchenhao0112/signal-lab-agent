from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(
    client: TestClient,
):
    response = client.get(
        "/agent/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ok"] is True
    assert data["mysql"] is True
    assert data["redis"] is True
    assert data["mcp"] is True
    assert data["graph"] is True
    assert data["tool_count"] > 0