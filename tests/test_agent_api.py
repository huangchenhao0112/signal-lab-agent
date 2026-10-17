from __future__ import annotations

import pytest

from fastapi.testclient import (
    TestClient,
)

def test_root(
    client: TestClient,
):
    response = client.get(
        "/"
    )

    assert (
        response.status_code
        == 200
    )

@pytest.mark.parametrize(
    "payload, expected_status",
    [
        (
            {"message": "介绍一下你的能力"},
            200,
        ),
        (
            {"message": ""},
            422,
        ),
        (
            {},
            422,
        ),
    ],
)
def test_agent_chat_parameter_validation(
    client,
    auth_headers,
    payload,
    expected_status,
):
    response = client.post(
        "/agent/chat",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == expected_status

def test_agent_health(
    client: TestClient,
):
    """
    Agent核心组件必须可用。
    """

    response = client.get(
        "/agent/health"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["ok"]
        is True
    )

    assert (
        data["mysql"]
        is True
    )

    assert (
        data["redis"]
        is True
    )

    assert (
        data["mcp"]
        is True
    )

    assert (
        data["graph"]
        is True
    )

    assert (
        data["tool_count"]
        > 0
    )

def test_agent_tools(
    client: TestClient,
):
    """
    MCP Tool列表应该可以获取。
    """

    response = client.get(
        "/agent/tools"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["ok"]
        is True
    )

    assert (
        data["count"]
        > 0
    )

    tool_names = {
        tool["name"]
        for tool
        in data["tools"]
    }

    assert (
        "get_model_metrics"
        in tool_names
    )

    assert (
        "search_research_docs"
        in tool_names
    )

    assert (
        "analyze_model_performance"
        in tool_names
    )

def test_agent_rag_query(
    client: TestClient,
    auth_headers: dict,
):
    response = client.post(
        "/agent/chat",
        json={
            "message": (
                "NCSD-Net的模型结构是什么？"
            )
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ok"] is True
    assert data["route"] == "rag"
    assert data["answer"]
    assert data["session_id"]
    assert data["trace_id"]
    assert data["latency_ms"] >= 0

    assert (
        "search_research_docs"
        in data["tools_used"]
    )


def test_agent_metric_query(
    client: TestClient,
    auth_headers: dict,
):
    response = client.post(
        "/agent/chat",
        json={
            "message": (
                "NCSD-Net在overall下"
                "的hit_rate是多少？"
            )
        },
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["ok"] is True
    assert data["route"] == "metric"

    assert (
        "get_model_metrics"
        in data["tools_used"]
    )

    assert (
        "0.948271"
        in data["answer"]
    )

def test_agent_memory(
    client: TestClient,
    auth_headers: dict,
):
    first_response = client.post(
        "/agent/chat",
        json={
            "message": (
                "请介绍NCSD-Net"
                "的模型结构。"
            )
        },
        headers=auth_headers,
    )

    assert (
        first_response.status_code
        == 200
    )

    first_data = (
        first_response.json()
    )

    session_id = (
        first_data["session_id"]
    )

    second_response = client.post(
        "/agent/chat",
        json={
            "session_id": session_id,
            "message": (
                "那它主要有哪些创新点？"
            ),
        },
        headers=auth_headers,
    )

    assert (
        second_response.status_code
        == 200
    )

    second_data = (
        second_response.json()
    )

    assert (
        second_data["session_id"]
        == session_id
    )

    assert second_data["answer"]

    delete_response = (
        client.delete(
            (
                "/agent/sessions/"
                f"{session_id}"
            ),
            headers=auth_headers,
        )
    )

    assert (
        delete_response.status_code
        == 200
    )

def test_agent_planner(
    client: TestClient,
    auth_headers: dict,
):
    response = client.post(
        "/agent/chat",
        json={
            "message": (
                "比较NCSD-Net、TFA-Net"
                "和HCTFR-Net在low_snr"
                "下的表现，解释原因，"
                "并生成peak_f1对比图。"
            )
        },
        headers=auth_headers,
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert data["ok"] is True

    assert (
        data["route"]
        == "plan"
    )

    assert (
        len(data["tool_plan"])
        >= 2
    )

    assert (
        len(data["tools_used"])
        >= 2
    )

    assert data["answer"]