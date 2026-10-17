from __future__ import annotations

from types import SimpleNamespace

from agent.workflow.router import (
    create_router_node,
)


class FakeLLM:
    def __init__(
        self,
        result,
    ):
        self.result = result
        self.called = False

    def chat(
        self,
        messages,
        temperature=0.0,
    ):
        self.called = True
        return self.result


def _make_state(
    query: str,
) -> dict:
    return {
        "user_query": query,
        "conversation_summary": "",
        "conversation_history": [],
    }


def test_router_llm_failure():
    llm = FakeLLM(
        SimpleNamespace(
            ok=False,
            content=None,
            error_type="TimeoutError",
            error_message="LLM timeout",
        )
    )

    router = create_router_node(
        llm=llm,
        tools=[],
    )

    result = router(
        _make_state(
            "介绍NCSD-Net模型结构"
        )
    )

    assert result["route"] == "none"

    assert (
        "Router LLM 调用失败"
        in result["error"]
    )


def test_router_empty_response():
    llm = FakeLLM(
        SimpleNamespace(
            ok=True,
            content="",
            error_type=None,
            error_message=None,
        )
    )

    router = create_router_node(
        llm=llm,
        tools=[],
    )

    result = router(
        _make_state(
            "介绍NCSD-Net模型结构"
        )
    )

    assert result["route"] == "none"

    assert (
        "Router LLM 返回为空"
        in result["error"]
    )


def test_router_invalid_json():
    llm = FakeLLM(
        SimpleNamespace(
            ok=True,
            content="这不是JSON",
            error_type=None,
            error_message=None,
        )
    )

    router = create_router_node(
        llm=llm,
        tools=[],
    )

    result = router(
        _make_state(
            "介绍NCSD-Net模型结构"
        )
    )

    assert result["route"] == "none"

    assert (
        "Router 结果解析失败"
        in result["error"]
    )


def test_direct_metric_does_not_call_llm():
    llm = FakeLLM(
        SimpleNamespace(
            ok=True,
            content="{}",
            error_type=None,
            error_message=None,
        )
    )

    router = create_router_node(
        llm=llm,
        tools=[],
    )

    result = router(
        _make_state(
            "NCSD-Net在overall下"
            "的hit_rate是多少？"
        )
    )

    assert result["route"] == "metric"

    assert (
        result["metric_tool"]
        == "get_model_metrics"
    )

    assert llm.called is False