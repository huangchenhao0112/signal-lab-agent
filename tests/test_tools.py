import json
from dataclasses import dataclass

import pytest

from agent.basic_agent import BasicAgent
from agent.tool_factory import (
    create_signal_lab_tool_registry,
)
from tools import metrics_tools
from tools import plot_tools


@pytest.fixture
def tool_registry(
    tmp_path,
    monkeypatch,
):
    """
    为测试创建临时实验数据。

    这些数据只存在于pytest临时目录，
    不会进入正式项目数据。
    """

    experiment_data = {
        "metric_directions": {
            "hit_rate": "higher",
            "weighted_l2": "lower",
        },
        "models": {
            "Model-A": {
                "groups": {
                    "low_snr": {
                        "metrics": {
                            "hit_rate": 0.8,
                            "weighted_l2": 0.2,
                        }
                    }
                }
            },
            "Model-B": {
                "groups": {
                    "low_snr": {
                        "metrics": {
                            "hit_rate": 0.7,
                            "weighted_l2": 0.3,
                        }
                    }
                }
            },
        },
        "experiments": {
            "exp-test": {
                "description": (
                    "pytest临时实验"
                ),
                "models": [
                    "Model-A",
                    "Model-B",
                ],
                "group": "low_snr",
            }
        },
    }

    data_path = (
        tmp_path
        / "experiment_data.json"
    )

    data_path.write_text(
        json.dumps(
            experiment_data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        metrics_tools,
        "EXPERIMENT_DATA_PATH",
        data_path,
    )

    monkeypatch.setattr(
        plot_tools,
        "PLOT_OUTPUT_DIR",
        tmp_path / "plots",
    )

    def search_fn(
        question: str,
        top_k: int = 3,
    ):
        results = [
            {
                "chunk_id": "chunk-001",
                "content": (
                    "多窗口FFT提供不同"
                    "分辨率的参考证据。"
                ),
                "metadata": {
                    "source": "test.md"
                },
                "score": 0.9,
            },
            {
                "chunk_id": "chunk-002",
                "content": (
                    "FFT参考不是网络"
                    "最终输出。"
                ),
                "metadata": {
                    "source": "test.md"
                },
                "score": 0.8,
            },
        ]

        return results[:top_k]

    return create_signal_lab_tool_registry(
        search_fn=search_fn
    )


@pytest.mark.parametrize(
    ("arguments", "expected_ok"),
    [
        (
            {
                "model": "Model-A",
                "metric": "hit_rate",
                "group": "low_snr",
            },
            True,
        ),
        (
            {
                "model": "model-a",
                "metric": "HIT_RATE",
                "group": "LOW_SNR",
            },
            True,
        ),
        (
            {
                "model": "Unknown",
                "metric": "hit_rate",
                "group": "low_snr",
            },
            False,
        ),
        (
            {
                "model": "Model-A",
                "metric": "unknown_metric",
                "group": "low_snr",
            },
            False,
        ),
    ],
)
def test_get_model_metrics(
    tool_registry,
    arguments,
    expected_ok,
):
    result = tool_registry.execute(
        "get_model_metrics",
        arguments,
    )

    assert result.ok is expected_ok


@pytest.mark.parametrize(
    ("arguments", "expected_ok"),
    [
        (
            {
                "model_a": "Model-A",
                "model_b": "Model-B",
                "metric": "hit_rate",
                "group": "low_snr",
            },
            True,
        ),
        (
            {
                "model_a": "Model-A",
                "model_b": "Model-B",
                "metric": "weighted_l2",
                "group": "low_snr",
            },
            True,
        ),
        (
            {
                "model_a": "Model-A",
                "model_b": "Model-A",
                "metric": "hit_rate",
                "group": "low_snr",
            },
            True,
        ),
        (
            {
                "model_a": "Model-A",
                "model_b": "Model-B",
                "metric": "hit_rate",
                "group": "unknown_group",
            },
            False,
        ),
    ],
)
def test_compare_models(
    tool_registry,
    arguments,
    expected_ok,
):
    result = tool_registry.execute(
        "compare_models",
        arguments,
    )

    assert result.ok is expected_ok


@pytest.mark.parametrize(
    ("experiment_id", "expected_ok"),
    [
        ("exp-test", True),
        ("EXP-TEST", True),
        ("missing", False),
        ("", False),
    ],
)
def test_get_experiment_config(
    tool_registry,
    experiment_id,
    expected_ok,
):
    result = tool_registry.execute(
        "get_experiment_config",
        {
            "experiment_id": (
                experiment_id
            )
        },
    )

    assert result.ok is expected_ok


@pytest.mark.parametrize(
    ("arguments", "expected_ok"),
    [
        (
            {
                "question": (
                    "多窗口FFT有什么作用？"
                ),
                "top_k": 2,
            },
            True,
        ),
        (
            {
                "question": "",
                "top_k": 2,
            },
            False,
        ),
        (
            {
                "question": "FFT",
                "top_k": 20,
            },
            False,
        ),
    ],
)
def test_search_research_docs(
    tool_registry,
    arguments,
    expected_ok,
):
    result = tool_registry.execute(
        "search_research_docs",
        arguments,
    )

    assert result.ok is expected_ok


@pytest.mark.parametrize(
    ("arguments", "expected_ok"),
    [
        (
            {
                "models": ["Model-A"],
                "metric": "hit_rate",
                "group": "low_snr",
            },
            True,
        ),
        (
            {
                "models": [
                    "Model-A",
                    "Model-B",
                ],
                "metric": "hit_rate",
                "group": "low_snr",
            },
            True,
        ),
        (
            {
                "models": [
                    "Model-A",
                    "Model-A",
                ],
                "metric": "hit_rate",
                "group": "low_snr",
            },
            False,
        ),
        (
            {
                "models": ["Model-A"],
                "metric": "unknown",
                "group": "low_snr",
            },
            False,
        ),
    ],
)
def test_generate_metric_plot(
    tool_registry,
    arguments,
    expected_ok,
):
    result = tool_registry.execute(
        "generate_metric_plot",
        arguments,
    )

    assert result.ok is expected_ok


@dataclass
class FakeLLMResult:
    ok: bool
    content: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class FakeLLMClient:
    def __init__(self):
        self.responses = [
            FakeLLMResult(
                ok=True,
                content=json.dumps(
                    {
                        "thought": (
                            "需要查询指标"
                        ),
                        "action": "tool",
                        "tool_name": (
                            "get_model_metrics"
                        ),
                        "tool_input": {
                            "model": "Model-A",
                            "metric": "hit_rate",
                            "group": "low_snr",
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
            FakeLLMResult(
                ok=True,
                content=json.dumps(
                    {
                        "thought": (
                            "已经获得指标"
                        ),
                        "action": "finish",
                        "answer": (
                            "Model-A的hit_rate"
                            "为0.8。"
                        ),
                    },
                    ensure_ascii=False,
                ),
            ),
        ]

        self.index = 0

    def chat(
        self,
        messages,
        model=None,
        temperature=0.2,
    ):
        response = self.responses[
            self.index
        ]

        self.index += 1

        return response


def test_basic_agent_calls_real_tool_code(
    tool_registry,
):
    """
    第20条测试：

    FakeLLM负责选择工具，
    真实get_model_metrics代码负责查询。
    """

    agent = BasicAgent(
        llm_client=FakeLLMClient(),
        tool_registry=tool_registry,
        max_steps=5,
        max_tool_calls=3,
    )

    result = agent.run(
        "Model-A的hit_rate是多少？"
    )

    assert result.ok is True
    assert result.tool_call_count == 1
    assert result.steps[0].tool_ok is True
    assert "0.8" in (
        result.steps[0].observation
        or ""
    )