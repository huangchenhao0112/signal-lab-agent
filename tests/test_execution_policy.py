from dataclasses import dataclass
from typing import Any

import pytest

from agent.execution_policy import execute_with_policy


@dataclass
class FakeResult:
    ok: bool
    tool_name: str = "get_model_metrics"
    output: Any = None
    error_type: str | None = None
    error_message: str | None = None


class FakeRegistry:
    """记录调用，按顺序返回设定结果。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append({
            "name": name,
            "arguments": dict(arguments),
        })

        if not self.results:
            raise AssertionError("工具被多调用了")

        result = self.results.pop(0)

        if isinstance(result, Exception):
            raise result

        return result


def success():
    return FakeResult(
        ok=True,
        output={"value": 0.9},
    )


def failure(error_type):
    return FakeResult(
        ok=False,
        error_type=error_type,
        error_message="模拟错误",
    )


# 共10条异常恢复测试。
@pytest.mark.parametrize(
    "scenario",
    [
        "direct_success",
        "timeout_then_success",
        "network_then_success",
        "rate_limit_then_success",
        "unavailable_then_success",
        "retry_exhausted",
        "auth_no_retry",
        "repair_alias",
        "missing_argument_no_guess",
        "repair_only_once",
    ],
)
def test_recovery(scenario):
    arguments = {"metric": "hit_rate"}
    repair = None

    if scenario == "direct_success":
        results = [success()]
        expected_calls = 1
        expected_ok = True

    elif scenario in {
        "timeout_then_success",
        "network_then_success",
        "rate_limit_then_success",
        "unavailable_then_success",
    }:
        errors = {
            "timeout_then_success": "timeout",
            "network_then_success": "network_error",
            "rate_limit_then_success": "rate_limit",
            "unavailable_then_success": "service_unavailable",
        }
        results = [
            failure(errors[scenario]),
            success(),
        ]
        expected_calls = 2
        expected_ok = True

    elif scenario == "retry_exhausted":
        results = [
            failure("timeout"),
            failure("timeout"),
            failure("timeout"),
        ]
        expected_calls = 3
        expected_ok = False

    elif scenario == "auth_no_retry":
        results = [failure("authentication_error")]
        expected_calls = 1
        expected_ok = False

    elif scenario == "repair_alias":
        arguments = {"metric": "hitrate"}
        results = [
            failure("validation_error"),
            success(),
        ]
        expected_calls = 2
        expected_ok = True

    elif scenario == "missing_argument_no_guess":
        arguments = {}
        results = [failure("validation_error")]
        expected_calls = 1
        expected_ok = False

    else:
        arguments = {"metric": "hitrate"}
        results = [
            failure("validation_error"),
            failure("validation_error"),
        ]
        expected_calls = 2
        expected_ok = False

    registry = FakeRegistry(results)

    result = execute_with_policy(
        registry,
        "get_model_metrics",
        arguments,
        retry_delay=0,
    )

    assert result["ok"] is expected_ok
    assert len(registry.calls) == expected_calls

    if scenario in {"repair_alias", "repair_only_once"}:
        assert registry.calls[1]["arguments"]["metric"] == "hit_rate"

        repairs = [
            item
            for item in result["execution_trace"]
            if item["event"] == "repair"
        ]
        assert len(repairs) == 1


# 共10条权限与副作用边界测试。
@pytest.mark.parametrize(
    "name, decision, expected_calls, expected_ok",
    [
        ("start_training", None, 0, False),
        ("delete_file", None, 0, False),
        ("overwrite_file", None, 0, False),
        ("start_training", False, 0, False),
        ("delete_file", False, 0, False),
        ("overwrite_file", False, 0, False),
        ("run_shell", True, 0, False),
        ("unknown_tool", True, 0, False),
        ("start_training", True, 1, True),
        ("generate_metric_plot", None, 1, False),
    ],
)
def test_permission_and_side_effects(
    name,
    decision,
    expected_calls,
    expected_ok,
):
    if name == "generate_metric_plot":
        results = [failure("timeout")]
    elif expected_calls:
        results = [success()]
    else:
        results = []

    registry = FakeRegistry(results)

    confirm = (
        None
        if decision is None
        else lambda tool_name, arguments: decision
    )

    result = execute_with_policy(
        registry,
        name,
        {"example": "test"},
        confirm=confirm,
        retry_delay=0,
    )

    assert len(registry.calls) == expected_calls
    assert result["ok"] is expected_ok