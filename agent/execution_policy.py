from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Callable
import json
import time


# 可以自动执行，且失败后允许重试的只读工具。
READ_ONLY_TOOLS = {
    "search_research_docs",
    "get_model_metrics",
    "compare_models",
    "get_experiment_config",
}

# 已有绘图工具会写文件，因此不自动重试。
# 前提：绘图工具使用新文件名，不覆盖已有文件。
CREATE_ONLY_TOOLS = {
    "generate_metric_plot",
}

# 这里只定义权限规则，不代表已经注册这些真实工具。
CONFIRM_REQUIRED_TOOLS = {
    "start_training",
    "delete_file",
    "overwrite_file",
}

RETRYABLE_ERRORS = {
    "timeout",
    "network_error",
    "rate_limit",
    "service_unavailable",
}

VALIDATION_ERRORS = {
    "validation_error",
    "argument_validation_error",
    "invalid_arguments",
}

# 只修正有明确对应关系的名称，不猜实验数据。
METRIC_ALIASES = {
    # hit_rate
    "hitrate": "hit_rate",
    "hit rate": "hit_rate",
    "hit-rate": "hit_rate",
    "命中率": "hit_rate",

    # weighted_l2
    "weightedl2": "weighted_l2",
    "weighted l2": "weighted_l2",
    "weighted-l2": "weighted_l2",
    "加权l2": "weighted_l2",
    "加权l2误差": "weighted_l2",

    # mse
    "mse": "mse",
    "mean squared error": "mse",
    "均方误差": "mse",

    # psnr
    "psnr": "psnr",
    "peak signal-to-noise ratio": "psnr",
    "峰值信噪比": "psnr",

    # Renyi entropy
    "renyi": "renyi",
    "rényi": "renyi",
    "renyi entropy": "renyi",
    "rényi entropy": "renyi",
    "renyi_entropy": "renyi",
    "rényi_entropy": "renyi",
    "renyi熵": "renyi",
}



def console_confirm(
    name: str,
    arguments: dict[str, Any],
) -> bool:
    """仅供本地CLI运行使用，不能放进Web请求中等待输入。"""
    print("\n需要人工确认")
    print("工具：", name)
    print(
        "参数：",
        json.dumps(
            arguments,
            ensure_ascii=False,
            indent=2,
        ),
    )

    answer = input("输入 YES 才执行，其他输入均取消：")
    return answer.strip() == "YES"


def repair_metric_alias(
    name: str,
    arguments: dict[str, Any],
    error: dict[str, Any],
) -> dict[str, Any] | None:
    """修正明确的指标别名，不补全缺失信息。"""
    metric = arguments.get("metric")

    if not isinstance(metric, str):
        return None

    corrected_metric = METRIC_ALIASES.get(
        metric.strip().lower()
    )

    if corrected_metric is None:
        return None

    corrected = deepcopy(arguments)
    corrected["metric"] = corrected_metric

    if corrected == arguments:
        return None

    return corrected


def _to_dict(result: Any) -> dict[str, Any]:
    """兼容现有工具返回的dict、dataclass或Pydantic对象。"""
    if isinstance(result, dict):
        return dict(result)

    if is_dataclass(result):
        return asdict(result)

    if hasattr(result, "model_dump"):
        return result.model_dump()

    raise TypeError(
        "工具返回值必须是dict、dataclass或Pydantic对象"
    )


def _failure(
    name: str,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool_name": name,
        "output": None,
        "error_type": error_type,
        "error_message": message,
    }


def _call_registry(
    registry: Any,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = registry.execute(
            name=name,
            arguments=deepcopy(arguments),
        )
        return _to_dict(result)

    except TimeoutError as error:
        return _failure(name, "timeout", str(error))

    except ConnectionError as error:
        return _failure(name, "network_error", str(error))

    except Exception as error:
        # 不把所有异常都当成网络错误。
        return _failure(
            name,
            "tool_execution_error",
            f"{type(error).__name__}: {error}",
        )


def execute_with_policy(
    registry: Any,
    name: str,
    arguments: dict[str, Any],
    *,
    confirm: Callable[
        [str, dict[str, Any]], bool
    ] | None = None,
    repair: Callable[
        [str, dict[str, Any], dict[str, Any]],
        dict[str, Any] | None,
    ] = repair_metric_alias,
    max_retries: int = 2,
    retry_delay: float = 0.5,
) -> dict[str, Any]:
    """对现有ToolRegistry增加权限、修正和有限重试。"""
    if not 0 <= max_retries <= 2:
        raise ValueError("max_retries必须在0到2之间")

    if retry_delay < 0:
        raise ValueError("retry_delay不能为负数")

    events: list[dict[str, Any]] = []
    current_arguments = deepcopy(arguments)

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        return {
            **result,
            "execution_trace": list(events),
        }

    allowed_tools = (
        READ_ONLY_TOOLS
        | CREATE_ONLY_TOOLS
        | CONFIRM_REQUIRED_TOOLS
    )

    if name not in allowed_tools:
        events.append({"event": "blocked"})
        return finish(
            _failure(
                name,
                "permission_denied",
                "工具不在允许执行的列表中",
            )
        )

    if name in CONFIRM_REQUIRED_TOOLS:
        approved = (
            confirm is not None
            and confirm(name, deepcopy(current_arguments)) is True
        )

        events.append({
            "event": "approval",
            "approved": approved,
        })

        if not approved:
            return finish(
                _failure(
                    name,
                    "approval_required",
                    "未获得人工确认，工具未执行",
                )
            )

    repaired = False
    retries = 0

    while True:
        result = _call_registry(
            registry,
            name,
            current_arguments,
        )

        events.append({
            "event": "execute",
            "arguments": deepcopy(current_arguments),
            "ok": result.get("ok") is True,
            "error_type": result.get("error_type"),
        })

        if result.get("ok") is True:
            return finish(result)

        error_type = result.get("error_type")

        # 只读操作才自动修正一次。
        # 修正后仍由原ToolRegistry进行参数校验。
        if (
            name in READ_ONLY_TOOLS
            and error_type in VALIDATION_ERRORS
            and not repaired
        ):
            repaired = True

            corrected = repair(
                name,
                deepcopy(current_arguments),
                result,
            )

            if corrected is not None:
                if not isinstance(corrected, dict):
                    raise TypeError("修正函数必须返回dict或None")

                events.append({
                    "event": "repair",
                    "before": deepcopy(current_arguments),
                    "after": deepcopy(corrected),
                })

                current_arguments = deepcopy(corrected)
                continue

        # 只读工具遇到明确的临时错误才重试。
        if (
            name in READ_ONLY_TOOLS
            and error_type in RETRYABLE_ERRORS
            and retries < max_retries
        ):
            retries += 1

            events.append({
                "event": "retry",
                "retry_number": retries,
            })

            time.sleep(
                retry_delay * (2 ** (retries - 1))
            )
            continue

        return finish(result)