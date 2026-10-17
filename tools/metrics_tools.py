from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

EXPERIMENT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "experiment_data.json"
)


def _load_experiment_data() -> dict[str, Any]:
    """
    读取完整实验数据库。
    """

    if not EXPERIMENT_DATA_PATH.exists():
        raise FileNotFoundError(
            "找不到实验数据文件："
            f"{EXPERIMENT_DATA_PATH}"
        )

    try:
        with EXPERIMENT_DATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "experiment_data.json格式错误："
            f"{error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "experiment_data.json最外层"
            "必须是JSON对象"
        )

    return data


def _find_key(
    mapping: dict[str, Any],
    target: str,
    field_name: str,
) -> tuple[str, Any]:
    """
    不区分大小写查找字典 key。
    """

    cleaned_target = target.strip()

    for key, value in mapping.items():
        if (
            key.casefold()
            == cleaned_target.casefold()
        ):
            return key, value

    available = ", ".join(
        mapping.keys()
    )

    raise ValueError(
        f"不存在{field_name}：{target}。"
        f"可选值：{available or '无'}"
    )


def _get_model_data(
    data: dict[str, Any],
    model: str,
) -> tuple[str, dict[str, Any]]:
    """
    获取模型数据。
    """

    models = data.get(
        "models",
        {},
    )

    if not isinstance(
        models,
        dict,
    ):
        raise ValueError(
            "models字段必须是JSON对象"
        )

    actual_model, model_data = (
        _find_key(
            models,
            model,
            "模型",
        )
    )

    if not isinstance(
        model_data,
        dict,
    ):
        raise ValueError(
            f"{actual_model}数据格式错误"
        )

    return (
        actual_model,
        model_data,
    )


def _get_group_data(
    model_data: dict[str, Any],
    group: str,
) -> tuple[str, dict[str, Any]]:
    """
    获取指定模型的数据分组。
    """

    groups = model_data.get(
        "groups",
        {},
    )

    if not isinstance(
        groups,
        dict,
    ):
        raise ValueError(
            "groups字段必须是JSON对象"
        )

    actual_group, group_data = (
        _find_key(
            groups,
            group,
            "数据分组",
        )
    )

    if not isinstance(
        group_data,
        dict,
    ):
        raise ValueError(
            f"{actual_group}数据格式错误"
        )

    return (
        actual_group,
        group_data,
    )


def _get_metrics_dict(
    group_data: dict[str, Any],
) -> dict[str, Any]:
    """
    获取 group 下的 metrics。
    """

    metrics = group_data.get(
        "metrics",
        {},
    )

    if not isinstance(
        metrics,
        dict,
    ):
        raise ValueError(
            "metrics字段必须是JSON对象"
        )

    return metrics


def _find_metric_direction(
    data: dict[str, Any],
    metric: str,
) -> str | None:
    """
    查找指标优化方向。
    """

    directions = data.get(
        "metric_directions",
        {},
    )

    if not isinstance(
        directions,
        dict,
    ):
        return None

    for key, value in (
        directions.items()
    ):
        if (
            key.casefold()
            == metric.casefold()
        ):
            return value

    return None


def _infer_protocols(
    data: dict[str, Any],
    metric: str,
    group: str,
) -> list[dict[str, Any]]:
    """
    根据指标和分组寻找可能对应的协议。

    MATLAB 分组优先匹配 MATLAB_FIXED_S6；
    其他分组按 experiments.metrics 匹配。
    """

    experiments = data.get(
        "experiments",
        {},
    )

    if not isinstance(
        experiments,
        dict,
    ):
        return []

    matched: list[
        dict[str, Any]
    ] = []

    is_matlab_group = (
        group.casefold()
        .startswith("matlab_")
    )

    for protocol_id, config in (
        experiments.items()
    ):
        if not isinstance(
            config,
            dict,
        ):
            continue

        protocol_metrics = (
            config.get(
                "metrics",
                [],
            )
        )

        if (
            metric
            not in protocol_metrics
        ):
            continue

        if (
            is_matlab_group
            and protocol_id
            != "MATLAB_FIXED_S6"
        ):
            continue

        if (
            not is_matlab_group
            and protocol_id
            == "MATLAB_FIXED_S6"
        ):
            continue

        matched.append(
            {
                "protocol_id": (
                    protocol_id
                ),
                "name": config.get(
                    "name"
                ),
                "comparability": (
                    config.get(
                        "comparability"
                    )
                ),
                "warning": (
                    config.get(
                        "warning"
                    )
                ),
                "sources": (
                    config.get(
                        "sources",
                        [],
                    )
                ),
            }
        )

    return matched


def get_model_metrics(
    model: str,
    metric: str,
    group: str,
) -> dict[str, Any]:
    """
    查询单个模型在指定分组下
    的单个实验指标。
    """

    data = _load_experiment_data()

    actual_model, model_data = (
        _get_model_data(
            data,
            model,
        )
    )

    actual_group, group_data = (
        _get_group_data(
            model_data,
            group,
        )
    )

    metrics = _get_metrics_dict(
        group_data
    )

    actual_metric, value = (
        _find_key(
            metrics,
            metric,
            "指标",
        )
    )

    if value is None:
        raise ValueError(
            f"{actual_model}的"
            f"{actual_group}/"
            f"{actual_metric}"
            "尚未填写真实数值"
        )

    if not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            "指标值必须是数字，"
            f"实际得到：{value}"
        )

    direction = (
        _find_metric_direction(
            data,
            actual_metric,
        )
    )

    return {
        "ok": True,
        "tool": (
            "get_model_metrics"
        ),
        "data": {
            "model": actual_model,
            "group": actual_group,
            "metric": actual_metric,
            "value": float(value),
            "direction": direction,
            "protocols": (
                _infer_protocols(
                    data,
                    actual_metric,
                    actual_group,
                )
            ),
        },
    }


def list_model_metrics(
    model: str,
    group: str,
) -> dict[str, Any]:
    """
    查询某个模型 + 分组下
    实际有哪些指标。

    解决 Agent 自己猜指标的问题。
    """

    data = _load_experiment_data()

    actual_model, model_data = (
        _get_model_data(
            data,
            model,
        )
    )

    actual_group, group_data = (
        _get_group_data(
            model_data,
            group,
        )
    )

    metrics = _get_metrics_dict(
        group_data
    )

    metric_items: list[
        dict[str, Any]
    ] = []

    for metric, value in (
        metrics.items()
    ):
        metric_items.append(
            {
                "metric": metric,
                "value": value,
                "direction": (
                    _find_metric_direction(
                        data,
                        metric,
                    )
                ),
            }
        )

    return {
        "ok": True,
        "tool": (
            "list_model_metrics"
        ),
        "data": {
            "model": actual_model,
            "group": actual_group,
            "metrics": metric_items,
        },
    }


def get_metric_definition(
    metric: str,
) -> dict[str, Any]:
    """
    查询数据库内记录的指标定义。
    """

    data = _load_experiment_data()

    definitions = data.get(
        "metric_definitions",
        {},
    )

    if not isinstance(
        definitions,
        dict,
    ):
        raise ValueError(
            "metric_definitions字段格式错误"
        )

    actual_metric, definition = (
        _find_key(
            definitions,
            metric,
            "指标定义",
        )
    )

    return {
        "ok": True,
        "tool": (
            "get_metric_definition"
        ),
        "data": {
            "metric": actual_metric,
            "direction": (
                _find_metric_direction(
                    data,
                    actual_metric,
                )
            ),
            "definition": definition,
        },
    }


def get_protocol_info(
    metric: str,
    group: str,
) -> dict[str, Any]:
    """
    查询某指标 + 分组可能对应的
    实验协议和可比较性规则。
    """

    data = _load_experiment_data()

    protocols = _infer_protocols(
        data,
        metric,
        group,
    )

    return {
        "ok": True,
        "tool": (
            "get_protocol_info"
        ),
        "data": {
            "metric": metric,
            "group": group,
            "protocols": protocols,
            "global_warnings": (
                data.get(
                    "global_warnings",
                    [],
                )
            ),
        },
    }


def compare_models(
    model_a: str,
    model_b: str,
    metric: str,
    group: str,
) -> dict[str, Any]:
    """
    比较两个模型同一分组、
    同一指标。
    """

    result_a = get_model_metrics(
        model=model_a,
        metric=metric,
        group=group,
    )

    result_b = get_model_metrics(
        model=model_b,
        metric=metric,
        group=group,
    )

    data_a = result_a["data"]
    data_b = result_b["data"]

    value_a = data_a["value"]
    value_b = data_b["value"]

    difference = (
        value_a
        - value_b
    )

    direction = data_a.get(
        "direction"
    )

    better_model: str | None

    if value_a == value_b:
        better_model = "tie"

    elif direction == "higher":
        better_model = (
            data_a["model"]
            if value_a > value_b
            else data_b["model"]
        )

    elif direction == "lower":
        better_model = (
            data_a["model"]
            if value_a < value_b
            else data_b["model"]
        )

    else:
        better_model = None

    return {
        "ok": True,
        "tool": "compare_models",
        "data": {
            "model_a": data_a[
                "model"
            ],
            "model_b": data_b[
                "model"
            ],
            "group": data_a[
                "group"
            ],
            "metric": data_a[
                "metric"
            ],
            "value_a": value_a,
            "value_b": value_b,
            "difference_a_minus_b": (
                difference
            ),
            "direction": direction,
            "better_model": (
                better_model
            ),
            "protocols": (
                data_a.get(
                    "protocols",
                    [],
                )
            ),
        },
    }


def analyze_model_performance(
    models: list[str],
    group: str,
    metrics: list[str] | None = None,
    exclude_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """
    对多个模型在同一 group 下
    进行自动多指标分析。

    如果 metrics=None：
        自动寻找所有模型共同拥有的指标。

    如果某指标并非所有模型都有：
        不报整个任务失败，
        而是记录 missing_models。

    这是复杂模型比较的主 Tool。
    """

    if len(models) < 2:
        raise ValueError(
            "至少需要两个模型"
        )

    data = _load_experiment_data()

    exclude_set = {
        item.casefold()
        for item in (
            exclude_metrics or []
        )
    }

    model_metric_maps: dict[
        str,
        dict[str, Any],
    ] = {}

    actual_group: str | None = None

    # -------------------------
    # 收集每个模型的真实指标
    # -------------------------

    for model in models:
        actual_model, model_data = (
            _get_model_data(
                data,
                model,
            )
        )

        group_name, group_data = (
            _get_group_data(
                model_data,
                group,
            )
        )

        if actual_group is None:
            actual_group = (
                group_name
            )

        model_metric_maps[
            actual_model
        ] = _get_metrics_dict(
            group_data
        )

    actual_models = list(
        model_metric_maps.keys()
    )

    # -------------------------
    # union / common metrics
    # -------------------------

    metric_sets = [
        set(metric_map.keys())
        for metric_map
        in model_metric_maps.values()
    ]

    union_metrics = set().union(
        *metric_sets
    )

    common_metrics = set.intersection(
        *metric_sets
    )

    # -------------------------
    # 用户指定指标 or 自动共同指标
    # -------------------------

    if metrics:
        requested_metrics = [
            metric
            for metric in metrics
            if (
                metric.casefold()
                not in exclude_set
            )
        ]

    else:
        requested_metrics = sorted(
            metric
            for metric
            in common_metrics
            if (
                metric.casefold()
                not in exclude_set
            )
        )

    comparison_results: list[
        dict[str, Any]
    ] = []

    unavailable_results: list[
        dict[str, Any]
    ] = []

    # -------------------------
    # 每个指标独立分析
    # -------------------------

    for requested_metric in (
        requested_metrics
    ):
        values: dict[
            str,
            float,
        ] = {}

        missing_models: list[
            str
        ] = []

        actual_metric_name: (
            str | None
        ) = None

        for model_name, metric_map in (
            model_metric_maps.items()
        ):
            found = False

            for metric_name, value in (
                metric_map.items()
            ):
                if (
                    metric_name.casefold()
                    == requested_metric.casefold()
                ):
                    found = True

                    actual_metric_name = (
                        metric_name
                    )

                    if isinstance(
                        value,
                        (int, float),
                    ):
                        values[
                            model_name
                        ] = float(
                            value
                        )

                    else:
                        missing_models.append(
                            model_name
                        )

                    break

            if not found:
                missing_models.append(
                    model_name
                )

        if actual_metric_name is None:
            unavailable_results.append(
                {
                    "metric": (
                        requested_metric
                    ),
                    "reason": (
                        "该指标在所选模型"
                        "和分组中均不存在"
                    ),
                }
            )
            continue

        direction = (
            _find_metric_direction(
                data,
                actual_metric_name,
            )
        )

        winner: str | None = None

        # 只有所有模型都有该指标，
        # 才做完整横向 winner 判断
        if (
            not missing_models
            and values
        ):
            if direction == "higher":
                winner = max(
                    values,
                    key=values.get,
                )

            elif direction == "lower":
                winner = min(
                    values,
                    key=values.get,
                )

        comparison_results.append(
            {
                "metric": (
                    actual_metric_name
                ),
                "direction": (
                    direction
                ),
                "values": (
                    values
                ),
                "missing_models": (
                    missing_models
                ),
                "winner": winner,
                "fully_comparable": (
                    len(
                        missing_models
                    )
                    == 0
                ),
                "protocols": (
                    _infer_protocols(
                        data,
                        actual_metric_name,
                        actual_group or group,
                    )
                ),
            }
        )

    # -------------------------
    # 指出某些模型拥有、
    # 但并非全部模型共有的指标
    # -------------------------

    non_common_metrics = sorted(
        union_metrics
        - common_metrics
    )

    return {
        "ok": True,
        "tool": (
            "analyze_model_performance"
        ),
        "data": {
            "models": (
                actual_models
            ),
            "group": (
                actual_group
                or group
            ),
            "analysis_mode": (
                "requested_metrics"
                if metrics
                else "all_common_metrics"
            ),
            "common_metrics": sorted(
                common_metrics
            ),
            "non_common_metrics": (
                non_common_metrics
            ),
            "excluded_metrics": (
                exclude_metrics or []
            ),
            "comparisons": (
                comparison_results
            ),
            "unavailable": (
                unavailable_results
            ),
            "overall_judgement": (
                "不同指标可能来自不同协议，"
                "不得把多个异质指标直接"
                "合成为单一总分。"
                "应按各指标方向分别比较。"
            ),
            "global_warnings": (
                data.get(
                    "global_warnings",
                    [],
                )
            ),
        },
    }


def get_experiment_config(
    experiment_id: str,
) -> dict[str, Any]:
    """
    根据实验 ID 查询实验配置。
    """

    data = _load_experiment_data()

    experiments = data.get(
        "experiments",
        {},
    )

    if not isinstance(
        experiments,
        dict,
    ):
        raise ValueError(
            "experiments字段必须是JSON对象"
        )

    actual_id, config = (
        _find_key(
            experiments,
            experiment_id,
            "实验ID",
        )
    )

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"实验{actual_id}"
            "的配置必须是JSON对象"
        )

    return {
        "ok": True,
        "tool": (
            "get_experiment_config"
        ),
        "data": {
            "experiment_id": (
                actual_id
            ),
            "config": config,
        },
    }