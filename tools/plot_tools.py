from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免在没有显示环境的情况下报错
import matplotlib.pyplot as plt

from tools.metrics_tools import (
    PROJECT_ROOT,
    get_model_metrics,
)


PLOT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "plots"
)


def _safe_filename(
    value: str,
) -> str:
    """
    删除文件名中的不安全字符。
    """

    cleaned = re.sub(
        r"[^0-9A-Za-z_-]+",
        "_",
        value,
    )

    return cleaned.strip("_") or "metric"


def generate_metric_plot(
    models: list[str],
    metric: str | None,
    group: str,
) -> dict[str, Any]:
    """
    为多个模型生成指标柱状图。

    Args:
        models:
            模型名称列表，至少包含一个模型。
        metric:
            指标名称，例如hit_rate。
        group:
            数据分组，例如low_snr。

    Returns:
        {
            "ok": True,
            "tool": "generate_metric_plot",
            "data": {
                "models": list[str],
                "values": list[float],
                "metric": str,
                "group": str,
                "plot_path": str
            }
        }

    Raises:
        ValueError:
            模型列表为空、模型重复，
            或指标没有真实数值。
    """

    if not isinstance(models, list):
        raise ValueError(
            "models必须是模型名称列表"
        )

    cleaned_models = [
        model.strip()
        for model in models
        if isinstance(model, str)
        and model.strip()
    ]

    if not cleaned_models:
        raise ValueError(
            "至少需要提供一个模型"
        )

    normalized_names = [
        model.casefold()
        for model in cleaned_models
    ]

    if len(normalized_names) != len(
        set(normalized_names)
    ):
        raise ValueError(
            "models中不能包含重复模型"
        )

    values: list[float] = []
    actual_models: list[str] = []
    direction: str | None = None

    for model in cleaned_models:
        result = get_model_metrics(
            model=model,
            metric=metric,
            group=group,
        )

        metric_data = result["data"]

        actual_models.append(
            metric_data["model"]
        )

        values.append(
            metric_data["value"]
        )

        if direction is None:
            direction = (
                metric_data.get(
                    "direction"
                )
            )

    PLOT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    safe_metric = _safe_filename(
        metric
    )

    safe_group = _safe_filename(
        group
    )

    output_path = (
        PLOT_OUTPUT_DIR
        / (
            f"{safe_metric}_"
            f"{safe_group}_"
            f"{timestamp}.png"
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    bars = axis.bar(
        actual_models,
        values,
        color="#4C78A8",
    )

    axis.set_title(
        f"{metric} comparison - {group}"
    )

    axis.set_xlabel("Model")
    axis.set_ylabel(metric)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    relative_path = output_path.resolve()

    return {
        "ok": True,
        "tool": "generate_metric_plot",
        "data": {
            "models": actual_models,
            "values": values,
            "metric": metric,
            "group": group,
            "direction": direction,
            "plot_path": str(
                relative_path
            ),
        },
    }