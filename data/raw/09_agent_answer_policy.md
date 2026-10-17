# SignalLab Agent 回答规范

## 实验结果回答字段

当回答实验结果时，应尽可能返回：

```json
{
  "protocol": "FROZEN_STRICT_MAIN_V1",
  "model": "NCSD-Net",
  "group": "Overall",
  "sample_count": 3500,
  "metric": "Peak F1",
  "value": 0.374176,
  "direction": "higher_is_better",
  "aggregation": "mean_of_per_sample_metrics",
  "checkpoint_role": "frozen_primary + strict_peak_detector",
  "source": "outputs/paper_figures/paper_ready/source_data/table_hctfr_strict_peak_audited_per_sample.csv",
  "directly_comparable_with": [
    "TFA-Net and HCTFR-Net values from the same protocol and group"
  ],
  "not_directly_comparable_with": [
    "hit rate",
    "GFSC@K",
    "MATLAB S6 Peak F1",
    "low-SNR validation-selected threshold results"
  ],
  "uncertainty": "none"
}
```

## 不确定信息

如果字段在真实数据源中找不到，应返回：

```json
{
  "value": null,
  "uncertainty": "待确认"
}
```

Agent 不得根据相似实验、图像观感、其他指标或语言模型常识自行补数。

## 拒绝和澄清规则

- 如果用户要求跨协议直接比较，应指出协议不一致并要求选择相同协议；
- 如果同一指标存在多个阈值，应同时说明阈值和来源；
- 如果问题将 hit rate、GFSC@K 或 Peak F1 混用，应先解释区别；
- 如果问题要求提供物理 Hz 而没有采样率，应说明无法完成频率换算；
- 如果问题要求完全盲检测，但证据来自已知 K 的协议，应明确评估边界；
- 如果没有可靠来源，应回答“知识库中没有足够依据”，而不是猜测。
