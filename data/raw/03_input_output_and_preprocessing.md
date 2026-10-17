# NCSD-Net 输入、输出与预处理

## 输入协议

模型默认输入张量形状为：

```text
[B, 2, L]
```

- `B` 表示 batch size；
- 第 0 通道是复信号实部 I；
- 第 1 通道是复信号虚部 Q；
- 默认 `L=256`。

输入端需要检查 NaN、Inf、全零信号、通道顺序、数据类型以及信号长度。单通道实信号不能在没有明确转换规则时直接作为模型输入。

## 输出协议

主输出形状为：

```text
[B, L, F_out]
```

默认输出为：

```text
[B, 256, 256]
```

Wrapper 可能返回 `output`、`tfr_pred`、`scale_weights`、`uncertainties`、`fft_weights`、`fft_prior_gain`、`fft_base`、`ncsd_map`、`ridge_mask`、`peak_map` 和 `peak_center_map` 等字段。

其中 `output` 或 `tfr_pred` 是稠密时频预测。辅助图、中间权重和峰值图不能在没有说明的情况下替代主输出。

## I/Q 归一化

冻结评估使用的主要归一化形式为：

```text
abs_max = max(sqrt(I^2 + Q^2))
I_norm = I / abs_max
Q_norm = Q / abs_max
```

I/Q 必须共享同一个复幅值最大值，不能分别归一化两个通道，否则会改变实部和虚部之间的相对幅度关系。

当 `B=1` 时，该方式可以视为单条请求归一化。对于批量在线推理，使用整批共享最大值还是逐样本最大值目前待确认，不能擅自选择后声称完全复现训练或冻结评估流程。

## 频率坐标

模型输出位于归一化频率网格。物理采样率以及频率 bin 到 Hz 的映射没有固定写死在模型内部，需要调用端提供采样率和经过确认的转换规则。

当前实现使用原生 FFT bin 顺序，不应未经确认自动执行 `fftshift`。
