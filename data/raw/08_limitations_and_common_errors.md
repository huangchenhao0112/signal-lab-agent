# NCSD-Net 限制与常见错误

## 已知限制

- 固定阈值 0 时，NCSD-Net 的 Low-SNR 严格 Peak F1 低于 TFA-Net 和 HCTFR-Net；
- 在部分 MATLAB weak 场景中并非第一；
- HCTFR-Net 项目复现在冻结协议的 MSE、PSNR 和 Rényi entropy 指标上更好；
- 峰值置信度可能需要针对部署数据重新校准；
- 严格检测协议使用已知真实分量数 K，并限制候选数为 `2.5K`，不能直接描述成完全盲检测；
- 批量在线推理采用整批归一化还是逐样本归一化仍待确认；
- 任意长度信号的滑窗、重叠和拼接策略尚未形成唯一标准；
- 采样率、频率 bin 到 Hz 的映射和外部频率校准取决于应用端；
- HCTFR-Net 是项目内复现结果，不代表作者官方实现。

## 常见错误

### 混淆 hit rate、GFSC@K 与 Peak F1

0.948271 是冻结协议 Overall hit rate，0.409332 是 S5 Overall GFSC@K，0.374176 是严格峰值检测 Overall Peak F1。三个数值含义不同。

### 跨协议比较

不能用 MATLAB S6 场景的 Peak F1 与冻结测试集 Overall Peak F1 判断模型进步或退步。

### 把 FFT 当作最终输出

FFTReferenceBranch 只提供多窗口频谱参考。最终时频表示来自神经特征提取、跨尺度交互、融合和 Spectral Head。

### 使用错误检查点

冻结主结果使用 `best_primary.pth`；严格检测使用冻结基础模型加 `best_peak_detector.pth`；MATLAB 场景使用 `best_peak_recall.pth`。

### 分别归一化 I/Q

分别归一化会改变实部和虚部的相对幅度关系。I/Q 应共享同一个复幅值最大值。

### 擅自执行 fftshift

当前实现使用原生 FFT bin 顺序，未经确认执行 `fftshift` 会导致输出与标签错位。

### 只看 hit rate 宣称检测最好

hit rate 不充分惩罚误检。严格检测需要同时看 Precision、Recall、F1 和 FP/frame。

### 忽略阈值来源

Low-SNR 固定阈值 0 与验证集选定阈值 -10 的结果不是同一工作点，不能合并或取平均。

### 使用旧Word表格作为数值真值

规范化 CSV、JSON 和协议注册表应作为优先数值来源。Word 表格与规范数据冲突时，应报告冲突，不能静默覆盖。
