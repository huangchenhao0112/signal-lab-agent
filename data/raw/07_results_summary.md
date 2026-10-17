# NCSD-Net 主要实验结果摘要

## 冻结支撑恢复与重建质量

在 `FROZEN_SUPPORT_V1` 的 3500 条冻结样本中，NCSD-Net Overall hit rate 为 0.948271，TFA-Net 为 0.872569，HCTFR-Net 项目复现结果为 0.797715。

NCSD-Net 的支撑覆盖能力较好，但并非所有重建指标最优。HCTFR-Net 的 MSE 为 16.618754、PSNR 为 20.391590，优于 NCSD-Net 对应结果。因此不能声称 NCSD-Net 在全部指标上全面领先。

数据来源：`outputs/paper_figures/paper_ready/source_data/table_hctfr_frozen_protocol.csv`。

## 严格峰值检测

在 `FROZEN_STRICT_MAIN_V1` 的 Overall 分组中：

- NCSD-Net Peak F1：0.374176；
- TFA-Net Peak F1：0.322781；
- HCTFR-Net Peak F1：0.319155；
- NCSD-Net FP/frame：2.1935。

在固定阈值 0 的 Low-SNR 分组中：

- NCSD-Net Peak F1：0.190849；
- TFA-Net Peak F1：0.222030；
- HCTFR-Net Peak F1：0.229682。

这说明 NCSD-Net 在 Overall 分组表现较好，但低信噪比严格检测结果受阈值和置信度校准影响，不能根据 hit rate 直接推断 Peak F1。

数据来源：`outputs/paper_figures/paper_ready/source_data/table_hctfr_strict_peak_audited_per_sample.csv`。

## GFSC@K

在 Table S5 的五个 GFSC@K 分组中，NCSD-Net 均为最高，Overall 为 0.409332。该结果只说明全局频率支撑覆盖能力，不能称为 Peak F1 或 hit rate。

数据来源：`outputs/paper_figures/paper_ready/source_data/tableS5_global_frequency_support_coverage_clarified.csv`。

## MATLAB 可控信号实验

在 Table S6 的六个 MATLAB 场景中，NCSD-Net 获得 3 个第一、2 个第二和 1 个第三，并非所有场景领先。该实验使用 MATLAB 应用检查点，不能与冻结主测试直接比较。

数据来源：`outputs/paper_figures/paper_ready/source_data/table_matlab_strict_peak_extended_baselines_clean.csv`。

## 效率

NCSD-Net 参数量约为 0.481904 M。在 Python 3.10.20、PyTorch 2.9.0+cu128、CUDA 12.8、RTX 5090、batch=1 条件下，平均模型前向耗时约为 10.582457 ms/sample。

该耗时不包含上传、切窗、绘图、Agent和大模型延迟。Profiler 也可能未完整覆盖 FFT 或自定义算子。

数据来源：`outputs/results/model_efficiency.csv`。
