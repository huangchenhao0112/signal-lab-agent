# NCSD-Net 检查点角色

## 冻结主测试模型

```text
outputs/checkpoints/NCSD-Net/frozen_test/best_primary.pth
```

用途包括冻结测试集主实验、hit rate、稠密重建指标和 GFSC@K。文件大小为 3,991,622 bytes。已核查的 SHA-256 摘要以 `3C04` 开头、以 `BE` 结尾；正式部署必须保存和校验完整哈希。

冻结主模型不能被 MATLAB 应用检查点自动替代。

## 严格峰值检测器

```text
outputs/checkpoints/NCSD-Net/strict_peak_detector/best_peak_detector.pth
```

该检查点用于严格 Peak Precision、Peak Recall 和 Peak F1 检测。文件大小为 95,557 bytes，已核查的 SHA-256 摘要以 `88AA` 开头、以 `C8` 结尾。

严格峰值检测器依赖冻结基础时频模型，不能单独作为完整 NCSD-Net 时频模型运行。

## MATLAB 应用模型

```text
outputs/checkpoints/NCSD-Net/matlab_application/best_peak_recall.pth
```

该检查点用于 MATLAB 可控信号应用实验。文件大小为 6,009,551 bytes，已核查的 SHA-256 摘要以 `60D2` 开头、以 `A4` 结尾。

MATLAB 应用检查点不能用于宣称冻结测试集主结果。

## 使用原则

Agent 可以选择任务类型，但不能允许用户自由传入任意检查点路径。任务类型与检查点的映射应由受控后端固定。结构消融、损失消融、冻结主模型、MATLAB 应用模型和峰值检测器承担不同角色，不能互换。
