# NCSD-Net 指标与实验协议

## 指标区别

### hit rate

hit rate 衡量真实频率支撑是否落入预测候选频率附近，容差为 2 个频点。它主要反映目标覆盖能力，不充分惩罚误检，不能称为 Peak F1。

### GFSC@K

GFSC@K 将二维时频图沿时间维聚合，再取全局 Top-K 候选频率计算支撑覆盖率。它是全局频率支撑覆盖指标，不能与 hit rate 或 Peak F1 直接比较。

### Peak Precision、Recall 与 F1

这些指标来自严格峰值检测的一对一匹配。结果依赖 NMS、阈值、候选数、真实分量数和匹配容差。不同阈值和检测协议下的 Peak F1 不能直接比较。

### FP/frame 与 Localization MAE

FP/frame 是每帧平均误检峰值数，越小越好，应结合 Recall 或 F1 解释。Localization MAE 是成功匹配峰值的平均频点定位误差，只描述已匹配峰值的位置误差。

### 稠密重建指标

Weighted L2 和 MSE 越小越好；PSNR 越大越好；Rényi entropy 通常越小表示能量更集中。这些指标描述稠密时频图，不等价于峰值检测性能。

## 主要实验协议

### FROZEN_SUPPORT_V1

输出 hit rate、Weighted L2、MSE、PSNR 和 Rényi entropy。只允许在同协议、同分组、同指标下比较。

### FROZEN_STRICT_MAIN_V1

输出 Peak Precision、Peak Recall、Peak F1、FP/frame 和 Localization MAE。核心规则包括逐帧一维 NMS、最小峰间隔 4、Hungarian 一对一匹配、容差 2 个频点、已知真实分量数 K，以及最大候选数 `2.5K`。

### Frozen GFSC@K / Table S5

输出 GFSC@K，只能与同一表和同一分组中的 GFSC@K 结果比较。

### MATLAB fixed operating point / Table S6

输出场景级 Peak F1，只能在同一场景和同一工作点下比较，不能与冻结严格峰值结果直接横向比较。

### MATLAB same-FP

比较相同误检水平下的 Recall，属于曲线级比较，不等于固定阈值表格结果。

### 消融与机制干预

结构消融、正式因果消融和 FFT 删除干预只用于解释组件或目标函数贡献，不能替代冻结主实验，也不能外推到全部测试集。
