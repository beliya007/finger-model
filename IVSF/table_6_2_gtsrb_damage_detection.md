# 表6-2 在6-layer CNN（GTSRB）上模型损坏的检测准确率（%）

本表的 VMIISF 列使用 30 枚完整指纹，并按与 ResNet20（CIFAR-10）相同的概率边缘分布局部互信息规则检测。对干净模型和攻击模型分别计算：

$$
MI_i=\sum_c p_{i,c}\log\frac{p_{i,c}}{\bar p_c},\qquad \bar p_c=\frac{1}{30}\sum_{i=1}^{30}p_{i,c}.
$$

若预测标签变化，或 $|MI_i^{attack}-MI_i^{clean}|>\tau$，则该枚指纹检测成功。GTSRB 实验阈值设置为 $\tau=0.0003907681$（取自 `run_gtsrb_clean_repeat_check.py` 实测的干净模型自身计算噪声上限，即单样本前向与整批前向两条路径下同一指纹 MI 差值的最大值）；NS=1 为单枚指纹检测率，NS=5 为从 30 枚指纹中无放回抽取 5 枚时至少检测成功一次的概率。

| Attack Types / NS | MiSentry 1 | MiSentry 5 | SSF 1 | SSF 5 | PublicCheck 1 | PublicCheck 5 | VMIISF 1 | VMIISF 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BadNet | 85.3 | 100.0 | 63.1 | 98.3 | 69.2 | 99.3 | 100.0 | 100.0 |
| TrojanNN | 87.7 | 100.0 | 71.2 | 99.2 | 71.3 | 99.4 | 100.0 | 100.0 |
| CleanLabel | 89.3 | 100.0 | 68.5 | 100.0 | 83.2 | 100.0 | 100.0 | 100.0 |
| Attack4 (Pruning 15%) | 84.6 | 100.0 | 58.4 | 97.7 | 74.8 | 100.0 | 83.3 | 100.0 |
| Attack5 (Last-layer Fine-tuning) | 36.7 | 90.3 | 0.0 | 0.0 | 17.5 | 68.4 | 100.0 | 100.0 |

VMIISF 的实测命中数依次为 BadNet $30/30$、TrojanNN $30/30$、CleanLabel $30/30$、Attack4 $25/30$、Attack5 $30/30$。BadNet、TrojanNN、CleanLabel 和 Attack5 均采用 400 轮攻击训练，Attack4 对可训练参数执行 15% 幅度剪枝。五种攻击模型保存在 `./outputs/gtsrb_attack_models_400`，本次统计仅复用这些检查点。

VMIISF 原始结果见 `./outputs/gtsrb_damage_detection_ivsf30_taunoisefloor/gtsrb_damage_detection_results.csv`。MiSentry、SSF、PublicCheck 三组数值沿用现有表格数据；它们并非本次 GTSRB 运行重新计算的结果。