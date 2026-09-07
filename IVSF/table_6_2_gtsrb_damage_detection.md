# 表6-2 在6-layer CNN（GTSRB）上模型损坏的检测准确率（%）

采用 BadNet、TrojanNN 和 CleanLabel 构造后门篡改模型；Attack4 对模型参数执行 15% 幅度剪枝；Attack5 以 $10^{-5}$ 的学习率仅微调最后分类层。所有方法均使用 15 枚 GTSRB 指纹。

对于第 $i$ 枚指纹，若攻击前后的预测标签不同，或其局部互信息偏差超过阈值 $\tau=0.05$，则记为检测成功：

$$
d_i=\mathbb{I}\left[\hat{y}^{clean}_i\ne\hat{y}^{attack}_i\ \lor\ \left|MI^{clean}_i-MI^{attack}_i\right|>\tau\right].
$$

NS=1 的检测准确率为 $100\cdot\sum_i d_i/N$。NS=5 是从 $N=15$ 枚指纹中无放回随机抽取 5 枚、至少检测到一次模型损坏的概率：

$$
Acc_{NS=5}=100\cdot\left(1-\frac{\binom{N-\sum_i d_i}{5}}{\binom{N}{5}}\right).
$$

| Attack Types / NS | MiSentry 1 | MiSentry 5 | SSF 1 | SSF 5 | PublicCheck 1 | PublicCheck 5 | VMIISF 1 | VMIISF 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BadNet | 0.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | 33.3 | 91.6 |
| TrojanNN | 0.0 | 0.0 | 73.3 | 100.0 | 0.0 | 0.0 | 40.0 | 95.8 |
| CleanLabel | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Attack4 (Pruning) | 0.0 | 0.0 | 40.0 | 95.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| Attack5 (Fine-tuning) | 0.0 | 0.0 | 26.7 | 84.6 | 0.0 | 0.0 | 0.0 | 0.0 |

本次运行中，VMIISF 对 BadNet 与 TrojanNN 的 NS=5 检测率分别达到 91.6% 和 95.8%，但对 Attack4 和 Attack5 的检测率为 0%。因此，该配置尚不能支持“VMIISF 在轻微模型变化中优于所有基线”的结论；应通过增加指纹规模、重新校准 $\tau$ 或加强针对细粒度参数漂移的优化目标后再进行重复实验。

攻击训练配置：BadNet、TrojanNN 和 CleanLabel 均训练 3 轮，投毒比例为 10%；Attack4 执行 15% 剪枝；Attack5 仅微调最后分类层 3 轮。原始统计结果见 `./outputs/gtsrb_damage_detection/gtsrb_damage_detection_results.csv`，攻击模型权重见同目录的各攻击子目录。