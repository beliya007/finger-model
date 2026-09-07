# 表6-4 GTSRB 数据集实验配置

GTSRB（German Traffic Sign Recognition Benchmark，德国交通标志识别基准）发布于 IJCNN 2011，是自动驾驶与交通标志分类中的经典数据集，也常用于后门攻击、鲁棒性和小样本学习实验。

## 数据集与模型配置

| 项目 | 设置 |
| --- | --- |
| 数据集 | GTSRB |
| 本地数据目录 | `./data/gtsrb`（torchvision 官方下载版本） |
| 训练集 | 39,209 张交通标志图像 |
| 测试集 | 12,630 张交通标志图像 |
| 图像尺寸 | $32 \times 32$ RGB |
| 类别数 | 43 |
| 目标模型 | 六层 CNN（`gtsrb_convnet`） |
| 卷积通道 | 32, 32, 64, 64, 128, 128 |
| 池化设置 | 每两个卷积层后进行 $2 \times 2$ 最大池化，共 3 次 |
| 分类层 | `Linear(4 * 4 * 128, 43)` |
| 输入归一化均值 | [0.3337, 0.3064, 0.3171] |
| 输入归一化标准差 | [0.2672, 0.2564, 0.2629] |
| 训练配置 | `train_gtsrb_six_layer_cnn.py`，Adam，初始学习率 0.001，训练 60 轮 |
| 最佳测试准确率 | 98.1552%（第 60 轮） |
| 模型权重 | `./checkpoints/gtsrb_six_layer_cnn.pth` |
| IVSF 运行命令 | `python run_ivsf_fingerprint.py --dataset gtsrb --checkpoint-path ./checkpoints/gtsrb_six_layer_cnn.pth --data-root ./data --device cuda` |

## GTSRB 时间开销对比

所有方法随机使用 100 个候选样本，保留 5 个基础样本，并为每个基础样本生成 3 枚指纹，共 15 枚。VMIISF 与三个基线均使用 220 次优化迭代，时间由 `time.perf_counter()` 在 CUDA 操作前后同步后记录。

| 数据集 | 方法 | 候选样本筛选（s） | 指纹生成（s） | 验证（s） | 总时间（s） |
| --- | --- | ---: | ---: | ---: | ---: |
| GTSRB | MiSentry | - | 16.577 | 0.010 | 16.588 |
| GTSRB | SSF | - | 13.810 | 0.001 | 13.811 |
| GTSRB | PublicCheck | - | 13.141 | 0.001 | 13.142 |
| GTSRB | VMIISF | 1.074 | 6.231 | 0.011 | 7.316 |

VMIISF 的候选样本筛选耗时为 1.074 s，但指纹生成耗时仅为 6.231 s，较生成最快的基线 PublicCheck 减少 52.6%。这是因为 VMIISF 仅对 5 个基础样本执行决策边界优化，再从局部扰动分布中采样出 15 枚指纹；三个基线则需对每枚指纹独立优化。基线方法没有 VMIISF 的候选样本评价模块，因此候选样本筛选列记为“-”。

对应原始结果位于 `./outputs/gtsrb_time_evaluation`，其中 `gtsrb_time_cost_results.csv` 和 `gtsrb_time_cost_results.md` 为自动汇总的对比报告。

## VMIISF 补充质量指标

为进一步评估指纹质量和隐蔽性，在 CUDA 环境下基于上述模型运行完整 IVSF 预设：从 300 个候选样本中筛选 10 个基础样本，每个基础样本生成 3 枚指纹，共 30 枚。优化步数为 220，$\epsilon_\infty=3/255$。

| 指标 | 结果 |
| --- | ---: |
| 基础样本预测正确数 | 10 / 10 |
| 生成指纹数 | 30 |
| 指纹预测标签保持率 | 30 / 30（100%） |
| 平均 MI | 1.858852 |
| 平均 JS 散度（相对基础样本） | 0.099821 |
| 平均 LPIPS（相对基础样本） | 0.000487 |
| 平均 $L_\infty$ 扰动 | 0.011763 |
| 平均 $L_2$ 扰动 | 0.203277 |
| 平均 PSNR | 49.09 dB |
| 候选样本筛选耗时 | 2.344 s |
| 指纹生成耗时（含边界优化） | 12.772 s |
| 验证耗时 | 0.020 s |
| 总耗时 | 15.136 s |

结果目录：`./outputs/gtsrb_table_6_4`。其中 `fingerprints.npz` 保存指纹，`verification_info.csv` 保存 MI、JS 和 LPIPS，`stealth_metrics.csv` 保存扰动隐蔽性指标，`timing_summary.json` 保存时间统计。