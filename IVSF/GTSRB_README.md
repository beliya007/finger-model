# GTSRB 实验代码与产物汇总

本目录汇总在 GTSRB 数据集下复现论文表 6-2 / 表 6-4 / 表 6-5 所需的全部实验脚本和产物。

## 1. 一键式流水线

- `run_gtsrb_full_pipeline.py` — 按顺序执行指纹生成、时间评估报告、隐蔽性对比报告

```bash
cd IVSF
python run_gtsrb_full_pipeline.py
```

可调参数：
- `--skip-fingerprints` 跳过指纹生成阶段
- `--skip-baselines` 不生成基线指纹
- `--device` 指定运行设备
- `--steps` 基线方法优化步数

## 2. 单步脚本

| 阶段 | 脚本 | 说明 |
| --- | --- | --- |
| 模型训练 | `train_gtsrb_six_layer_cnn.py` | 训练六层 CNN，目标模型 |
| VMIISF 指纹生成 | `run_ivsf_fingerprint.py --dataset gtsrb --time-evaluation` | VMIISF 的 time evaluation 配置（100 → 5 → 15） |
| 基线指纹生成 | `run_baseline_fingerprints.py --dataset gtsrb` | 同时支持 SSF / PublicCheck / MiSentry |
| 表 6-4 时间报告 | `create_gtsrb_timing_report.py` | 汇总 4 个方法的 timing_summary.json |
| 表 6-5 隐蔽性对比 | `run_gtsrb_method_comparison_report.py` | LPIPS + JS 对比 + 垂直对比图 |
| 表 6-2 损坏检测 | `run_gtsrb_damage_detection.py` | BadNet / TrojanNN / CleanLabel / 剪枝 / 微调下的检测率 |
| 干净模型一致性 | `run_gtsrb_clean_repeat_check.py` | 验证 MI 噪声上限（用于设置 τ 阈值） |

## 3. 产物检查

```bash
python check_gtsrb_artifacts.py
```

## 4. 关键实验产物

### 时间评估（表 6-4）

| Dataset | Method | Candidate Selection (s) | Fingerprint Generation (s) | Verification (s) | Total (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| GTSRB | MiSentry | - | 16.577 | 0.010 | 16.588 |
| GTSRB | SSF | - | 13.810 | 0.001 | 13.811 |
| GTSRB | PublicCheck | - | 13.141 | 0.001 | 13.142 |
| GTSRB | VMIISF | 1.074 | 6.231 | 0.011 | 7.316 |

VMIISF 较最快基线（PublicCheck）减少约 52.6% 指纹生成时间。

### LPIPS / JS 隐蔽性（表 6-5）

| Dataset | Method | LPIPS | JS |
| --- | --- | ---: | ---: |
| GTSRB | MiSentry | ~0 | ~0 |
| GTSRB | SSF | 0.008876 | 0.213612 |
| GTSRB | PublicCheck | 0.044961 | 0.692686 |
| GTSRB | VMIISF | 0.000305 | 0.104314 |

VMIISF 在保持高检测能力的前提下，输入与输出空间隐蔽性均优于 SSF / PublicCheck。

## 5. 文件位置索引

```
IVSF/
├── run_gtsrb_full_pipeline.py         # 一键式流水线
├── create_gtsrb_timing_report.py      # 表 6-4 CSV/Markdown 生成器
├── check_gtsrb_artifacts.py           # 产物检查工具
├── train_gtsrb_six_layer_cnn.py       # GTSRB 模型训练
├── run_gtsrb_damage_detection.py      # 表 6-2 模型损坏检测
├── run_gtsrb_method_comparison_report.py  # 表 6-5 隐蔽性对比
├── run_gtsrb_clean_repeat_check.py    # 干净模型 MI 噪声上限
├── outputs/
│   ├── gtsrb_time_evaluation/
│   │   ├── ivsf/                    # VMIISF 指纹 + 验证 + 隐蔽性
│   │   ├── misentry/, ssf/, publiccheck/  # 基线指纹
│   │   ├── gtsrb_time_cost_results.{csv,md}  # 表 6-4
│   │   └── ...
│   ├── gtsrb_method_comparison/
│   │   ├── method_lpips_js_comparison.{csv,md}
│   │   └── vertical_fingerprint_comparison_*.png
│   └── gtsrb_damage_detection/
│       ├── gtsrb_damage_detection_results.csv
│       └── table_6_2_gtsrb_damage_detection.md
```

## 6. 模型与数据依赖

- 训练好的模型：`checkpoints/gtsrb_six_layer_cnn.pth`（约 1.5 MB）
- GTSRB 数据：`./data/gtsrb/`（torchvision 官方下载版，含 39,209 / 12,630 张 32×32 RGB 图像）
- IVSF 核心代码包：`./ivsf_pipeline/`
- 攻击代码：`./attack/`（含 BadNet、TrojanNN、CleanLabel 实现）

## 7. 与 CIFAR-10 的对应关系

| 论文表 / 章节 | CIFAR-10 | GTSRB |
| --- | --- | --- |
| 模型训练 | `train_resnet20.py`（BIVBench 内） | `train_gtsrb_six_layer_cnn.py` |
| 时间评估报告（表 6-4） | `create_cifar10_timing_report.py` | `create_gtsrb_timing_report.py` |
| 方法对比报告（表 6-5） | `run_method_comparison_report.py` | `run_gtsrb_method_comparison_report.py` |
| 损坏检测（表 6-2） | `run_resnet20_damage_detection.py` | `run_gtsrb_damage_detection.py` |
| 干净模型一致性 | `run_resnet20_clean_repeat_check.py` | `run_gtsrb_clean_repeat_check.py` |

## 8. 复现实验的最小命令集

```bash
# 1. 模型训练（如已有 weights 可跳过）
python train_gtsrb_six_layer_cnn.py

# 2. VMIISF 指纹生成（time-evaluation 配置）
python run_ivsf_fingerprint.py --time-evaluation --dataset gtsrb \
    --checkpoint-path ./checkpoints/gtsrb_six_layer_cnn.pth \
    --output-dir ./outputs/gtsrb_time_evaluation/ivsf

# 3. 基线指纹生成（依次执行）
for m in misentry ssf publiccheck; do
    python run_baseline_fingerprints.py --methods $m --dataset gtsrb \
        --checkpoint-path ./checkpoints/gtsrb_six_layer_cnn.pth \
        --output-root ./outputs/gtsrb_time_evaluation \
        --num-screen-samples 100 --num-selected-bases 5 \
        --num-candidates-per-base 3 --steps 220
done

# 4. 时间评估报告（表 6-4 GTSRB 行）
python create_gtsrb_timing_report.py

# 5. 隐蔽性对比（表 6-5 GTSRB 行）
python run_gtsrb_method_comparison_report.py

# 6. 损坏检测（表 6-2 GTSRB 行）
python run_gtsrb_damage_detection.py --methods ivsf

# 7. 一键式（自动跑完 1-6）：python run_gtsrb_full_pipeline.py
python check_gtsrb_artifacts.py
```
