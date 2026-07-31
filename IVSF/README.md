# IVSF 指纹验证与攻击模拟流程

本项目用于生成深度学习模型的指纹（Fingerprinting），并模拟攻击场景下的模型所有权验证与一致性对比。支持 ResNet20 在 CIFAR-10 数据集上的完整实验流程。

## 1. 环境准备

请在项目根目录 `shiyan-finger-model` 下执行以下操作：

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. 安装必要的依赖包
python -m pip install --upgrade pip
python -m pip install numpy matplotlib torch torchvision tqdm mlconfig pyyaml scikit-learn
```

## 2. 目录结构

```
shiyan-finger-model/
  ├── .venv/               # 虚拟环境目录
  └── IVSF/                # IVSF 实验核心目录
      ├── attack/          # 攻击模块（数据投毒与训练逻辑）
      ├── checkpoints/     # 存放模型权重 (resnet20_cifar10.pth)
      ├── data/            # 存放数据集 (CIFAR-10)
      ├── ivsf_pipeline/   # 指纹生成核心管线
      ├── outputs/         # 实验结果产出
      ├── run_ivsf_fingerprint.py  # 入口 1：指纹生成
      ├── run_ivsf_attack_train.py # 入口 2：攻击训练
      └── run_ivsf_compare.py      # 入口 3：对比验证
```

## 3. 标准操作流程 (Standard Workflow)

**前提条件**：确保已激活虚拟环境 `(.venv)`，并且当前位于 `IVSF` 目录下。

```powershell
# 切换到 IVSF 目录
cd IVSF
```

### 第一步：指纹生成与原始模型评估
**描述**：从原始模型生成候选指纹，并记录在原始模型下的标签输出与点互信息（MI）。
```powershell
python run_ivsf_fingerprint.py --quick --checkpoint-path ./checkpoints/resnet20_cifar10.pth --device cuda
```
*   **产出**：`./outputs/quick_run/fingerprints.npz`
*   **规模说明**：快速模式默认筛选 5 个样本，保留 2 个基础样本，每个基础样本生成 6 个候选指纹（总计最多 12 个）。

如需使用更完整配置（扩大筛选池并增加每样本候选数），可使用：
```powershell
python run_ivsf_fingerprint.py --full --checkpoint-path ./checkpoints/resnet20_cifar10.pth --device cuda
```
*   **产出**：`./outputs/full_run/fingerprints.npz`
*   **规模说明**：完整模式默认筛选 100 个样本，保留 2 个基础样本，每个基础样本生成 24 个候选指纹（总计最多 48 个）。

### 第二步：后门攻击模型训练
**描述**：在原始模型基础上进行后门攻击微调（例如 BadNets），生成被攻击后的模型权重。
```powershell
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run --device cuda
```
*   **产出**：`./outputs/attack_run/attacked_resnet20_cifar10.pth`


# 第二步：攻击训练 300 轮
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run_300ep --device cuda --epochs 300

# 第三步：使用 full 指纹，对 300 轮攻击模型做对比
python run_ivsf_compare.py --fingerprints ./outputs/full_run/fingerprints.npz --checkpoint-path ./checkpoints/resnet20_cifar10.pth --attacked-checkpoint ./outputs/attack_run_300ep/attacked_resnet20_cifar10.pth --output-dir ./outputs/attack_run_300ep --device cuda

### 第三步：指纹一致性对比与评估
**描述**：将第一步生成的指纹输入给第二步产生的攻击模型，对比攻击前后的标签变化与点互信息变化。
```powershell
python run_ivsf_compare.py --fingerprints ./outputs/quick_run/fingerprints.npz --checkpoint-path ./checkpoints/resnet20_cifar10.pth --attacked-checkpoint ./outputs/attack_run/attacked_resnet20_cifar10.pth --output-dir ./outputs/attack_run --device cuda
```
*   **产出**：
    *   `fingerprint_attack_comparison.csv`
    *   `fingerprint_attack_comparison.npz`

## 4. 参数说明

### 4.1 指纹生成脚本

```powershell
python run_ivsf_fingerprint.py [--quick | --full] [--config-json '{...}'] [--output-dir PATH] [--checkpoint-path PATH] [--device cuda|cpu]
```

*   `--quick`: 使用快速预设（输出目录 `./outputs/quick_run`）。
*   `--full`: 使用完整预设（输出目录 `./outputs/full_run`）。
*   `--config-json`: 传入 JSON 字符串覆盖配置字段，例如：
    ```powershell
    python run_ivsf_fingerprint.py --config-json '{"num_screen_samples":20,"num_candidates_per_sample":12}'
    ```
*   `--output-dir`: 覆盖输出目录。
*   `--checkpoint-path`: 原始干净模型权重路径。
*   `--device`: 指定运行设备（`cuda` 或 `cpu`）。

### 4.2 攻击训练脚本

```powershell
python run_ivsf_attack_train.py --checkpoint-path PATH --output-dir PATH [--device cuda|cpu] [--epochs N] [--batch-size N] [--lr LR] [--poison-ratio R] [--trigger-label C] [--trigger-size N] [--trigger-alpha A] [--seed N] [--num-workers N] [--data-root PATH] [--no-amp]
```

*   `--checkpoint-path`: 原始干净模型权重路径。
*   `--output-dir`: 攻击模型输出目录。
*   `--epochs`: 攻击微调轮数（默认 3）。
*   `--batch-size`: 批大小（默认 128）。
*   `--lr`: 学习率（默认 `1e-3`）。
*   `--poison-ratio`: 投毒比例（默认 0.1）。
*   `--trigger-label`: 触发器目标类别（默认 0）。
*   `--trigger-size`: 触发器边长（默认 4）。
*   `--trigger-alpha`: 触发器覆盖强度（默认 1.0）。
*   `--seed`: 随机种子（默认 1337）。
*   `--num-workers`: DataLoader 线程数（默认 2）。
*   `--data-root`: 数据目录（默认 `./data`）。
*   `--no-amp`: 关闭 AMP 自动混合精度。

### 4.3 指纹对比脚本

```powershell
python run_ivsf_compare.py --fingerprints PATH --checkpoint-path PATH --output-dir PATH [--attacked-checkpoint PATH] [--device cuda|cpu]
```

*   `--fingerprints`: 第一步生成的 `fingerprints.npz` 路径。
*   `--checkpoint-path`: 原始干净模型权重路径。
*   `--output-dir`: 对比结果输出目录。
*   `--attacked-checkpoint`: 手动指定攻击模型路径；不指定时默认使用 `--output-dir` 下的标准路径。
*   `--device`: 指定运行设备（`cuda` 或 `cpu`）。

### 4.4 兼容入口（Legacy）

```powershell
python run_ivsf_attack.py --fingerprints PATH --checkpoint-path PATH --output-dir PATH [--attacked-checkpoint PATH] [--device cuda|cpu]
```

该脚本保留用于兼容旧流程，功能上与 `run_ivsf_compare.py` 接近，建议新流程优先使用 `run_ivsf_compare.py`。

## 5. 开发者备注

*   **攻击类型**：当前已实现 BadNets 风格后门攻击（右下角触发器）。
*   **权重加载**：程序内置了 Key 自动转换逻辑，兼容从 BIVBench 或 chenyaofo/pytorch-cifar-models 来源的 ResNet20 权重。
*   **GPU 优化**：所有训练流程默认支持 AMP 自动混合精度加速与非阻塞数据搬运。



----------------------------------------------------------------------------

