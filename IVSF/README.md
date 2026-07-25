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

如需使用更完整的训练配置（保持 2 个原始样本、每样本保留 3 个指纹不变），可使用：
```powershell
python run_ivsf_fingerprint.py --full --checkpoint-path ./checkpoints/resnet20_cifar10.pth --device cuda
```
*   **产出**：`./outputs/full_run/fingerprints.npz`

### 第二步：后门攻击模型训练
**描述**：在原始模型基础上进行后门攻击微调（例如 BadNets），生成被攻击后的模型权重。
```powershell
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run --device cuda
```
*   **产出**：`./outputs/attack_run/attacked_resnet20_cifar10.pth`

### 第三步：指纹一致性对比与评估
**描述**：将第一步生成的指纹输入给第二步产生的攻击模型，对比攻击前后的标签变化与点互信息变化。
```powershell
python run_ivsf_compare.py --fingerprints ./outputs/quick_run/fingerprints.npz --checkpoint-path ./checkpoints/resnet20_cifar10.pth --attacked-checkpoint ./outputs/attack_run/attacked_resnet20_cifar10.pth --output-dir ./outputs/attack_run --device cuda
```
*   **产出**：
    *   `fingerprint_attack_comparison.csv`
    *   `fingerprint_attack_comparison.npz`

## 4. 参数说明

*   `--device`: 指定运行设备 (`cuda` 或 `cpu`)。
*   `--quick`: 使用快速预设配置运行。
*   `--full`: 使用完整预设配置运行（样本数量与最终指纹数量不变）。
*   `--attacked-checkpoint`: 手动指定要加载的攻击模型权重路径。
*   `--checkpoint-path`: 原始干净模型的权重路径。

## 5. 开发者备注

*   **攻击类型**：当前已实现 BadNets 风格后门攻击（右下角触发器）。
*   **权重加载**：程序内置了 Key 自动转换逻辑，兼容从 BIVBench 或 chenyaofo/pytorch-cifar-models 来源的 ResNet20 权重。
*   **GPU 优化**：所有训练流程默认支持 AMP 自动混合精度加速与非阻塞数据搬运。



----------------------------------------------------------------------------

