# IVSF 指纹验证与攻击模拟流程

本项目用于生成深度学习模型的指纹（Fingerprinting），并模拟攻击场景下的模型所有权验证与一致性对比。指纹生成模块支持以下数据集与模型组合：

* CIFAR-10 + ResNet20
* GTSRB + 六层 CNN（43 类）

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

### GTSRB + 六层 CNN

先在 IVSF 中训练六层 CNN。训练入口会自动下载 GTSRB、将图像缩放至 $32 \times 32$，并将最佳权重保存到 `./checkpoints/gtsrb_six_layer_cnn.pth`：

```powershell
python train_gtsrb_six_layer_cnn.py --device cuda
```

训练完成后，使用该权重生成 IVSF 指纹：

```powershell
python run_ivsf_fingerprint.py --quick --dataset gtsrb --data-root ./data --checkpoint-path ./checkpoints/gtsrb_six_layer_cnn.pth --device cuda
```

GTSRB 的六层 CNN 依次包含 6 个 $3 \times 3$ 卷积层（通道数为 32、32、64、64、128、128）、3 个最大池化层以及 43 类全连接输出层。其层命名与 BIVBench 的 `gtsrb_convnet` 相同，IVSF 可加载两种训练流程导出的 `model` 权重。

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
**描述**：在原始模型基础上执行攻击流程，生成被攻击后的模型权重。支持 `badnet / trojan / clean_label_backdoor / weight_pruning / ftll`。
```powershell
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run --device cuda --attack-type badnet
```
*   **产出**：`./outputs/attack_run/attacked_resnet20_cifar10.pth`

### 第二步补充：批量训练 5 种攻击（默认 400 轮）
**描述**：一次顺序跑完 5 种攻击，统一输出到同一个根目录下的不同子目录。
```powershell
python run_ivsf_attack_train_batch.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-root ./outputs/attack_runs_400ep --device cuda
```
*   **默认攻击顺序**：`badnet,trojan,clean_label_backdoor,weight_pruning,ftll`
*   **默认训练轮数**：
    *   `badnet / trojan / clean_label_backdoor`: 400 轮
    *   `ftll`: 400 轮
    *   `weight_pruning`: 不依赖 epoch，直接做 15% 权值剪枝
*   **默认输出结构**：
    *   `./outputs/attack_runs_400ep/badnet/`
    *   `./outputs/attack_runs_400ep/trojan/`
    *   `./outputs/attack_runs_400ep/clean_label_backdoor/`
    *   `./outputs/attack_runs_400ep/weight_pruning/`
    *   `./outputs/attack_runs_400ep/ftll/`
    *   `./outputs/attack_runs_400ep/batch_attack_summary.json`

```powershell
# 只跑部分攻击
python run_ivsf_attack_train_batch.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-root ./outputs/attack_runs_partial --device cuda --attacks badnet,trojan,ftll
```

```powershell
# Trojan 后门
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run_trojan --device cuda --attack-type trojan

# Clean-label backdoor
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run_clean_label --device cuda --attack-type clean_label_backdoor

# 权值剪枝（默认移除 15%）
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run_prune --device cuda --attack-type weight_pruning --pruning-ratio 0.15

# FTLL（最后一层，1e-5）
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run_ftll --device cuda --attack-type ftll --ftll-lr 1e-5
```


# 第二步：攻击训练 300 轮
python run_ivsf_attack_train.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-dir ./outputs/attack_run_300ep --device cuda --epochs 300

# 第三步：使用 full 指纹，对 300 轮攻击模型做对比
python run_ivsf_compare.py --fingerprints ./outputs/full_run/fingerprints.npz --checkpoint-path ./checkpoints/resnet20_cifar10.pth --attacked-checkpoint ./outputs/attack_run_400ep/attacked_resnet20_cifar10.pth --output-dir ./outputs/attack_run_400ep --device cuda

### 第三步：指纹一致性对比与评估
**描述**：将第一步生成的指纹输入给第二步产生的攻击模型，对比攻击前后的标签变化与点互信息变化。
```powershell
python run_ivsf_compare.py --fingerprints ./outputs/quick_run/fingerprints.npz --checkpoint-path ./checkpoints/resnet20_cifar10.pth --attacked-checkpoint ./outputs/attack_run/attacked_resnet20_cifar10.pth --output-dir ./outputs/attack_run --device cuda
```
*   **产出**：
    *   `fingerprint_attack_comparison.csv`
    *   `fingerprint_attack_comparison.npz`

### 补充：三种对比实验指纹生成（SSF / PublicCheck / MiSentry）
**描述**：生成三种对比方法的指纹文件，便于后续统一做完整性验证或攻击对比。
```powershell
python run_baseline_fingerprints.py --methods ssf,publiccheck,misentry --checkpoint-path ./checkpoints/resnet20_cifar10.pth --output-root ./outputs/baseline_fingerprints --device cuda
```
*   **产出目录**：
    *   `./outputs/baseline_fingerprints/ssf/`
    *   `./outputs/baseline_fingerprints/publiccheck/`
    *   `./outputs/baseline_fingerprints/misentry/`
*   **每个方法目录产出**：
    *   `fingerprints.npz`（可直接给 `run_ivsf_compare.py --fingerprints` 使用）
    *   `fingerprints.npy`（兼容 BIVBench 的 `--fp_path` 用法）
    *   `meta.json`

说明：
*   SSF 仓库已放到 `../third_party/Sensitive_Sample_Fingerprinting`，用于对照与参考。
*   当前脚本在 CIFAR-10 + ResNet20 流程下给出统一可运行的三方法基线实现，方便与你现有 IVSF 流程直接联动。

#### 对比方案按 IVSF full 规模运行（推荐用于论文主表）
为保证与 `outputs/full_run/fingerprints.npz` 的 IVSF 结果同规模，建议使用如下配置（每个方法 30 枚指纹）：

```powershell
..\.venv\Scripts\python.exe run_baseline_fingerprints.py --methods ssf,publiccheck,misentry --output-root ./outputs/baseline_fingerprints_full --checkpoint-path ./checkpoints/resnet20_cifar10.pth --device cuda --num-screen-samples 300 --num-selected-bases 10 --num-candidates-per-base 3 --steps 220 --lr 0.004 --l2-radius 3.0 --seed 1337
```

生成 full 对 full 指标表（LPIPS/JS）：
```powershell
..\.venv\Scripts\python.exe -c "exec('''import csv\nimport numpy as np\nimport torch\nfrom pathlib import Path\nfrom lpips import LPIPS as LPIPSModel\nfrom ivsf_pipeline.model import load_resnet20_cifar10\nfrom ivsf_pipeline.data import cifar10_mean_std, normalize_batch\nfrom ivsf_pipeline.losses import softmax_probs, js_divergence\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel, _ = load_resnet20_cifar10(Path('./checkpoints/resnet20_cifar10.pth'), device=device)\nmodel.eval()\nmean, std = cifar10_mean_std(device)\nlpips_model = LPIPSModel(net='alex', version='0.1').to(device)\nlpips_model.eval()\n\nmethods = [('MiSentry', Path('outputs/baseline_fingerprints_full/misentry/fingerprints.npz')), ('SSF', Path('outputs/baseline_fingerprints_full/ssf/fingerprints.npz')), ('PublicCheck', Path('outputs/baseline_fingerprints_full/publiccheck/fingerprints.npz')), ('IVSF', Path('outputs/full_run/fingerprints.npz'))]\nrows = []\nfor name, p in methods:\n    d = np.load(p, allow_pickle=False)\n    fps = d['selected_images'].astype(np.float32)\n    bases = d['base_images'].astype(np.float32)\n    base_idx = d['selected_base_sample_indices'].astype(np.int64)\n    if 'selected_probs' in d.files:\n        fp_probs = d['selected_probs'].astype(np.float32)\n    else:\n        x = torch.from_numpy(fps).to(device=device, dtype=torch.float32)\n        out = []\n        with torch.no_grad():\n            for i in range(0, x.shape[0], 256):\n                logits = model(normalize_batch(x[i:i+256], mean, std))\n                out.append(softmax_probs(logits).detach().cpu().numpy())\n        fp_probs = np.concatenate(out, axis=0).astype(np.float32)\n    b = torch.from_numpy(bases).to(device=device, dtype=torch.float32)\n    with torch.no_grad():\n        base_logits = model(normalize_batch(b, mean, std))\n        base_probs = softmax_probs(base_logits)\n    js_vals, lpips_vals = [], []\n    for i in range(fps.shape[0]):\n        bi = int(base_idx[i])\n        p_ref = base_probs[bi:bi+1]\n        p_fp = torch.from_numpy(fp_probs[i:i+1]).to(device=device, dtype=torch.float32)\n        js_vals.append(float(js_divergence(p_ref, p_fp).item()))\n        base_img = torch.from_numpy(bases[bi:bi+1]).to(device=device, dtype=torch.float32)\n        fp_img = torch.from_numpy(fps[i:i+1]).to(device=device, dtype=torch.float32)\n        lpips_vals.append(float(lpips_model(base_img * 2.0 - 1.0, fp_img * 2.0 - 1.0).item()))\n    rows.append({'Dataset':'CIFAR-10','Method':name,'LPIPS':float(np.mean(lpips_vals)),'JS':float(np.mean(js_vals)),'num_fingerprints':int(fps.shape[0]),'source_fingerprints':str(p).replace('\\\\','/')})\nout_csv = Path('outputs/baseline_fingerprints_full/cifar10_method_lpips_js_full_vs_ivsf_full.csv')\nwith out_csv.open('w', newline='', encoding='utf-8') as f:\n    w = csv.DictWriter(f, fieldnames=['Dataset','Method','LPIPS','JS','num_fingerprints','source_fingerprints'])\n    w.writeheader()\n    w.writerows(rows)\nprint(out_csv)\n''')"
```

*   **产出文件**：
    *   `./outputs/baseline_fingerprints_full/cifar10_method_lpips_js_full_vs_ivsf_full.csv`
    *   `./outputs/baseline_fingerprints_full/cifar10_method_lpips_js_full_vs_ivsf_full.md`

### 补充：纵向指纹对比图生成（原图 + IVSF + MiSentry + SSF + PublicCheck）
**描述**：将同一基础样本下的 5 张图按纵向拼接，便于论文插图或直观对比。

先生成与基线方法同样本索引对齐的 IVSF 指纹（示例索引：46043, 47328）：
```powershell
..\.venv\Scripts\python.exe -c "from ivsf_pipeline import IVSFConfig, IVSFFingerprintGenerator; cfg=IVSFConfig.quick(output_dir='./outputs/ivsf_baseline_aligned'); cfg.num_screen_samples=8; cfg.num_selected_bases=2; cfg.num_candidates_per_sample=2; cfg.steps=60; cfg.lr=0.01; cfg.base_sample_indices=[46043,47328]; cfg.random_base_sampling=False; cfg.enforce_linf_constraint=True; cfg.eps_linf=3.0/255.0; cfg.candidate_sampling_std=1.0/255.0; cfg.checkpoint_path='./checkpoints/resnet20_cifar10.pth'; cfg.device='cuda'; IVSFFingerprintGenerator(cfg).run()"
```

生成带标题的纵向对比图：
```powershell
..\.venv\Scripts\python.exe -c "import numpy as np; import matplotlib.pyplot as plt; from pathlib import Path; iv=np.load('outputs/ivsf_baseline_aligned/fingerprints.npz'); mi=np.load('outputs/baseline_fingerprints_quick/misentry/fingerprints.npz'); ss=np.load('outputs/baseline_fingerprints_quick/ssf/fingerprints.npz'); pc=np.load('outputs/baseline_fingerprints_quick/publiccheck/fingerprints.npz'); base_idx=0; rank=0; orig=iv['base_images'][base_idx]; iv_img=iv['selected_images'][np.where((iv['selected_base_sample_indices']==base_idx)&(iv['selected_rank_in_sample']==rank))[0][0]]; mi_img=mi['selected_images'][np.where((mi['selected_base_sample_indices']==base_idx)&(mi['selected_rank_in_sample']==rank))[0][0]]; ss_img=ss['selected_images'][np.where((ss['selected_base_sample_indices']==base_idx)&(ss['selected_rank_in_sample']==rank))[0][0]]; pc_img=pc['selected_images'][np.where((pc['selected_base_sample_indices']==base_idx)&(pc['selected_rank_in_sample']==rank))[0][0]]; labels=['Original sample','IVSF fingerprint','MiSentry fingerprint','SSF fingerprint','PublicCheck fingerprint']; imgs=[orig,iv_img,mi_img,ss_img,pc_img]; fig,axes=plt.subplots(5,1,figsize=(4,18)); [ax.imshow(np.clip(np.transpose(img,(1,2,0)),0,1)) or ax.set_title(lab,fontsize=12) or ax.axis('off') for ax,img,lab in zip(axes,imgs,labels)]; fig.tight_layout(); out=Path('outputs/baseline_fingerprints_quick/vertical_fingerprint_comparison.png'); fig.savefig(out,dpi=220); print(out)"
```

生成无文字说明版本（纯图版）：
```powershell
..\.venv\Scripts\python.exe -c "import numpy as np; import matplotlib.pyplot as plt; from pathlib import Path; iv=np.load('outputs/ivsf_baseline_aligned/fingerprints.npz'); mi=np.load('outputs/baseline_fingerprints_quick/misentry/fingerprints.npz'); ss=np.load('outputs/baseline_fingerprints_quick/ssf/fingerprints.npz'); pc=np.load('outputs/baseline_fingerprints_quick/publiccheck/fingerprints.npz'); base_idx=0; rank=0; orig=iv['base_images'][base_idx]; iv_img=iv['selected_images'][np.where((iv['selected_base_sample_indices']==base_idx)&(iv['selected_rank_in_sample']==rank))[0][0]]; mi_img=mi['selected_images'][np.where((mi['selected_base_sample_indices']==base_idx)&(mi['selected_rank_in_sample']==rank))[0][0]]; ss_img=ss['selected_images'][np.where((ss['selected_base_sample_indices']==base_idx)&(ss['selected_rank_in_sample']==rank))[0][0]]; pc_img=pc['selected_images'][np.where((pc['selected_base_sample_indices']==base_idx)&(pc['selected_rank_in_sample']==rank))[0][0]]; imgs=[orig,iv_img,mi_img,ss_img,pc_img]; fig,axes=plt.subplots(5,1,figsize=(3.6,16)); plt.subplots_adjust(left=0,right=1,top=1,bottom=0,hspace=0.02); [ax.imshow(np.clip(np.transpose(img,(1,2,0)),0,1)) or ax.axis('off') for ax,img in zip(axes,imgs)]; out=Path('outputs/baseline_fingerprints_quick/vertical_fingerprint_comparison_no_text.png'); fig.savefig(out,dpi=260,bbox_inches='tight',pad_inches=0); print(out)"
```

*   **产出文件**：
    *   `./outputs/baseline_fingerprints_quick/vertical_fingerprint_comparison.png`
    *   `./outputs/baseline_fingerprints_quick/vertical_fingerprint_comparison_no_text.png`

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
*   `--dataset`: 数据集/模型组合，可选 `cifar10`（默认）或 `gtsrb`。
*   `--data-root`: 数据集根目录，默认 `./data`。
*   `--device`: 指定运行设备（`cuda` 或 `cpu`）。

### 4.2 攻击训练脚本

```powershell
python run_ivsf_attack_train.py --checkpoint-path PATH --output-dir PATH [--device cuda|cpu] [--attack-type badnet|trojan|clean_label_backdoor|weight_pruning|ftll] [--epochs N] [--batch-size N] [--lr LR] [--poison-ratio R] [--trigger-label C] [--trigger-size N] [--trigger-alpha A] [--trojan-steps N] [--trojan-lr LR] [--trojan-batch-size N] [--clean-label-base-label C] [--clean-label-eps E] [--clean-label-alpha A] [--clean-label-steps N] [--clean-label-max-poison N] [--pruning-ratio R] [--ftll-epochs N] [--ftll-lr LR] [--seed N] [--num-workers N] [--data-root PATH] [--no-amp]
```

*   `--checkpoint-path`: 原始干净模型权重路径。
*   `--output-dir`: 攻击模型输出目录。
*   `--attack-type`: 攻击类型，支持 `badnet / trojan / clean_label_backdoor / weight_pruning / ftll`。
*   `--epochs`: 攻击微调轮数（默认 3）。
*   `--batch-size`: 批大小（默认 128）。
*   `--lr`: 学习率（默认 `1e-3`）。
*   `--poison-ratio`: 投毒比例（默认 0.1）。
*   `--trigger-label`: 触发器目标类别（默认 0）。
*   `--trigger-size`: 触发器边长（默认 4）。
*   `--trigger-alpha`: 触发器覆盖强度（默认 1.0）。
*   `--trojan-steps`: Trojan 触发器优化步数（默认 200）。
*   `--trojan-lr`: Trojan 触发器优化学习率（默认 0.1）。
*   `--trojan-batch-size`: Trojan 触发器优化批大小（默认 64）。
*   `--clean-label-base-label`: clean-label 投毒源类别（默认 1）。
*   `--clean-label-eps`: clean-label 扰动上限（默认 `8/255`）。
*   `--clean-label-alpha`: clean-label 每步更新幅度（默认 `1/255`）。
*   `--clean-label-steps`: clean-label 优化步数（默认 5）。
*   `--clean-label-max-poison`: clean-label 最大投毒样本数（默认 256）。
*   `--pruning-ratio`: 权值剪枝比例（默认 `0.15`）。
*   `--ftll-epochs`: FTLL 训练轮数（默认 3）。
*   `--ftll-lr`: FTLL 学习率（默认 `1e-5`）。
*   `--seed`: 随机种子（默认 1337）。
*   `--num-workers`: DataLoader 线程数（默认 2）。
*   `--data-root`: 数据目录（默认 `./data`）。
*   `--no-amp`: 关闭 AMP 自动混合精度。

### 4.2.1 批量攻击训练脚本

```powershell
python run_ivsf_attack_train_batch.py --checkpoint-path PATH --output-root PATH [--device cuda|cpu] [--attacks badnet,trojan,clean_label_backdoor,weight_pruning,ftll] [--epochs 400] [--ftll-epochs 400] [--batch-size N] [--lr LR] [--ftll-lr LR] [--poison-ratio R] [--trigger-label C] [--trigger-size N] [--trigger-alpha A] [--trojan-steps N] [--trojan-lr LR] [--trojan-batch-size N] [--clean-label-base-label C] [--clean-label-eps E] [--clean-label-alpha A] [--clean-label-steps N] [--clean-label-max-poison N] [--pruning-ratio R] [--seed N] [--num-workers N] [--data-root PATH] [--no-amp]
```

*   `--output-root`: 批量运行输出根目录，脚本会为每种攻击创建同名子目录。
*   `--attacks`: 逗号分隔的攻击列表，默认依次运行全部 5 种攻击。
*   `--epochs`: `badnet / trojan / clean_label_backdoor` 的训练轮数，默认 400。
*   `--ftll-epochs`: `ftll` 的训练轮数，默认 400。
*   `batch_attack_summary.json`: 汇总每种攻击的输出目录、checkpoint、配置和元信息。

### 4.3 指纹对比脚本

```powershell
python run_ivsf_compare.py --fingerprints PATH --checkpoint-path PATH --output-dir PATH [--attacked-checkpoint PATH] [--attack-type badnet|trojan|clean_label_backdoor|weight_pruning|ftll] [--device cuda|cpu]
```

*   `--fingerprints`: 第一步生成的 `fingerprints.npz` 路径。
*   `--checkpoint-path`: 原始干净模型权重路径。
*   `--output-dir`: 对比结果输出目录。
*   `--attacked-checkpoint`: 手动指定攻击模型路径；不指定时默认使用 `--output-dir` 下的标准路径。
*   `--attack-type`: 当未指定 `--attacked-checkpoint` 时，用于自动推导默认攻击模型文件名。
*   `--device`: 指定运行设备（`cuda` 或 `cpu`）。

### 4.4 兼容入口（Legacy）

```powershell
python run_ivsf_attack.py --fingerprints PATH --checkpoint-path PATH --output-dir PATH [--attacked-checkpoint PATH] [--attack-type badnet|trojan|clean_label_backdoor|weight_pruning|ftll] [--device cuda|cpu]
```

该脚本保留用于兼容旧流程，功能上与 `run_ivsf_compare.py` 接近，建议新流程优先使用 `run_ivsf_compare.py`。

## 5. 开发者备注

*   **攻击类型**：已实现 BadNets、Trojan、Clean-label backdoor、Weight Pruning（默认 15%）、FTLL（最后一层 1e-5）。
*   **权重加载**：程序内置了 Key 自动转换逻辑，兼容从 BIVBench 或 chenyaofo/pytorch-cifar-models 来源的 ResNet20 权重。
*   **GPU 优化**：所有训练流程默认支持 AMP 自动混合精度加速与非阻塞数据搬运。



----------------------------------------------------------------------------

