# IVSF 指纹生成流程说明（独立版）

本项目目录为独立工作目录，不依赖 BIVBench。
你只需要准备一个已训练好的 ResNet20(CIFAR-10) 权重文件，即可在本目录完成离线指纹生成。

当前实现范围：
- 离线指纹生成
- 候选指纹评分与 Top-K 筛选
- 结果产物导出（fingerprints.npz、scores.csv、meta.json、preview.png）
- 在线验证接口占位（本阶段不实现在线验证）

## 1. 目录约定（在 IVSF 目录下）

建议目录结构如下：

	IVSF/
	  checkpoints/
		resnet20_cifar10.pth
	  data/
	  outputs/
	  ivsf_pipeline/
	  run_ivsf_fingerprint.py

说明：
- checkpoint 默认路径为 ./checkpoints/resnet20_cifar10.pth
- CIFAR-10 默认下载到 ./data
- 输出默认写入 ./outputs/quick_run
- 若本地 checkpoint 不存在，程序会自动尝试 BIVBench 常见权重路径（如 ../BIVBench-main/outputs/.../best_0.pth）
- checkpoint 加载规则模仿 BIVBench：优先读取字典中的 model 字段，也兼容 plain state_dict 与 module. 前缀
- 若上述路径都不存在，程序会尝试自动下载公开的 CIFAR-10 ResNet20 权重并保存到 ./checkpoints/resnet20_cifar10.pth

## 2. 环境准备（在 IVSF 目录执行）

	F:/python3.13/python.exe -m venv .venv
	.\.venv\Scripts\python.exe -m pip install --upgrade pip
	.\.venv\Scripts\python.exe -m pip install numpy matplotlib torch torchvision tqdm mlconfig pyyaml scikit-learn

## 3. 执行命令（只在 IVSF 目录执行）

快速运行（使用默认 checkpoint 路径）：

	.\.venv\Scripts\python.exe run_ivsf_fingerprint.py --quick

指定 checkpoint 路径：

	.\.venv\Scripts\python.exe run_ivsf_fingerprint.py --quick --checkpoint-path ./checkpoints/resnet20_cifar10.pth

直接复用 BIVBench 权重（若你已经在相邻目录有 BIVBench-main）：

	.\.venv\Scripts\python.exe run_ivsf_fingerprint.py --quick --checkpoint-path ../BIVBench-main/outputs/cifar10/random_start_models/resnet20/best_0.pth

CPU 运行：

	.\.venv\Scripts\python.exe run_ivsf_fingerprint.py --quick --checkpoint-path ./checkpoints/resnet20_cifar10.pth --device cpu
    
& ".\.venv\Scripts\python.exe" .\IVSF\run_ivsf_fingerprint.py --device cpu
## 4. 输出文件

默认输出目录：./outputs/quick_run

主要产物说明：
- fingerprints.npz：基础样本、候选指纹、入选指纹、扰动与概率输出
- scores.csv：每个样本每个候选指纹的评分明细
- meta.json：运行参数、checkpoint 路径、运行统计
- preview.png：可视化预览图（原图、指纹、扰动、评分）

## 5. 常见问题与排查

1) 报错：Checkpoint not found
- 原因：未放置权重文件，或路径错误。
- 处理：把权重放到 ./checkpoints/resnet20_cifar10.pth，或通过 --checkpoint-path 指定正确路径。

2) 报错：ModuleNotFoundError
- 原因：没有使用 IVSF/.venv 或依赖未安装。
- 处理：使用 .\.venv\Scripts\python.exe 执行，并重装依赖。

3) 报错：CUDA 相关错误
- 原因：CUDA 不可用或驱动不匹配。
- 处理：添加 --device cpu。

4) 运行较慢
- 原因：快速模式仍有多目标迭代优化。
- 处理：用 --config-json 降低样本数、候选数、迭代步数。

## 6. 参数覆盖示例

	.\.venv\Scripts\python.exe run_ivsf_fingerprint.py --checkpoint-path ./checkpoints/resnet20_cifar10.pth --config-json '{"num_base_samples":20,"num_candidates_per_sample":3,"steps":20}'

说明：
- --quick 使用内置快速配置。
- --config-json 可覆盖任意配置项。

## 7. 代码结构

- run_ivsf_fingerprint.py：主入口
- ivsf_pipeline/config.py：参数定义
- ivsf_pipeline/generator.py：指纹生成与筛选主流程
- ivsf_pipeline/losses.py：多目标损失与评分
- ivsf_pipeline/model.py：ResNet20 与 checkpoint 加载
- ivsf_pipeline/verification_interface.py：在线验证接口占位
