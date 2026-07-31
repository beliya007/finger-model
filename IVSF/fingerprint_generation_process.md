# IVSF 指纹生成过程梳理（结合当前代码实现）

本文档对应当前 IVSF 目录中的可运行实现，目标是说明从输入参数到指纹产物的完整离线流程。

## 1. 总体目标

给定一个已训练的 CIFAR-10 ResNet20 模型，针对一批基础样本生成局部扰动候选集合，通过多目标优化与评分筛选得到高敏感指纹样本，并导出可用于后续验证阶段的参考数据。

当前实现聚焦离线阶段，不实现在线验证。

## 2. 入口与调用链

1. 命令入口
- 文件: run_ivsf_fingerprint.py
- 作用: 解析参数，构造配置，启动生成器

2. 配置层
- 文件: ivsf_pipeline/config.py
- 作用: 定义超参数、路径和 quick 模式默认值

3. 主流程
- 文件: ivsf_pipeline/generator.py
- 作用: 加载模型、遍历样本、优化候选、筛选 top-k、导出产物

4. 依赖模块
- 模型加载: ivsf_pipeline/model.py
- 数据读取与标准化: ivsf_pipeline/data.py
- 损失与评分: ivsf_pipeline/losses.py
- 结果保存与可视化: ivsf_pipeline/io_utils.py

## 3. 输入、输出与关键配置

### 3.1 输入

- 模型权重路径: checkpoint_path
- 数据目录: data_root（./IVSF/data）
- 运行设备: device

### 3.2 关键超参数

- steps: 每个候选的优化步数
- lr: 扰动优化学习率
- eps_linf: 扰动 L∞ 约束半径
- enforce_linf_constraint: 是否启用 L∞ 硬约束截断（当前默认关闭）
- num_screen_samples: 筛选阶段的基础样本池大小 (n)
- num_selected_bases: 筛选后保留的基础样本数 (topk)
- candidate_sampling_std: 以边界 delta 为均值做高斯采样时的标准差
- tau_js: 目标分布差异参考值
- lambda_boundary, lambda_detect, lambda_js, lambda_stealth: 多目标损失权重
- score_alpha, score_beta: 筛选评分权重 (S_total = alpha * S_BU + beta * S_sens)

### 3.3 输出产物

默认目录: IVSF/outputs/quick_run

- fingerprints.npz: 所有数组型产物
- scores.csv: 每个候选的评分明细（含 screen 和 generation 阶段）
- verification_info.csv: 每个最终入选指纹的验证参考信息
- prediction_report.md: 每个基础样本及其对应指纹的标签与预测概率分布报告
- meta.json: 运行配置和统计信息（含筛选统计）
- preview.png: 可视化预览图
- one_base_three_fingerprints.png: 1 个基础样本及其 3 个入选指纹示例图
- two_bases_three_fingerprints.png: 2 个基础样本及其各自 3 个入选指纹示例图

## 4. 模型加载流程（兼容多来源权重）

实现文件: ivsf_pipeline/model.py

流程如下:

1. 解析 checkpoint 路径
- 优先使用用户传入路径
- 若不存在，尝试 BIVBench 常见路径
- 若仍不存在，尝试下载公开 CIFAR-10 ResNet20 权重

2. 解析 checkpoint 结构
- 优先读取 model 字段
- 兼容 state_dict/net/weights 字段
- 兼容 plain state_dict
- 兼容 DataParallel 的 module. 前缀

3. 键名适配
- 若检测到公开仓库风格键名（如 bn1.weight），自动映射到本地 ResNet20 实现键名（如 conv1.1.weight）

4. 加载到模型并切换 eval 模式

## 5. 数据准备与标准化

实现文件: ivsf_pipeline/data.py

1. 从 `IVSF/data` 读取 CIFAR-10 训练集，download=True
2. 按 `num_screen_samples` 选取初选子集池
3. 采用固定均值方差进行标准化:
- mean = [0.4914, 0.4822, 0.4465]
- std = [0.247, 0.243, 0.261]

## 6. 指纹生成流程 (两阶段重构实现)

实现文件: ivsf_pipeline/generator.py

### 6.1 阶段 1: 基础样本评分与筛选 (Screening Phase)

1. 从 DataLoader 顺序取出一批样本 $(x, y)$，数量为 `num_screen_samples` (n)。
2. 对每个样本，生成一个**随机扰动** $\delta_{random} \sim U(-\epsilon, \epsilon)$。
3. 计算该样本的初步评分 $S_{total} = \alpha \cdot S_{BU} + \beta \cdot S_{sens}$：
    - $S_{BU}$ (Boundary Uncertainty)：衡量样本天生的边界不确定性。
    - $S_{sens}$ (Sensitivity)：衡量样本对该随机扰动的预测分布偏离度。
4. 按 $S_{total}$ 对这 n 个样本进行降序排列，选取前 `num_selected_bases` (topk) 个样本进入下一阶段。

### 6.2 阶段 2: 深度优化与指纹导出 (Generation Phase)

仅对入选的 Top-K 基础样本执行以下操作：

1. **边界扰动优化 (Boundary Optimization)**：
    - 调用 `_optimize_boundary_delta`。
    - 以 Adam 优化器在 `steps` 步内寻找使总损失（边界、检测、JS、隐蔽性）最小的中心扰动 `boundary_delta`。
2. **指纹采样 (Candidate Sampling)**：
    - 在 `boundary_delta` 周围进行 `num_candidates_per_sample` 次高斯采样：
      $$\delta_i \sim N(boundary\_delta, candidate\_sampling\_std^2 I)$$
    - 得到的每一个采样结果直接作为最终指纹输出，不再进行二次评分筛选。
3. **元数据记录**：
    - `meta.json` 中会记录筛选阶段和生成阶段的具体样本数量及耗时。
    - `scores.csv` 中 `phase` 字段用于区分 `screen` (筛选) 和 `generation` (生成) 数据。

## 7. 评分系统细节 (Scoring)

当前评分公式已移除隐蔽性惩罚项，专注于指纹的辨别性：
$$S_{total} = \alpha \cdot S_{BU} + \beta \cdot S_{sens}$$

- **S_BU**: 计算公式为 `entropy(p_adv) - lambda * margin`。
- **S_sens**: 计算公式为 `JS_Divergence(p_ref, p_adv)`。
- **S_stealth**: 虽然不参与总分累加，但仍记录在 CSV 中供隐蔽性分析参考。

## 8. 多目标损失定义

实现文件: ivsf_pipeline/losses.py

设 p_adv 为扰动样本输出概率，y 为真实标签，delta 为扰动。

1. 边界项
- boundary_loss = |p_true - p_max_non_true|
- 作用: 推动样本靠近局部决策边界

2. 检测敏感项
- detect_loss = -JS(p_ref, p_adv)
- 作用: 鼓励输出分布变化

3. JS 范围约束项
- js_range_loss = (JS(p_ref, p_adv) - tau_js)^2
- 作用: 避免过大或过小的分布偏移

4. 隐蔽性项
- stealth_loss = ||delta||_2
- 作用: 控制扰动幅度
