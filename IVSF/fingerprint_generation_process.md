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
- 数据目录: data_root（CIFAR-10 训练集子集）
- 运行设备: device

### 3.2 关键超参数

- num_base_samples: 基础样本数量
- num_candidates_per_sample: 每个基础样本生成候选数
- top_k_per_sample: 每个基础样本保留的指纹数
- steps: 每个候选的优化步数
- lr: 扰动优化学习率
- eps_linf: 扰动 L∞ 约束半径
- tau_js: 目标分布差异参考值
- lambda_boundary, lambda_detect, lambda_js, lambda_stealth: 多目标损失权重
- score_alpha, score_beta, score_gamma: 筛选评分权重

### 3.3 输出产物

默认目录: outputs/quick_run

- fingerprints.npz: 所有数组型产物
- scores.csv: 每个候选的评分明细
- meta.json: 运行配置和统计信息
- preview.png: 可视化预览图

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

1. 读取 CIFAR-10 训练集，download=True
2. 按 num_base_samples 截取子集
3. DataLoader 使用 batch_size=1 顺序遍历
4. 采用固定均值方差进行标准化:
- mean = [0.4914, 0.4822, 0.4465]
- std = [0.247, 0.243, 0.261]

## 6. 候选指纹生成（核心优化）

实现文件: ivsf_pipeline/generator.py

对每个基础样本 x 执行:

1. 先计算参考输出分布 p_ref = softmax(f(x))
2. 循环生成 num_candidates_per_sample 个候选扰动
3. 每个候选初始化 delta ~ U(-eps_linf, eps_linf)
4. 用 Adam 优化 delta，迭代 steps 次
5. 每步执行:
- x_adv = clamp(x + delta, 0, 1)
- p_adv = softmax(f(x_adv))
- 计算多目标损失并反向更新
- 将 delta 投影回 [-eps_linf, eps_linf]

## 7. 多目标损失定义

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

5. 总损失
- L = lambda_boundary * L_boundary + lambda_detect * L_detect + lambda_js * L_js + lambda_stealth * L_stealth

## 8. 候选评分与 Top-K 筛选

实现文件: ivsf_pipeline/generator.py + ivsf_pipeline/losses.py

每个候选优化结束后计算:

1. 结构不确定性分数
- S_BU = entropy(p_adv) - margin_lambda * (top1 - top2)

2. 动态敏感性分数
- S_sens = JS(p_ref, p_adv)

3. 隐蔽性惩罚
- S_stealth = ||delta||_2

4. 总分
- S_total = alpha * S_BU + beta * S_sens - gamma * S_stealth

然后对每个基础样本的候选按 S_total 降序排序，保留 top_k_per_sample 个作为最终指纹。

## 9. 产物组织细节

实现文件: ivsf_pipeline/generator.py + ivsf_pipeline/io_utils.py

1. fingerprints.npz 包含:
- base_images, base_labels, ref_probs
- candidate_images, candidate_deltas, candidate_probs, candidate_scores
- selected_images, selected_deltas, selected_probs, selected_scores

2. scores.csv
- 每行对应一个候选
- 包含 sample_idx, candidate_idx, label, S_BU, S_sens, S_stealth, S_total 及最终损失分解

3. meta.json
- 全部配置
- 实际使用设备
- checkpoint 实际路径
- 样本规模与耗时
- 验证接口占位说明

4. preview.png
- 每个样本展示 base / fingerprint / delta / score

## 10. 与论文方法的对应关系

1. 已对应部分
- 多目标扰动优化
- 候选评分与筛选
- 指纹分布构建（每个基础样本对应多个候选）

2. 当前简化点
- 边界项采用局部 top-1 margin 的绝对差实现
- 验证阶段仅预留接口，未执行在线互信息判定

## 11. 建议执行命令（IVSF 目录）

1. 快速运行
- python run_ivsf_fingerprint.py --quick --device cpu

2. 自定义小规模调试
- python run_ivsf_fingerprint.py --device cpu --config-json "{\"num_base_samples\":20,\"num_candidates_per_sample\":3,\"top_k_per_sample\":1,\"steps\":10}"

3. 指定权重
- python run_ivsf_fingerprint.py --device cpu --checkpoint-path ./checkpoints/resnet20_cifar10.pth

## 12. 流程小结

当前代码已经形成可复现实验闭环:
参数解析 -> 模型加载 -> 数据子集 -> 候选扰动优化 -> 评分筛选 -> 结构化落盘。

这份离线产物可直接作为后续在线完整性验证模块的输入基础。