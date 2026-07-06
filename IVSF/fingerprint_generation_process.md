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

### 6.1 基础样本如何得到多个候选指纹

当前实现中，一个基础样本不会只生成一个扰动结果，而是会独立重复多次候选优化流程。

具体过程如下:

1. 固定同一个基础样本 x 和它的参考输出分布 p_ref
- 也就是在同一个样本局部邻域内寻找多个可区分的扰动点

2. 按 num_candidates_per_sample 重复执行候选生成
- 每次都重新随机初始化一个 delta
- 每次都从不同的随机起点出发优化

3. 每次优化结束后得到一个候选指纹
- 对应扰动后的样本 x_adv
- 对应扰动 delta
- 对应输出概率 p_adv
- 对应候选评分 score_dict

4. 这些候选先全部保存在 sample_candidates 中
- 也就是说，一个基础样本会先得到一个候选集合
- 还不会立刻决定哪些是最终指纹

因此，“互选指纹”在当前代码语义里更准确地说是“同一基础样本生成得到的多个候选指纹”，它们共同构成该基础样本的局部指纹集合，再从中筛选最优的若干个。

### 6.2 单个候选指纹优化结束后保存了什么

对每个候选，代码会额外记录:

1. candidate_pred_label
- 即 argmax p_adv

2. score_dict
- 包含 S_BU、S_sens、S_stealth、S_total
- 也包含 final_loss、final_boundary、final_detect、final_js、final_stealth

3. scores.csv 对应的一行
- sample_idx: 第几个基础样本
- dataset_idx: 该基础样本在原 CIFAR-10 子集中的索引
- candidate_idx: 该基础样本下的第几个候选
- is_selected: 初始记为 0，后续若进入 top-k 再改为 1

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

### 8.1 基础样本如何得到筛选后的指纹

对同一个基础样本，代码会先收集该样本下所有候选的 S_total，形成 sample_candidate_scores。

随后执行:

1. 按 S_total 从高到低排序
- 使用 argsort(...)[::-1]
- 分数越高，表示该候选越适合作为指纹

2. 取前 k 个候选
- k = min(top_k_per_sample, 候选总数)
- 得到 top_idx

3. 将 top_idx 对应的候选写入 selected_* 系列字段
- selected_images
- selected_deltas
- selected_probs
- selected_scores
- selected_pred_labels

4. 记录该指纹来自哪个基础样本
- selected_base_sample_indices 记录它属于哪个 sample_idx
- selected_rank_in_sample 记录它在该基础样本内部的排序名次

5. 回写标记到 scores.csv
- 若某个候选进入 top-k，则其 is_selected 从 0 改成 1

因此，最终“筛选后的指纹”不是对全体候选做全局筛选，而是“每个基础样本内部各自保留 top_k_per_sample 个最优候选”。

### 8.2 筛选后指纹的组织方式

当前实现对筛选结果同时保留两种视图:

1. 扁平视图
- selected_images、selected_probs、selected_scores 等
- 适合后续统一计算标签与互信息参考量

2. 按基础样本分组视图
- selected_images_grouped、selected_probs_grouped、selected_scores_grouped 等
- 第 0 维对应基础样本，第 1 维对应该样本内部保留下来的 top-k 指纹

这样既能追踪“每个基础样本选中了哪些指纹”，也能把所有入选指纹合并起来做后续验证准备。

## 9. 指纹标签与互信息准备阶段

实现文件: ivsf_pipeline/generator.py

在完成 Top-K 筛选后，当前实现会继续整理一批离线验证参考信息，用于后续在线完整性验证阶段。

### 9.1 指纹标签的生成

1. 每个入选指纹都保留其输出概率向量
- 字段: selected_probs
- 含义: 每个最终指纹样本的 softmax 输出

2. 指纹标签来自该概率向量的 argmax
- 实现上先在候选整理阶段得到 selected_pred_labels
- 随后直接复制为 verification_pred_labels
- 含义: 后续验证时，这个标签就是该指纹在参考模型上的期望预测类别

可理解为:

- pred_label_i = argmax p(y|x_i^{fp})

其中 x_i^{fp} 表示第 i 个最终保留的指纹样本。

### 9.2 互信息的生成方式

1. 输入数据
- 使用所有最终入选指纹的 selected_probs
- 形状可理解为 N x C，其中 N 是最终指纹数，C 是类别数

2. 先估计边际类别分布
- 对所有指纹的概率向量按样本维求平均，得到 p(y)
- 这一步表示“参考模型在整组指纹上的平均输出分布”

3. 对每个指纹计算局部互信息代理值
- 当前实现不是论文中的完整 KSG 互信息估计
- 这里采用逐指纹的 KL 形式代理:

	MI_i = KL(p(y|x_i^{fp}) || p(y))

- 展开后为:

	MI_i = \sum_c p(y=c|x_i^{fp}) [\log p(y=c|x_i^{fp}) - \log p(y=c)]

4. 生成两个结果
- verification_mi_per_fingerprint: 每个指纹对应一个 MI_i
- verification_mi_ref: 所有 MI_i 的平均值，作为整组参考互信息

可理解为:

- mi_ref = (1 / N) \sum_i MI_i

### 9.3 当前实现与论文互信息的关系

1. 当前代码中的互信息是离线验证代理量
- 作用: 为后续在线验证预先保存“标签 + 互信息参考值”
- 优点: 简单、稳定、可直接由 softmax 概率生成

2. 当前没有实现论文中的完整输入输出互信息估计流程
- config.py 中的 mi_pca_components_input、mi_pca_components_output、mi_ksg_k 目前属于预留参数
- 现阶段实际参与导出的核心阈值参数是 mi_threshold

3. 因此当前 verification_mi_per_fingerprint 更准确地说是“基于输出分布的逐指纹互信息代理值”
- 不是严格意义上对原始输入 x 与输出 p(y|x) 的高维非参数互信息估计

### 9.4 落盘结果

这一阶段会把以下字段写入 fingerprints.npz:

- verification_pred_labels
- verification_mi_per_fingerprint
- verification_mi_ref
- verification_threshold

同时生成 verification_info.csv，每行包含:

- fingerprint_idx
- base_sample_idx
- dataset_idx
- fingerprint_rank
- pred_label
- mi_input_output

其中 mi_input_output 当前对应的就是上面定义的逐指纹互信息代理值。

## 10. 产物组织细节

实现文件: ivsf_pipeline/generator.py + ivsf_pipeline/io_utils.py

1. fingerprints.npz 包含:
- base_images, base_labels, ref_probs
- candidate_images, candidate_deltas, candidate_probs, candidate_scores
- selected_images, selected_deltas, selected_probs, selected_scores
- verification_pred_labels, verification_mi_per_fingerprint, verification_mi_ref, verification_threshold

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

## 11. 与论文方法的对应关系

1. 已对应部分
- 多目标扰动优化
- 候选评分与筛选
- 指纹分布构建（每个基础样本对应多个候选）

2. 当前简化点
- 边界项采用局部 top-1 margin 的绝对差实现
- 验证阶段仅完成离线参考标签与互信息代理值准备，未执行在线互信息判定

## 12. 建议执行命令（IVSF 目录）

1. 快速运行
- python run_ivsf_fingerprint.py --quick --device cpu

2. 自定义小规模调试
- python run_ivsf_fingerprint.py --device cpu --config-json "{\"num_base_samples\":20,\"num_candidates_per_sample\":3,\"top_k_per_sample\":1,\"steps\":10}"

3. 指定权重
- python run_ivsf_fingerprint.py --device cpu --checkpoint-path ./checkpoints/resnet20_cifar10.pth

## 13. 流程小结

当前代码已经形成可复现实验闭环:
参数解析 -> 模型加载 -> 数据子集 -> 候选扰动优化 -> 评分筛选 -> 指纹标签与互信息准备 -> 结构化落盘。

这份离线产物可直接作为后续在线完整性验证模块的输入基础。