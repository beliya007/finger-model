## Plan: IVSF 指纹生成首版落地（ResNet20 + CIFAR-10）

目标是先完成离线阶段的“指纹生成 + 筛选 + 产物保存”，并预留后续验证接口，不进入在线验证实现。实现方式是复用 BIVBench 的模型与数据链路，但在 IVSF 下独立搭建流水线，降低耦合和回归风险。
计划已同步保存到 /memories/session/plan.md。

**Steps**
1. Phase A - 基线链路对齐（后续步骤依赖本阶段）
1.1 对齐 ResNet20+CIFAR-10 的加载方式、输入标准化和推理输出格式。
1.2 设计双模型来源：支持加载现有 checkpoint，也支持先训练再加载。
1.3 定义统一 I/O 契约：输入样本批次，输出 logits/softmax、候选扰动、筛选结果和统计信息。

2. Phase B - 指纹生成核心（依赖 Phase A）
2.1 实现四项损失：L_boundary、L_detect、L_JS、L_stealth，并支持可配置权重。
2.2 对每个原始样本进行多初值扰动优化，得到“每个样本对应的指纹分布”（不是单点）。
2.3 增加约束：像素范围裁剪、扰动范数限制、数值稳定参数化。
2.4 输出候选指纹池与每轮优化轨迹统计。

3. Phase C - 候选筛选与排序（依赖 Phase B）
3.1 实现结构不确定性评分 S_BU（熵 + margin）。
3.2 实现动态敏感性评分 S_sens（局部扰动下 JS 期望）。
3.3 实现隐蔽性惩罚 S_stealth。
3.4 合成总分并按预算 Top-K 选择最终指纹集。

4. Phase D - 管线与产物（与 Phase C 可部分并行）
4.1 提供单入口脚本，支持 quick 配置（100 样本、50 迭代）和自定义参数。
4.2 落盘产物：fingerprints.npz、scores.csv、meta.json、可视化图（原图/扰动/差分/评分示例）。
4.3 预留验证接口所需 reference artifact 结构，不实现在线验证逻辑。

5. Phase E - 可复现与健壮性（依赖 Phase B-D）
5.1 固定随机种子、记录设备与超参。
5.2 添加最小检查：损失有效、梯度非零、输出概率合法、扰动边界合法。
5.3 端到端 quick 冒烟：确保完整产物可生成并可读取。

**Relevant files**
- [IVSF/my_method.md](IVSF/my_method.md) - 方案定义与实现边界依据
- [BIVBench-main/train_protect_model.py](BIVBench-main/train_protect_model.py) - 训练与模型封装入口
- [BIVBench-main/configs/cifar10/train_configs/resnet20.yaml](BIVBench-main/configs/cifar10/train_configs/resnet20.yaml) - ResNet20/CIFAR-10 配置对齐
- [BIVBench-main/wrt/training/models/torch/classifier/resnet20.py](BIVBench-main/wrt/training/models/torch/classifier/resnet20.py) - 模型结构与 normalize 逻辑
- [BIVBench-main/wrt/training/datasets/cifar10.py](BIVBench-main/wrt/training/datasets/cifar10.py) - 数据增强与标准化统计
- [BIVBench-main/backdoor_attack/utils/utils.py](BIVBench-main/backdoor_attack/utils/utils.py) - 可复用优化循环思路
- [BIVBench-main/utils_torch.py](BIVBench-main/utils_torch.py) - checkpoint 加载结构参考

**Verification**
1. quick 模式运行后，四类产物全部生成。
2. 验证每个原始样本都对应多个候选扰动。
3. 抽样检查 softmax 行和约为 1，输入与扰动均在约束范围。
4. 重复两次同配置同种子，比较 Top-K 重合率和评分方差。
5. 可视化检查边界样本是否具备预期敏感性。

**Decisions**
- 已确认做：生成 + 筛选 + 预留验证接口。
- 已确认不做：在线验证逻辑。
- 已确认目录：在 IVSF 下独立实现。
- 已确认模型来源：加载已有权重和训练后加载都支持。
- 已确认首版预算：quick 配置优先。

如果你认可这份计划，我下一步会进入实现代理流程，按这个顺序落地代码与脚本。