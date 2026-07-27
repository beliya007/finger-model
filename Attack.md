# 模型攻击与移除方案对比 (BIVBench vs IVSF)

## #sym:BIVBench 模块所设计的攻击模型

BIVBench 的攻击体系基于 WRT 抽象层，核心是“创建攻击对象 + 执行 remove/extract 流程”。

### 1) 框架抽象层
- 攻击基类位于 [BIVBench-main/wrt/attacks/attack.py](BIVBench-main/wrt/attacks/attack.py)
- 关键抽象：
  - RemovalAttack：白盒条件下移除水印/指纹
  - ExtractionAttack：提取水印/指纹

### 2) Removal 攻击模块（当前仓库可见）
模块索引位于 [BIVBench-main/wrt/attacks/removal/__init__.py](BIVBench-main/wrt/attacks/removal/__init__.py)。

按机制可分为：

1. 微调/再训练类
- FTLLAttack / FTALAttack / RTLLAttack / RTALAttack / FZLLAttack
- ModelExtraction（retraining）
- TransferLearningAttack

2. 结构/参数修改类
- FinePruningAttack
- WeightPruning
- WeightQuantization
- WeightShifting
- Regularization
- FeatureRegularization

3. 蒸馏与替身模型类
- ModelDistillation
- KnockoffNets

4. 输入变换/预处理类
- InputNoising
- InputHorizontalFlipping
- InputMeanSmoothing / InputGaussianSmoothing / InputMedianSmoothing
- InputQuantization
- JPEGCompression
- FeatureSqueezing
- RandomOcclusion
- InputReconstruction

5. 标签与集成策略类
- LabelSmoothingAttack
- LabelNoisingAttack
- EnsembleAttack

6. 后门清洗/反制类
- NeuralCleanseUnlearning / Partial / Multi
- NeuralCleansePruning / Partial / Multi
- Laundering / Partial / Multi
- AdversarialTraining

### 3) BIVBench 脚本层如何驱动这些攻击
- 主入口是 [BIVBench-main/train_tampered_model.py](BIVBench-main/train_tampered_model.py)
- 通过 attack_config 动态装配：
  - attack_config.create(...) 构造攻击对象
  - attack_config.remove(...) 执行移除流程
- 也即：配置文件决定攻击范式，代码执行统一接口。

补充：在配置层，已提供多数据集 attack_configs（CIFAR10/GTSRB/MNIST/ImageNet），如：
- [BIVBench-main/configs/cifar10/attack_configs/ftll.yaml](BIVBench-main/configs/cifar10/attack_configs/ftll.yaml)
- [BIVBench-main/configs/cifar10/attack_configs/weight_pruning.yaml](BIVBench-main/configs/cifar10/attack_configs/weight_pruning.yaml)
- [BIVBench-main/configs/cifar10/attack_configs/distillation.yaml](BIVBench-main/configs/cifar10/attack_configs/distillation.yaml)

---

## IVSF 当前属于哪一种攻击

当前 IVSF 攻击实现集中在后门投毒路径：
- 模块入口：[IVSF/attack/__init__.py](IVSF/attack/__init__.py)
- 核心实现：[IVSF/attack/badnet_attack.py](IVSF/attack/badnet_attack.py)
- 数据投毒：[IVSF/attack/data/poisoned_dataset.py](IVSF/attack/data/poisoned_dataset.py)

本质类型：
1. 数据投毒后门（BadNets）
- 在训练数据中按 poison_ratio 注入 trigger
- 将触发样本重标到 trigger_label
- 学得“触发器 -> 目标类”后门行为

2. 在干净模型上继续微调（poisoned fine-tuning）
- 从 clean checkpoint 出发
- 经过多 epoch（你已执行到 400 轮）训练得到 attacked checkpoint

所以，IVSF 当前不是“广义移除攻击框架”，而是“后门篡改模型构造器 + 指纹对比验证器”。

---

## IVSF 已实现/未实现对照（按 BIVBench 能力映射）

已实现：
1. BadNets 式后门投毒训练（已支持 GPU、AMP、多轮训练）
2. 指纹在 clean vs attacked 模型上的差异对比

尚未实现（优先级从高到低建议）：
1. 权值剪枝 Weight Pruning
- 意义：最经典模型篡改/压缩方式，最先验证指纹鲁棒性

2. 微调家族 FTLL / FTAL / FZLL / RTLL / RTAL
- 意义：不引入后门，仅通过训练策略变化破坏指纹

3. 蒸馏 Model Distillation
- 意义：黑盒所有权验证里最关键的替身模型场景

4. 量化 Weight Quantization
- 意义：部署侧常见，直接影响指纹可分辨性

5. Fine-Pruning / TransferLearning
- 意义：更强对抗场景，可评估指纹跨域稳定性

6. 输入变换攻击族（JPEG、Noising、Smoothing、FeatureSqueezing）
- 意义：测试推理前处理对指纹统计特征的影响

7. Neural Cleanse / Laundering
- 意义：后门清洗防御反向作用下，检验指纹保真度

---

## 建议的下一步落地顺序

1. 先做 Weight Pruning + FTLL 两个最小可行攻击
- 原因：实现成本低、可直接形成对照基线

2. 再做 Distillation
- 原因：对黑盒验证最有说服力

3. 最后补 Quantization 与 Fine-Pruning
- 原因：完成“训练篡改 + 压缩部署 + 结构裁剪”三维评估闭环

---

## 一句话结论

BIVBench 是“多攻击范式的统一可配置平台”；当前 IVSF 属于其中的“后门投毒 + 微调”子类。若要对齐 BIVBench 的评估深度，下一步应优先补齐 Weight Pruning、FT 系列和 Distillation 三类攻击。
在 BIVBench 的 backdoor_attack 这套流程里，主要有 4 种攻击方式（对应 task）：

backdoor，BadNets 风格后门攻击（右下角方块触发器 + 目标标签重标）
trojan
clean_label_backdoor
data_poison
另外，data_poison 下面还细分两种投毒粒度：

sample 级投毒
class 级投毒
并且还支持一个“vulnerable_targeted_attack”开关，用来做更定向的 targeted 攻击。



第一种是后门攻击，它会损害模型，假设目标模型是已知的并且在攻击者的控制之下。我们研究的第二种场景是模型压缩，作为验证的困难案例（即模型变化最小）。（一）对于后门攻击，我们使用TrojanZoo [30]平台在图像分类模型上部署BadNet [31]、TrojanNN [6]和CleanLabel [5]攻击，这些都是典型的后门攻击，它们会损害训练数据以在测试时改变模型的行为。在我们的实现中，这3个对抗性模型的攻击成功率均超过95%。这些攻击的默认设置在附录的表VII中给出。（二）我们在三个数据集上应用权重剪枝来压缩模型。剪枝首先通过常规网络训练来学习连接性。接下来，移除所有权重低于阈值的连接，然后重新训练网络以学习剩余稀疏连接的最终权重。通过仅保留重要连接来剪枝网络，其中MNIST数据集上的LeNet模型移除了15%的权重，CIFAR-10数据集上的ResNet模型移除了20%的权重，FFHQ数据集上的VGG16模型移除了约24%的权重。剪枝导致每个模型的准确率仅发生微小变化（< 2%）。