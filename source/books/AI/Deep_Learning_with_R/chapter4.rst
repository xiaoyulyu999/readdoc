第四章：机器学习基础（深度笔记）
==========================

.. contents:: 目录
   :depth: 3
   :local:

本章概览
========

本章在前一章"三个实战案例"的基础上，把训练神经网络过程中积累的直觉，
系统化为一套可迁移到任意机器学习任务的**方法论**。核心内容包括：

- 机器学习问题的四大范式（不仅仅是分类和回归）
- 模型评估的正规化流程（防止信息泄漏）
- 数据预处理与特征工程的通用原则
- 过拟合的本质与五种应对策略
- 一套端到端的"机器学习通用工作流"（7 步法）

对于研究生阶段的学习者，这一章的价值不在于代码本身（R/Keras 的示例代码非常
基础），而在于其中隐含的**方法论纪律**：如何设计实验使得你对模型泛化能力的
估计是无偏的、可复现的、经得起同行评审的。这也是工业界与顶会论文中
"实验设置"（Experimental Setup）章节的理论基础。

1. 机器学习的四大分支
======================

1.1 监督学习（Supervised Learning）
------------------------------------

给定输入 :math:`X` 和已知标注 :math:`Y`，学习映射 :math:`f: X \rightarrow Y`。
这是目前深度学习聚光灯下几乎所有成功应用（OCR、语音识别、图像分类、机器翻译）
的基础范式。除了常见的分类/回归，还包括更"exotic"的变体：

- **序列生成（Sequence generation）**：给定图片生成描述文字（image captioning）。
  可以被重新表述为一系列分类问题（逐词/逐 token 预测）。
- **句法树预测（Syntax tree prediction）**：给定句子预测其句法分解结构。
- **目标检测（Object detection）**：给定图片画出物体的 bounding box。
  可视为分类问题（对候选框分类）+ 回归问题（预测框坐标）的联合任务。
- **图像分割（Image segmentation）**：像素级别的目标掩码预测。

.. note::
   **拓展知识点（研究生视角）**：现代深度学习中，上述"exotic"监督学习任务
   已经发展为独立的研究方向——序列生成对应 seq2seq / Transformer 解码器；
   目标检测对应 Faster R-CNN、YOLO、DETR 等系列；图像分割对应 U-Net、
   Mask R-CNN、Segment Anything (SAM) 等。理解本章的分类学有助于快速定位
   新论文属于哪一大类问题，从而判断评估指标该如何选择。

1.2 无监督学习（Unsupervised Learning）
-----------------------------------------

在没有目标标签的情况下，寻找输入数据中有趣的变换，用于数据可视化、压缩、
去噪，或更好地理解数据内部的相关结构。**降维**（PCA、t-SNE、UMAP）和
**聚类**（K-means、层次聚类、DBSCAN）是经典代表。

1.3 自监督学习（Self-supervised Learning）
---------------------------------------------

自监督学习是监督学习的一种特殊情形：标签依然存在，但它们是**从输入数据本身
通过启发式算法自动生成的**，而非人工标注。典型例子：

- **自编码器（Autoencoder）**：目标就是输入本身（重构任务）。
- **下一帧/下一词预测**：给定历史，预测未来（时间维度上的自我监督）。

.. important::
   这一分类在原书写作时（TensorFlow/Keras 时代）还只是"配角"，但在
   Master 阶段的今天，自监督学习已经是深度学习的**主流范式**之一：
   BERT 的掩码语言建模（MLM）、GPT 的自回归下一词预测、对比学习
   （SimCLR、MoCo）、以及多模态领域的 CLIP，本质上都是自监督学习。
   建议将本节视为理解大模型预训练范式的历史起点。

1.4 强化学习（Reinforcement Learning, RL）
---------------------------------------------

智能体（agent）通过与环境交互获得反馈信号（reward），学习选择能够最大化
累积奖励的动作序列。DeepMind 在 Atari 游戏和围棋（AlphaGo）上的突破使其
广受关注。原书成文时强化学习"仍主要是研究领域"，但此后已扩展到机器人控制、
自动驾驶、资源调度、以及**基于人类反馈的强化学习（RLHF）**——这正是训练
ChatGPT/Claude 等对话模型对齐人类偏好的核心技术，值得重点关注。

分类与回归术语表（速查）
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 术语
     - 定义
   * - Sample / Input
     - 输入模型的一个数据点
   * - Prediction / Output
     - 模型输出
   * - Target
     - 真实标签（ground truth）
   * - Loss value
     - 预测值与目标值之间距离的度量
   * - Classes
     - 分类问题中可选的标签集合
   * - Label
     - 某个样本对应的具体类别标注
   * - Binary classification
     - 每个样本被分到两个互斥类别之一
   * - Multiclass classification
     - 每个样本被分到 >2 个互斥类别之一
   * - Multilabel classification
     - 每个样本可同时拥有多个标签
   * - Scalar regression
     - 目标是连续标量（如房价）
   * - Vector regression
     - 目标是一组连续值（如 bounding box 坐标）
   * - Mini-batch
     - 一次梯度更新所用的小样本集（通常 8–128，2 的幂次便于 GPU 显存对齐）

2. 模型评估：如何可靠地衡量泛化能力
======================================

2.1 为什么不能只用训练集和测试集
------------------------------------

模型开发过程本质上包含**超参数调优**（层数、每层单元数、学习率等），而调优
所依赖的反馈信号——验证集上的表现——会逐渐"泄漏"进模型中。这被称为
**信息泄漏（information leak）**。每调一次超参数看一次验证集表现，
就泄漏若干比特（bit）信息；重复多次后，模型会对验证集产生隐性过拟合，
即便它从未被直接用于梯度更新。

因此需要三路划分：**训练集（training）→ 验证集（validation，用于调参）
→ 测试集（test，只用一次，衡量最终泛化能力）**。测试集绝不能以任何
形式（哪怕间接）参与模型选择。

.. tip::
   **Master 级别提醒**：这正是为什么 Kaggle 比赛设有 Public/Private
   Leaderboard 的双重机制，也是为什么顶会论文要求在"held-out test set"
   上只报告一次最终结果。如果你在写论文时反复用测试集调参再汇报最优结果，
   本质上就是制造了一次学术意义上的信息泄漏，是不严谨甚至可能构成
   p-hacking 的行为。

2.2 简单留出验证（Hold-out Validation）
-------------------------------------------

.. code-block:: r

   indices <- sample(1:nrow(data), size = 0.80 * nrow(data))               # 1
   evaluation_data  <- data[-indices, ]                                    # 2
   training_data <- data[indices, ]                                       # 3

   model <- get_model()                                                    # 4
   model %>% train(training_data)
   validation_score <- model %>% evaluate(validation_data)

   model <- get_model()                                                    # 5
   model %>% train(data)
   test_score <- model %>% evaluate(test_data)

   # 1 打乱数据顺序通常是必要的
   # 2 划分验证集
   # 3 划分训练集
   # 4 在训练集上训练，在验证集上评估
   # 5 调好超参数后，通常用全部非测试数据重新从头训练最终模型

**局限**：数据量小时，验证/测试集样本可能不具统计代表性。判断标准：
如果多次不同的随机划分给出差异很大的性能估计，说明数据量不足以支撑
简单留出法。

2.3 K 折交叉验证（K-fold Cross-Validation）
------------------------------------------------

将数据划分为 K 个大小相等的子集；每次用其中 1 份做验证、其余 K-1 份训练，
共训练 K 个模型，最终得分取 K 次结果的平均值。

.. code-block:: r

   k <- 4
   indices <- sample(1:nrow(data))
   folds <- cut(indices, breaks = k, labels = FALSE)

   validation_scores <- c()
   for (i in 1:k) {
     validation_indices <- which(folds == i, arr.ind = TRUE)
     validation_data <- data[validation_indices, ]                        # 1
     training_data <- data[-validation_indices, ]                         # 2

     model <- get_model()                                                 # 3
     model %>% train(training_data)
     results <- model %>% evaluate(validation_data)
     validation_scores <- c(validation_scores, results$accuracy)
   }

   validation_score <- mean(validation_scores)                            # 4

   model <- get_model()                                                   # 5
   model %>% train(data)
   results <- model %>% evaluate(test_data)

   # 1 取出该折作为验证集
   # 2 剩余数据作为训练集
   # 3 每折都创建全新（未训练）的模型实例
   # 4 验证得分 = K 折验证得分的平均
   # 5 最终在全部非测试数据上训练一次

2.4 带重复打乱的迭代 K 折验证
--------------------------------

数据量非常有限、需要极精确估计时使用：每次重新打乱数据后做一次完整的
K 折验证，重复 P 次，最终取 P×K 次结果的平均。计算成本是普通 K 折的
P 倍，但在 Kaggle 类竞赛中被证明非常有效。

2.5 评估协议设计中的三个陷阱
--------------------------------

- **数据代表性（Data representativeness）**：训练/测试集都应能代表整体
  数据分布。经典错误：数据按类别排序后直接切前 80% 做训练集——会导致
  训练集和测试集类别分布完全不重合。**务必先打乱再划分。**
- **时间箭头（The arrow of time）**：预测未来（天气、股价等）时**绝不能**
  随机打乱数据，否则会制造"时间穿越"式的信息泄漏——测试集数据的时间
  必须严格晚于训练集。这是时间序列 / 金融建模中最常见的评估陷阱。
- **数据冗余（Redundancy）**：若数据中存在重复样本，打乱后划分可能导致
  训练集与验证集出现重叠，本质上是"用训练数据的复制品做测试"，会严重
  高估模型的真实泛化能力。**务必保证训练集与验证集互不相交。**

3. 数据预处理、特征工程与特征学习
=====================================

3.1 数据预处理四要素
-----------------------

神经网络的输入输出必须是浮点数张量（Vectorization，向量化）。核心处理包括：

1. **向量化（Vectorization）**：将原始数据（文本、图像、声音）转为张量，
   例如用 one-hot 编码把整数序列转为浮点张量。
2. **值归一化（Value normalization）**：网络对"取值过大"或"分布异质"的
   数据非常敏感，容易触发过大的梯度更新导致不收敛。理想输入应满足：

   - **取值较小**：大多数值应落在 0–1 区间；
   - **同质性**：各特征取值范围应大致相同。

   更严格的标准化做法是让每个特征独立地满足均值为 0、标准差为 1：

   .. code-block:: r

      x <- scale(x)   # 假设 x 是形状为 (samples, features) 的二维矩阵

   .. important::
      **关键原则**：归一化的均值和标准差**必须只用训练集计算**，
      然后同样应用到训练集和测试集：

      .. code-block:: r

         mean <- apply(train_data, 2, mean)
         std  <- apply(train_data, 2, sd)
         train_data <- scale(train_data, center = mean, scale = std)
         test_data  <- scale(test_data,  center = mean, scale = std)

      如果连均值/标准差都用全体数据（含测试集）计算，本质上也是一种
      信息泄漏——测试集的统计信息间接影响了训练过程。这个原则同样适用于
      PCA、特征选择等任何依赖全局统计量的预处理步骤。

3. **缺失值处理**：只要 0 不是一个有意义的合法取值，用 0 填充缺失值通常
   是安全的——网络会在训练中学会把 0 解释为"缺失"并逐渐忽略它。但**若
   测试集中可能出现训练集未见过的缺失模式**，需要人为构造带缺失值的
   训练样本（复制部分样本并主动丢弃某些特征），让网络提前"见过"缺失情形。

3.2 特征工程（Feature Engineering）
---------------------------------------

特征工程 = 利用你对数据和算法的先验知识，对原始数据做**硬编码（非学习）
的变换**，从而降低模型的学习难度。经典类比：读钟表时间。

- 若直接用像素做输入 → 需要复杂的 CNN，计算成本高；
- 若先提取指针尖端的 (x, y) 坐标 → 一个简单模型即可学会；
- 若进一步转换为以钟面中心为原点的极坐标（角度 θ）→ 问题简单到
  只需四舍五入+查表，甚至不再需要机器学习。

**深度学习时代特征工程是否还重要？** 是的，原因有二：

1. 好的特征能让你用更少的计算资源更优雅地解决问题；
2. 好的特征能让你用**更少的数据**解决问题——当训练样本稀缺时，
   人工先验的价值急剧上升。

.. note::
   **拓展**：在 Master 阶段接触的大模型时代，"特征工程"的角色很大程度上
   被"表示学习"（representation learning）和"预训练 + 微调"范式取代，
   但在**表格数据（tabular data）**、**小样本/低资源场景**、以及
   **可解释性要求高的领域（医疗、金融风控）**中，手工特征工程仍是
   梯度提升树（XGBoost、LightGBM）等模型的核心竞争力来源，不应被忽视。

4. 过拟合与欠拟合
====================

4.1 优化与泛化的张力
------------------------

- **优化（Optimization）**：让模型在训练数据上表现尽可能好；
- **泛化（Generalization）**：模型在未见过的数据上的表现。

训练初期二者正相关（欠拟合阶段，模型还有提升空间）；经过若干轮迭代后，
验证指标停滞乃至恶化——模型开始学习训练数据中**特有但不具泛化性**的
模式，即**过拟合**。对抗过拟合的核心思路是**正则化（regularization）**：
限制模型能够存储的信息量或对其施加约束，迫使其只保留最"显著"、
最可能泛化的模式。

原书总结的四种（细分为五种）对抗过拟合的经典策略：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 策略
     - 核心思想
   * - 获取更多训练数据
     - 最有效但往往最不可行的方案
   * - 降低网络容量（capacity）
     - 减少可学习参数数量，压缩"记忆空间"
   * - 权重正则化（L1 / L2）
     - 对损失函数施加"权重大小惩罚"，实现 Occam's razor
   * - Dropout
     - 训练时随机丢弃部分神经元输出，打破"阴谋式"共适应模式
   * - （拓展）早停 Early Stopping
     - 原书未展开，但与验证曲线监控天然配套，是工程实践中最常用的手段之一

4.2 降低网络容量
--------------------

"容量（capacity）"= 模型可学习参数的数量，由层数与每层单元数决定。
容量过大的模型可以直接"记住"训练集（类似于查表），但完全不具泛化能力。

.. code-block:: r

   # 原始参考网络
   model <- keras_model_sequential() %>%
     layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
     layer_dense(units = 16, activation = "relu") %>%
     layer_dense(units = 1,  activation = "sigmoid")

   # 容量更小的网络：更晚开始过拟合，且过拟合后性能退化更慢
   model_small <- keras_model_sequential() %>%
     layer_dense(units = 4, activation = "relu", input_shape = c(10000)) %>%
     layer_dense(units = 4, activation = "relu") %>%
     layer_dense(units = 1, activation = "sigmoid")

   # 容量远超问题所需的网络：几乎立刻开始过拟合，且过拟合更严重、验证损失更嘈杂
   model_big <- keras_model_sequential() %>%
     layer_dense(units = 512, activation = "relu", input_shape = c(10000)) %>%
     layer_dense(units = 512, activation = "relu") %>%
     layer_dense(units = 1,   activation = "sigmoid")

**寻找合适容量没有公式解**，标准工作流是：从较少的层/参数起步，逐步
增加规模，直到验证损失的收益开始明显递减为止（即"先过拟合，再收缩"的
策略，详见第 5 节）。

4.3 权重正则化（L1 / L2）
----------------------------

基于 Occam's razor：同样能解释训练数据的模型中，参数分布更"规整"（更小、
更稀疏）的模型更不容易过拟合。

- **L1 正则化**：惩罚项正比于权重的绝对值之和（|w|），倾向于产生
  **稀疏**权重（许多权重被压到恰好为 0），因此天然具有特征选择效果。
- **L2 正则化**（又称权重衰减 weight decay）：惩罚项正比于权重的平方和
  （:math:`w^2`），倾向于让所有权重都"均匀变小"而非稀疏化。

.. code-block:: r

   model <- keras_model_sequential() %>%
     layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
                 activation = "relu", input_shape = c(10000)) %>%
     layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
                 activation = "relu") %>%
     layer_dense(units = 1, activation = "sigmoid")

   # 其他可选正则化器：
   regularizer_l1(0.001)
   regularizer_l1_l2(l1 = 0.001, l2 = 0.001)

``regularizer_l2(0.001)`` 表示该层权重矩阵中每个系数会向总损失额外贡献
``0.001 * weight_coefficient_value`` 的惩罚项。**注意**：该惩罚项只在
训练阶段被加入损失，因此同一模型的训练损失会明显高于测试损失，这是正常
现象而非 bug。

4.4 Dropout
---------------

由 Hinton 团队提出，是最有效、应用最广泛的正则化技术之一。训练时，
以概率 :math:`p`（dropout rate，通常 0.2–0.5）随机将该层部分输出置零；
测试时不丢弃任何单元，但需要对输出做缩放补偿。

.. code-block:: r

   # 训练阶段（概念示意）：
   layer_output <- layer_output * sample(0:1, length(layer_output), replace = TRUE)
   layer_output <- layer_output / 0.5   # "inverted dropout"：训练时放大而非测试时缩小

   # 在 Keras 中直接使用：
   layer_dropout(rate = 0.5)

   model <- keras_model_sequential() %>%
     layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
     layer_dropout(rate = 0.5) %>%
     layer_dense(units = 16, activation = "relu") %>%
     layer_dropout(rate = 0.5) %>%
     layer_dense(units = 1, activation = "sigmoid")

**Hinton 的直觉来源**：银行柜员定期轮岗是为了防止员工之间"串谋舞弊"；
类比到神经网络，每次训练随机丢弃不同的神经元子集，能防止神经元之间
形成脆弱的"共谋式"依赖（conspiracy），从而迫使网络学习更鲁棒、
更具冗余度的表示。

.. note::
   **拓展知识点**：Dropout 在 Transformer/大模型时代仍被广泛使用（例如
   attention dropout、feed-forward dropout），但其角色已被 LayerNorm、
   残差连接、更大规模数据、以及权重衰减（结合 AdamW 优化器）部分替代。
   理解 Dropout 与 L2 正则化的联系（Dropout 在一定近似下等价于对权重
   施加自适应的 L2 惩罚）是深入理解现代正则化理论的重要基础。

5. 机器学习通用工作流（7 步法）
===================================

这是本章的核心方法论产出，建议作为**任何机器学习/深度学习项目的检查清单**。

.. list-table::
   :header-rows: 1
   :widths: 8 30 62

   * - 步骤
     - 名称
     - 要点
   * - 1
     - 定义问题、收集数据
     - 明确输入/输出、问题类型（分类/回归/…）；两条隐含假设：
       (a) 输出可由输入预测；(b) 现有数据足以学到这种关系。
       注意**非平稳问题（nonstationary）**——如随季节变化的推荐系统，
       应定期用近期数据重训练，或将"时间"显式作为输入特征。
   * - 2
     - 选择成功度量
     - 度量标准决定损失函数的选择方向。平衡分类问题常用 accuracy / ROC AUC；
       类别不平衡问题常用 precision / recall；排序/多标签问题常用 mAP。
   * - 3
     - 确定评估协议
     - 数据充足选 hold-out；数据稀缺选 K 折；数据极稀缺且要求高精度选
       迭代 K 折。
   * - 4
     - 数据准备
     - 向量化、归一化（必要时）、处理缺失值、按需特征工程。
   * - 5
     - 开发优于基线的模型（获得"统计力量"）
     - 目标是显著超过随机基线（如 MNIST 上 >0.1，IMDB 上 >0.5）。
       需要确定三个关键选择：**最后一层激活函数、损失函数、优化器配置**
       （见下表）。若始终无法超过基线，需回头怀疑第 1 步的两条假设是否成立。
   * - 6
     - 刻意过拟合
     - 加层、加宽、增加训练轮数，直到验证集性能开始明显恶化——
       这标志着你已经跨过了"容量边界"，为第 7 步的收缩提供了参照系。
   * - 7
     - 正则化与超参数调优
     - 加 Dropout、尝试不同架构、加 L1/L2、调学习率与单元数、
       迭代特征工程。**每次利用验证集反馈调参都会造成微量信息泄漏**，
       次数过多会导致对验证过程本身的过拟合。最终在全部非测试数据
       （训练集+验证集）上重训练一次，仅在测试集上评估一次。

不同问题类型的输出层与损失函数对照表
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - 问题类型
     - 最后一层激活函数
     - 损失函数
   * - 二分类（Binary classification）
     - sigmoid
     - binary_crossentropy
   * - 多分类单标签（Multiclass, single-label）
     - softmax
     - categorical_crossentropy
   * - 多分类多标签（Multiclass, multilabel）
     - sigmoid
     - binary_crossentropy
   * - 回归到任意实数
     - 无（线性输出）
     - mse
   * - 回归到 [0, 1] 区间
     - sigmoid
     - mse 或 binary_crossentropy

.. warning::
   **为什么不直接优化目标度量（如 ROC AUC）？** 损失函数必须满足两个
   工程约束：(1) 可在单个 mini-batch（理想情况下单个样本）上计算；
   (2) 必须可微，才能支持反向传播。像 ROC AUC 这类基于排序的指标
   通常不满足这两点，因此工程上退而求其次，优化一个"代理指标"
   （proxy metric），如用 crossentropy 代替 ROC AUC——经验上，
   crossentropy 越低，ROC AUC 通常也越高，但**这只是经验相关性，
   并非数学等价**，在类别极度不平衡等场景下可能出现背离，需要
   额外关注（例如引入 focal loss 等针对性损失函数）。

6. 本章要点小结
==================

1. 明确问题与数据：先收集/标注数据，再谈建模。
2. 选定成功度量指标，它将决定损失函数的设计方向。
3. 确定评估协议（hold-out / K 折 / 迭代 K 折），并划分好训练/验证/测试集，
   全程警惕**信息泄漏**、**时间箭头**、**数据冗余**三大陷阱。
4. 先开发一个"优于基线"的模型，确认问题本身具有"统计可学习性"。
5. 主动把模型训练到过拟合，从而摸清容量的上界。
6. 通过正则化（降容量、L1/L2、Dropout 等）系统性地收缩模型，
   在欠拟合与过拟合之间找到最佳平衡点。
7. 全流程贯穿一个核心纪律：**验证集用于调参、测试集只用于最终一次性
   评估**，这是保证你汇报的泛化性能真实可信的根本保证。

.. seealso::
   **面向 Master 学生的延伸阅读方向**：

   - 偏差-方差权衡（Bias-Variance Tradeoff）与本章"欠拟合/过拟合"的
     统计学习理论对应关系（可参考 *Elements of Statistical Learning*）。
   - 现代正则化手段：Batch/Layer Normalization、残差连接、
     标签平滑（Label Smoothing）、数据增强（Data Augmentation）、
     混合精度与权重衰减解耦的 AdamW 优化器。
   - 大模型时代的评估范式演进：预训练-微调-对齐（RLHF）三阶段中，
     每一阶段如何重新定义"训练/验证/测试"边界与信息泄漏风险
     （如 benchmark contamination 问题）。
   - 时间序列与在线学习场景下，"时间箭头"原则如何演化为
     walk-forward validation、滚动窗口回测等更精细的评估协议。