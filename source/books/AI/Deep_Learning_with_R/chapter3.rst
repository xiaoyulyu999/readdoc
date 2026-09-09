Chapter 3 — Getting Started with Neural Networks
================================================

本章主要学习三类任务：

#. Binary classification（二分类）
#. Multiclass classification（多分类）
#. Scalar regression（标量回归）

最重要的训练流程：

.. code-block:: text

   Input data
       ↓
   Model / Network
       ↓
   Prediction
       ↓
   Loss Function
       ↓
   Loss
       ↓
   Backpropagation
       ↓
   Gradient
       ↓
   Optimizer
       ↓
   Update Weights
       ↓
   下一轮 Forward Pass


1. Neural Network 的基本组成
============================

1.1 核心组成
------------

训练一个 neural network，最核心的几个对象是：

``Layer``
   神经网络的基本计算模块。接收 tensor，经过计算后输出新的 tensor。

``Model / Network``
   由多个 layer 连接起来形成的整体模型。

``Input data``
   输入给模型的数据。

``Target``
   正确答案，也就是模型希望学会预测的结果。

``Loss function``
   衡量 prediction 和 target 差得有多远。

``Optimizer``
   根据 gradient 决定如何更新 weights。

可以把它们理解为：

.. code-block:: text

   Input
     ↓
   Model
     ↓
   Prediction
     ↓
   和 Target 比较
     ↓
   Loss
     ↓
   Backpropagation
     ↓
   Gradient
     ↓
   Optimizer
     ↓
   更新 Weights


1.2 Layer 是什么
----------------

``Layer`` 可以理解成一个“数据变换模块”。

它接收输入 tensor：

.. math::

   x

通过权重和偏置进行变换：

.. math::

   z = Wx+b

再通过 activation function：

.. math::

   a=f(z)

得到输出。

例如 Dense layer 常见形式：

.. math::

   a=\mathrm{ReLU}(Wx+b)

很多 layer 内部含有可以学习的参数：

* ``weights``
* ``biases``

这些参数会随着训练不断更新。

可以理解为：

**模型真正“学到的东西”主要存储在 weights 中。**


1.3 不同数据适合不同 Layer
--------------------------

.. list-table::
   :header-rows: 1

   * - 数据类型
     - 常见 tensor shape
     - 常用 layer
   * - 普通表格 / vector data
     - ``(samples, features)``
     - Dense layer
   * - Sequence data
     - ``(samples, timesteps, features)``
     - RNN / LSTM
   * - Image data
     - 通常是 4D tensor
     - Convolution layer

考试重点：

不同类型的 layer，本质上是为了处理不同的数据结构。


1.4 Input Shape 和 Output Shape
--------------------------------

layer 之间必须保证 shape 兼容。

例如：

.. code-block:: r

   layer_dense(
     units = 32,
     input_shape = c(784)
   )

表示：

输入：

.. code-block:: text

   784 dimensions

经过 layer 后：

.. code-block:: text

   32 dimensions

即：

.. code-block:: text

   784
    ↓
   Dense(32)
    ↓
   32

这里 ``units = 32`` 表示：

**这一层输出的 representation 有 32 个维度。**

后面的 layer 通常可以自动推断前一层的 output shape。


1.5 Model 是什么
----------------

``Model`` 是多个 layer 按照一定结构连接起来形成的网络。

最简单的是 Sequential model：

.. code-block:: text

   Input
     ↓
   Layer 1
     ↓
   Layer 2
     ↓
   Layer 3
     ↓
   Output

更复杂的 model 可能有：

* multi-input
* multi-output
* two-branch network
* multi-head network

Master's level 需要理解：

**Model architecture 决定了模型允许学习什么样的函数和 representation。**

这可以用 ``hypothesis space`` 表示。

换句话说：

你选择怎样的 architecture，就等于限制了模型“可以寻找什么样的答案”。


2. Loss Function
================

2.1 Loss 是什么
---------------

假设：

.. math::

   y = target

.. math::

   \hat{y}=prediction

那么：

.. math::

   C=L(y,\hat{y})

其中：

``C``
   loss

``L``
   loss function

Loss 的作用：

**把模型错得有多严重，压缩成一个 scalar。**

这个 scalar 会成为训练过程中的优化目标。


2.2 为什么 Loss 必须是 Scalar
-----------------------------

Gradient-based optimization 最终需要一个明确的数值目标。

即使模型有多个 output，也通常需要把多个 loss 合并成一个 scalar：

.. math::

   C=\sum_i \lambda_i C_i

然后 optimizer 才能统一进行优化。


2.3 Loss Function 必须与任务匹配
--------------------------------

常见对应关系：

.. list-table::
   :header-rows: 1

   * - 任务
     - 常见 loss
   * - Binary classification
     - Binary crossentropy
   * - Multiclass classification
     - Categorical crossentropy
   * - Regression
     - Mean squared error
   * - Sequence problem
     - 视任务而定，例如某些任务使用 CTC

重要理解：

**模型不会理解你的真实意图，它只会尽可能降低你给它的 objective。**

所以如果 loss function 设计错误，模型可能学到你不希望它学到的策略。


3. Gradient、Backpropagation 和 Optimizer
========================================

3.1 Gradient 是什么
-------------------

如果：

.. math::

   C = Loss

.. math::

   w = Weight

那么：

.. math::

   \frac{\partial C}{\partial w}

表示：

**当 ``w`` 发生一个很小的变化时，Loss 会如何变化。**

也可以理解为：

**Loss 对 weight 的敏感程度。**


3.2 Gradient 的正负代表什么
---------------------------

如果：

.. math::

   \frac{\partial C}{\partial w}>0

表示：

增加 ``w``，会让 Loss 增大。

所以 gradient descent 会反方向更新：

.. math::

   w_{\mathrm{new}}
   =
   w_{\mathrm{old}}
   -
   \eta
   \frac{\partial C}{\partial w}

如果 gradient 是负数：

.. math::

   \frac{\partial C}{\partial w}<0

说明增加 ``w`` 会降低 Loss。

因为公式中有减号：

.. math::

   -\eta \times negative

最后会使 ``w`` 增大。


3.3 Gradient 会一直变吗？
-------------------------

会。

Gradient 不是训练开始算一次，然后一直使用。

每次训练都会发生：

.. code-block:: text

   Weight 改变
      ↓
   Prediction 改变
      ↓
   Loss 改变
      ↓
   Gradient 改变

而且 mini-batch 也会变化。

因此每一个 training step 的 gradient 通常都不同。


3.4 为什么 dz/dw1 = x1
----------------------

假设：

.. math::

   z=w_1x_1+w_2x_2+b

对 ``w1`` 求偏导：

.. math::

   \frac{\partial z}{\partial w_1}
   =
   x_1

因为求 ``w1`` 的偏导时：

* ``x1`` 看作常数
* ``w2x2`` 与 ``w1`` 无关
* ``b`` 与 ``w1`` 无关

所以：

.. math::

   \frac{\partial(w_1x_1)}{\partial w_1}=x_1


3.5 Chain Rule
--------------

真正训练时，我们更关心：

.. math::

   \frac{\partial C}{\partial w_1}

如果：

.. math::

   w_1 \rightarrow z \rightarrow C

那么：

.. math::

   \frac{\partial C}{\partial w_1}
   =
   \frac{\partial C}{\partial z}
   \frac{\partial z}{\partial w_1}

代入：

.. math::

   \frac{\partial z}{\partial w_1}=x_1

得到：

.. math::

   \frac{\partial C}{\partial w_1}
   =
   \frac{\partial C}{\partial z}x_1

更完整的 neural network：

.. code-block:: text

   w
   ↓
   z
   ↓
   activation
   ↓
   prediction
   ↓
   loss

因此：

.. math::

   \frac{\partial C}{\partial w}
   =
   \frac{\partial C}{\partial \hat{y}}
   \frac{\partial \hat{y}}{\partial a}
   \frac{\partial a}{\partial z}
   \frac{\partial z}{\partial w}

这就是 ``backpropagation`` 的核心数学基础。


3.6 Optimizer 是什么
--------------------

``Optimizer`` 的作用是：

**根据 gradient 决定怎样更新 weights。**

最简单的 Gradient Descent：

.. math::

   w_{\mathrm{new}}
   =
   w_{\mathrm{old}}
   -
   \eta
   \frac{\partial C}{\partial w}

其中：

``η``
   learning rate

``∂C/∂w``
   gradient

注意：

Optimizer 本身通常不是负责重新计算 loss。

正确流程是：

.. code-block:: text

   Forward Pass
       ↓
   Prediction
       ↓
   Loss
       ↓
   Backpropagation
       ↓
   Gradient
       ↓
   Optimizer
       ↓
   Update Weights

一句话记忆：

``Loss``
   告诉我们错得有多严重。

``Gradient``
   告诉我们往哪个方向改变参数。

``Optimizer``
   决定具体怎么改参数。


4. Keras 基本流程
================

4.1 Keras Workflow
------------------

典型流程：

#. 准备 input 和 target。
#. 定义 model。
#. 选择 loss function、optimizer、metrics。
#. ``compile()``
#. ``fit()``
#. ``evaluate()``
#. ``predict()``


4.2 Sequential API
------------------

适合简单的线性 layer 堆叠：

.. code-block:: r

   model <- keras_model_sequential() %>%
     layer_dense(
       units = 32,
       input_shape = c(784)
     ) %>%
     layer_dense(
       units = 10,
       activation = "softmax"
     )


4.3 Functional API
------------------

适合更复杂的网络结构。

.. code-block:: r

   input_tensor <- layer_input(shape = c(784))

   output_tensor <- input_tensor %>%
     layer_dense(units = 32, activation = "relu") %>%
     layer_dense(units = 10, activation = "softmax")

   model <- keras_model(
     inputs = input_tensor,
     outputs = output_tensor
   )

考试理解：

``Sequential API``
   简单、线性堆叠。

``Functional API``
   适合任意 DAG 结构和复杂模型。


4.4 compile()
-------------

作用：

**配置模型的学习方式。**

.. code-block:: r

   model %>% compile(
     optimizer = "rmsprop",
     loss = "binary_crossentropy",
     metrics = c("accuracy")
   )

三个核心内容：

``optimizer``
   如何更新参数。

``loss``
   优化什么。

``metrics``
   训练时监控什么。


4.5 fit()
---------

用于训练：

.. code-block:: r

   history <- model %>% fit(
     x_train,
     y_train,
     epochs = 20,
     batch_size = 512
   )

``epoch``
   完整遍历一次 training dataset。

``batch_size``
   每次更新参数前使用多少个样本。

``history``
   保存每个 epoch 的训练指标和 validation 指标。


4.6 evaluate()
--------------

用于评估：

.. code-block:: r

   results <- model %>% evaluate(
     x_test,
     y_test
   )

不会训练模型，只做 performance evaluation。


4.7 predict()
-------------

用于生成 prediction：

.. code-block:: r

   predictions <- model %>% predict(x_test)

输出含义取决于最后一层：

* Sigmoid → 一个概率分数。
* Softmax → 一组 class probabilities。
* Linear → 连续数值。


5. Binary Classification
========================

5.1 定义
--------

Binary classification：

一个输入只能属于两个类别之一。

例如：

.. code-block:: text

   Positive
   Negative


5.2 IMDB Dataset
----------------

本章使用 IMDB movie reviews：

* 50,000 reviews
* 25,000 training
* 25,000 test
* 两类基本平衡

保留最常见的 10,000 个词：

.. code-block:: r

   dataset_imdb(num_words = 10000)


5.3 为什么 Training 和 Test 必须分开
------------------------------------

不能用 training data 同时作为最终 test data。

因为模型可能只是：

**memorize training data**

而不是真正学会可以 generalize 的规律。

所以：

``training set``
   用来训练。

``validation set``
   用来调模型。

``test set``
   用来做最终评价。


5.4 Word Index
--------------

每篇 review 会被转换成整数序列：

.. code-block:: text

   [1, 14, 22, 16, 43, ...]

每个数字代表 vocabulary 中的一个词。


5.5 Vectorization
-----------------

Dense layer 不能直接接受这种 variable-length list。

本章将文本转成固定长度 binary vector。

如果 vocabulary 有 10,000 个词：

.. code-block:: text

   [0, 1, 0, 0, 1, ...]

表示对应的词是否出现。

这种方法：

* 保留“词是否出现”
* 不保留完整词序


5.6 Binary Classification 的模型结构
------------------------------------

.. code-block:: r

   model <- keras_model_sequential() %>%
     layer_dense(
       units = 16,
       activation = "relu",
       input_shape = c(10000)
     ) %>%
     layer_dense(
       units = 16,
       activation = "relu"
     ) %>%
     layer_dense(
       units = 1,
       activation = "sigmoid"
     )

结构：

.. code-block:: text

   10000 inputs
      ↓
   Dense(16)
   ReLU
      ↓
   Dense(16)
   ReLU
      ↓
   Dense(1)
   Sigmoid
      ↓
   probability


5.7 Hidden Unit
---------------

``hidden unit`` 可以理解为 internal representation 的一个维度。

如果：

.. code-block:: text

   units = 16

意味着这一层输出的是 16-dimensional representation。

更多 units：

* 可以学习更复杂的模式。
* 参数更多。
* 计算成本更高。
* 更容易 overfit。


6. Activation Functions
=======================

6.1 为什么需要 Activation Function
----------------------------------

如果没有 nonlinear activation：

.. math::

   y=W_2(W_1x+b_1)+b_2

最终仍然可以化成：

.. math::

   y=W'x+b'

也就是说：

**多层 linear layer 叠在一起，最终仍然只是 linear transformation。**

所以必须加入 non-linearity。


6.2 ReLU
--------

.. math::

   \mathrm{ReLU}(z)=\max(0,z)

特点：

* ``z < 0`` → 0
* ``z > 0`` → 保持原值

常用于 hidden layers。


6.3 Sigmoid
-----------

.. math::

   \sigma(z)=\frac{1}{1+e^{-z}}

输出范围：

.. math::

   0<\sigma(z)<1

所以适合 binary classification 的最终输出。


6.4 Softmax
-----------

Softmax 用于多类别互斥分类。

.. math::

   p_i=
   \frac{e^{z_i}}
   {\sum_j e^{z_j}}

满足：

.. math::

   \sum_i p_i=1

所以可以把输出看作各类别的 probability distribution。


7. Binary Crossentropy
======================

公式：

.. math::

   L=
   -[
   y\log(\hat{y})
   +(1-y)\log(1-\hat{y})
   ]

适用于：

* Binary classification
* Sigmoid output

常见配置：

.. code-block:: r

   model %>% compile(
     optimizer = "rmsprop",
     loss = "binary_crossentropy",
     metrics = c("accuracy")
   )


8. Validation 和 Overfitting
============================

8.1 Validation Set
------------------

Validation set 用来：

* 观察模型是否开始 overfit。
* 比较不同 architecture。
* 选择 epoch。
* 调 hyperparameters。


8.2 Overfitting
---------------

典型情况：

.. code-block:: text

   training loss      一直下降
   training accuracy  一直上升

   validation loss    先下降，后上升
   validation accuracy 先上升，后停止甚至下降

这说明：

模型越来越擅长 training data，
但对 unseen data 的表现变差。

定义：

**Overfitting 是模型对 training data 学得过于具体，导致 generalization performance 下降。**


9. Multiclass Classification
============================

9.1 定义
--------

一个输入属于多个可能类别中的一个。

本章 Reuters 数据：

.. code-block:: text

   46 classes

每条 newswire 只属于一个 class。

因此是：

**single-label multiclass classification**


9.2 Single-Label vs Multi-Label
-------------------------------

``single-label multiclass``
   一个样本只能属于一个类别。

``multi-label``
   一个样本可以同时属于多个类别。


9.3 One-Hot Encoding
--------------------

例如四个 class，正确 class 为第 2 个：

.. code-block:: text

   [0, 1, 0, 0]

对于 46 类：

.. code-block:: text

   length = 46

且只有正确 class 的位置为 1。


9.4 Multiclass Model
--------------------

.. code-block:: r

   model <- keras_model_sequential() %>%
     layer_dense(
       units = 64,
       activation = "relu",
       input_shape = c(10000)
     ) %>%
     layer_dense(
       units = 64,
       activation = "relu"
     ) %>%
     layer_dense(
       units = 46,
       activation = "softmax"
     )

因为有 46 个 class：

.. code-block:: text

   output units = 46


9.5 Categorical Crossentropy
----------------------------

.. math::

   L=
   -\sum_i y_i\log(\hat{y}_i)

适用于：

* one-hot encoded labels
* softmax output


9.6 Sparse Categorical Crossentropy
-----------------------------------

如果 label 不做 one-hot，而是保留 integer：

.. code-block:: text

   0, 1, 2, ..., 45

可以使用：

.. code-block:: r

   loss = "sparse_categorical_crossentropy"

两者核心数学目标相同，区别主要在 label representation。


10. Information Bottleneck
==========================

如果中间层过小：

.. code-block:: text

   10000
     ↓
    64
     ↓
     4
     ↓
    46

可能会丢失太多信息。

这称为：

``information bottleneck``

核心理解：

**后面的 layer 只能使用前一层输出的信息。**

如果前一层已经把有用信息压缩丢失，后面的 layer 无法凭空恢复。

注意：

这不代表“所有 hidden layer 都必须大于 output layer”。

正确理解是：

**过度压缩 representation 可能影响模型学习能力。**


11. Regression
==============

11.1 定义
---------

Classification：

预测离散类别。

Regression：

预测连续数值。

例如：

* 房价
* 温度
* 销售额
* 时间


11.2 Boston Housing Example
---------------------------

本章使用：

* 404 training samples
* 102 test samples
* 13 features
* continuous target

数据量比较小，因此：

* 更容易 overfit
* validation 更不稳定


11.3 Feature Normalization
--------------------------

如果不同 feature 数值范围差异很大，会让训练更困难。

标准化：

.. math::

   x'
   =
   \frac{x-\mu}{\sigma}

使 feature 大致：

* mean ≈ 0
* standard deviation ≈ 1


11.4 Data Leakage
-----------------

mean 和 standard deviation 必须：

**只用 training data 计算。**

然后：

用 training mean/std 去转换：

* training
* validation
* test

不能用 test data 自己计算 normalization 参数。

否则会产生：

``data leakage``


11.5 Regression Model
---------------------

.. code-block:: r

   model <- keras_model_sequential() %>%
     layer_dense(
       units = 64,
       activation = "relu",
       input_shape = c(13)
     ) %>%
     layer_dense(
       units = 64,
       activation = "relu"
     ) %>%
     layer_dense(units = 1)

最后：

.. code-block:: text

   1 output
   no activation

因为要预测 unrestricted continuous value。


11.6 为什么 Regression 最后一层通常没有 Activation
---------------------------------------------------

如果使用 sigmoid：

.. math::

   0<\hat{y}<1

这样输出会被限制在 0 到 1。

但是房价可能是：

.. code-block:: text

   15
   20
   35
   50

所以通常使用 linear output。


12. MSE 和 MAE
==============

12.1 MSE
--------

Mean Squared Error：

.. math::

   \mathrm{MSE}
   =
   \frac{1}{N}
   \sum_i
   (y_i-\hat{y}_i)^2

特点：

* 对大误差惩罚更强。
* 常作为 regression loss。


12.2 MAE
--------

Mean Absolute Error：

.. math::

   \mathrm{MAE}
   =
   \frac{1}{N}
   \sum_i
   |y_i-\hat{y}_i|

优点：

单位容易解释。

比如：

.. code-block:: text

   MAE = 2.5

如果 target 单位是“千美元”，就意味着：

平均大约误差 $2,500。


12.3 Loss 和 Metric 的区别
--------------------------

例如：

.. code-block:: r

   loss = "mse"
   metrics = c("mae")

``MSE``
   optimizer 真正优化的目标。

``MAE``
   用来帮助人理解 performance。

所以：

**loss 和 metric 不一定是同一个函数。**


13. K-Fold Cross-Validation
===========================

13.1 为什么需要 K-Fold
----------------------

小数据集如果只分一次 validation set：

结果很可能受到这次随机划分影响。

这叫：

``high validation variance``

因此可以使用：

``K-fold cross-validation``


13.2 原理
---------

假设：

.. code-block:: text

   K = 4

则：

.. code-block:: text

   Fold 1: VALID | TRAIN | TRAIN | TRAIN
   Fold 2: TRAIN | VALID | TRAIN | TRAIN
   Fold 3: TRAIN | TRAIN | VALID | TRAIN
   Fold 4: TRAIN | TRAIN | TRAIN | VALID

最后计算：

.. math::

   \bar{s}
   =
   \frac{1}{K}
   \sum_{i=1}^{K}s_i

这样比单次 validation split 更稳定。


14. Training、Validation、Test
=============================

``Training set``
   用来学习 weights。

``Validation set``
   用来选择 model 和 hyperparameters。

``Test set``
   最后用于评价 generalization。

正确流程：

.. code-block:: text

   Training
      ↓
   训练模型

   Validation
      ↓
   选 architecture / epoch / hyperparameters

   Test
      ↓
   最终评估

考试非常常问：

**不要用 test set 调参数。**


15. Parameters 和 Hyperparameters
=================================

``Parameters``
   模型训练过程中自动学习。

例如：

* weights
* biases

``Hyperparameters``
   训练前或训练过程中由研究者决定。

例如：

* learning rate
* number of layers
* hidden units
* batch size
* epochs
* optimizer


16. Optimization 和 Generalization
==================================

``Optimization``
   关注 training objective 是否被有效降低。

``Generalization``
   关注 unseen data 上是否表现良好。

重要：

**training loss 很低，不代表 model 一定好。**

因为模型可能已经 overfit。


17. 三类任务对比
================

.. list-table::
   :header-rows: 1

   * - Task
     - Output layer
     - Activation
     - Loss
     - Metric
   * - Binary classification
     - 1 unit
     - Sigmoid
     - Binary crossentropy
     - Accuracy
   * - Single-label multiclass
     - N units
     - Softmax
     - Categorical crossentropy
     - Accuracy
   * - Scalar regression
     - 1 unit
     - Linear
     - MSE
     - MAE


18. 高频考试定义
===============

Layer
-----

接收 tensor 并输出 tensor 的数据处理模块，通常包含 trainable parameters。

Model
-----

由多个 layer 连接形成，用来把 input 映射到 prediction。

Weight
------

训练过程中学习的参数，决定输入信息对下一层 representation 的影响。

Bias
----

加在线性变换结果上的 trainable additive parameter。

Activation Function
-------------------

作用于 pre-activation 的函数，通常用于引入 nonlinearity。

Loss Function
-------------

衡量 prediction 和 target 差异，并作为训练优化目标的 scalar function。

Gradient
--------

Loss 对参数的偏导数，表示在当前位置参数变化对 Loss 的局部影响。

Optimizer
---------

使用 gradient 更新模型参数的算法。

Epoch
-----

完整遍历一次 training dataset。

Batch
-----

一次参数更新中使用的一小部分 training samples。

Overfitting
-----------

训练集表现继续变好，但 unseen data 上表现开始变差。

Generalization
--------------

模型在从未见过的数据上的表现能力。

One-Hot Encoding
----------------

类别编码方式，一个位置为 1，其余为 0。

K-Fold Cross-Validation
-----------------------

将数据分成 K 份，重复训练和验证，并平均多个 validation score。


19. 高频考试问答
===============

Question 1
----------

**Loss function 和 optimizer 有什么区别？**

Answer
~~~~~~

Loss function 负责定义“模型错得有多严重”，并产生需要最小化的 scalar objective。
Backpropagation 根据 loss 计算 gradient，optimizer 再根据 gradient 更新 weights。


Question 2
----------

**为什么 activation function 必须是 nonlinear？**

Answer
~~~~~~

如果所有 layer 都是 linear transformation，那么无论叠多少层，最终仍然等价于一个
linear transformation。加入 nonlinear activation 后，模型才能学习复杂的 nonlinear relationship。


Question 3
----------

**为什么 binary classification 常用 sigmoid？**

Answer
~~~~~~

因为 sigmoid 将任意实数映射到 0 到 1 之间，可以作为 binary class 1 的概率分数。


Question 4
----------

**为什么 multiclass classification 用 softmax？**

Answer
~~~~~~

Softmax 将多个 logits 转换成一组概率，并保证总和为 1，因此适合 mutually exclusive classes。


Question 5
----------

**categorical crossentropy 和 sparse categorical crossentropy 的区别？**

Answer
~~~~~~

核心数学目标相同，但 label representation 不同：

* categorical crossentropy → one-hot labels
* sparse categorical crossentropy → integer labels


Question 6
----------

**为什么 regression 要做 normalization？**

Answer
~~~~~~

不同 feature 的 scale 差距太大会增加 gradient-based optimization 的难度。
标准化可以让 feature 尺度更加一致。


Question 7
----------

**为什么 normalization 不能使用 test data 的 mean/std？**

Answer
~~~~~~

因为这样会让 test data 的信息提前进入 model development，造成 data leakage。


Question 8
----------

**为什么小数据集适合 K-fold？**

Answer
~~~~~~

单次 validation split 很容易受到样本划分影响，K-fold 可以通过多次验证并取平均，
得到更稳定的 generalization estimate。


Question 9
----------

**Optimization 和 generalization 有什么区别？**

Answer
~~~~~~

Optimization 关注训练目标是否下降；generalization 关注模型在 unseen data 上的表现。
模型可能 optimization 很好，但 generalization 很差。


Question 10
-----------

**什么是 information bottleneck？**

Answer
~~~~~~

如果 intermediate layer 维度过小，可能丢失后续分类所需要的信息，导致模型能力下降。


20. 必背公式
============

Dense Layer
-----------

.. math::

   z=Wx+b

ReLU
----

.. math::

   \mathrm{ReLU}(z)=\max(0,z)

Sigmoid
-------

.. math::

   \sigma(z)=\frac{1}{1+e^{-z}}

Softmax
-------

.. math::

   p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}

Binary Crossentropy
-------------------

.. math::

   L=
   -[
   y\log(\hat{y})
   +(1-y)\log(1-\hat{y})
   ]

Categorical Crossentropy
------------------------

.. math::

   L=
   -\sum_i y_i\log(\hat{y}_i)

MSE
---

.. math::

   \mathrm{MSE}
   =
   \frac{1}{N}
   \sum_i
   (y_i-\hat{y}_i)^2

MAE
---

.. math::

   \mathrm{MAE}
   =
   \frac{1}{N}
   \sum_i
   |y_i-\hat{y}_i|

Gradient Descent
----------------

.. math::

   w_{\mathrm{new}}
   =
   w_{\mathrm{old}}
   -
   \eta
   \frac{\partial C}{\partial w}

Standardization
---------------

.. math::

   x'
   =
   \frac{x-\mu}{\sigma}


21. 常见易错点
=============

``误区 1``
   Optimizer 负责计算 Loss。

正确：
   Loss function 计算 loss，backpropagation 计算 gradient，optimizer 更新 weights。

``误区 2``
   Gradient 在整个训练过程中不变。

正确：
   Weight、prediction、loss 和 mini-batch 都会变，所以 gradient 也会重新计算。

``误区 3``
   Training loss 越低，模型一定越好。

正确：
   可能已经 overfit。

``误区 4``
   Regression 也可以直接用 accuracy。

正确：
   Regression 更关心 prediction 与 target 的数值距离，例如 MAE、MSE。

``误区 5``
   Test set 可以拿来选 epoch。

正确：
   应使用 validation set 调模型，test set 应留到最后。


22. 考前最终检查
===============

考试前你应该能独立回答：

#. Layer 和 Model 有什么区别？
#. Weight 和 Bias 是什么？
#. Loss function 的作用是什么？
#. ``∂C/∂w`` 代表什么？
#. Gradient 为什么会不断变化？
#. Optimizer 的作用是什么？
#. Backpropagation 和 Chain Rule 有什么关系？
#. 为什么要用 nonlinear activation？
#. ReLU、Sigmoid、Softmax 分别适合什么场景？
#. Binary classification 最后一层通常怎么设计？
#. Multiclass classification 最后一层通常怎么设计？
#. Regression 最后一层为什么通常不用 activation？
#. Binary crossentropy 什么时候用？
#. Categorical crossentropy 什么时候用？
#. MSE 和 MAE 有什么区别？
#. 什么是 overfitting？
#. Training、Validation、Test 有什么区别？
#. 什么是 data leakage？
#. 什么是 K-fold cross-validation？
#. 什么是 information bottleneck？
#. 什么是 model capacity？
#. Parameters 和 hyperparameters 有什么区别？
#. Optimization 和 generalization 有什么区别？


23. 十句话总复习
===============

#. Neural network 由多个 layer 构成，用来把 input 转换成 prediction。
#. Weights 是模型在训练中真正学习的 parameters。
#. Loss function 衡量 prediction 和 target 的差异。
#. Backpropagation 用 Chain Rule 计算每个参数对应的 gradient。
#. Optimizer 根据 gradient 更新 weights，使未来的 loss 尽可能降低。
#. Binary classification 通常使用 1 个 sigmoid output 和 binary crossentropy。
#. Single-label multiclass classification 通常使用 N 个 softmax outputs 和 categorical crossentropy。
#. Scalar regression 通常使用 1 个 linear output、MSE loss 和 MAE metric。
#. Validation set 用来调模型，Test set 用来做最终评价。
#. Machine learning 的目标不只是降低 training loss，而是获得好的 generalization。
"""

3.7 三大 Loss Function（考试高频）
===================================

这一章最重要的三种 ``loss function``：

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - 任务（Task）
     - 最后一层（Output）
     - Loss Function
   * - Binary classification
     - Sigmoid（1 unit）
     - Binary Crossentropy
   * - Single-label Multiclass classification
     - Softmax（N units）
     - Categorical Crossentropy
   * - Regression
     - Linear（1 unit）
     - Mean Squared Error (MSE)

记忆口诀：

.. code-block:: text

   Binary      → Sigmoid → Binary Crossentropy

   Multiclass  → Softmax → Categorical Crossentropy

   Regression  → Linear  → MSE


3.7.1 Binary Crossentropy
--------------------------

**Binary Crossentropy（二分类交叉熵）** 用于 ``Binary classification``。

模型最后通常使用：

.. code-block:: text

   Dense(1)
      ↓
   Sigmoid
      ↓
   Probability
      ↓
   Binary Crossentropy

Sigmoid 会输出：

.. math::

   \hat y \in (0,1)

表示属于 **Class 1** 的概率。

数学公式：

.. math::

   L=-
   \left[
   y\log(\hat y)
   +(1-y)\log(1-\hat y)
   \right]

其中：

``y``
   真实标签（0 或 1）。

``ŷ``
   模型预测的概率。

理解：

如果真实标签：

.. math::

   y=1

则：

.. math::

   L=-\log(\hat y)

因此：

.. list-table::
   :header-rows: 1

   * - Prediction
     - Loss
   * - 0.99
     - 很小
   * - 0.90
     - 小
   * - 0.50
     - 中等
   * - 0.10
     - 很大
   * - 0.01
     - 非常大

考试重点：

- Prediction 越接近正确答案，Loss 越小。
- **模型越自信地预测错，惩罚越大。**

一句话记忆：

   Binary Crossentropy 衡量预测概率与二分类真实标签之间的差距。


3.7.2 Categorical Crossentropy
-------------------------------

**Categorical Crossentropy（类别交叉熵）** 用于：

**Single-label Multiclass Classification**

即：

- 有多个类别；
- 每个样本只能属于一个类别。

典型结构：

.. code-block:: text

   Dense(N)
      ↓
   Softmax
      ↓
   Probability Distribution
      ↓
   Categorical Crossentropy

例如：

.. code-block:: text

   Sports
   Finance
   Technology

真实标签（One-Hot Encoding）：

.. code-block:: text

   [0,1,0]

模型预测：

.. code-block:: text

   [0.10,0.80,0.10]

公式：

.. math::

   L=
   -\sum_i
   y_i
   \log(\hat y_i)

由于 One-Hot 中只有正确类别为 1，因此实际上只有正确类别参与计算。

例如：

.. math::

   L=-\log(0.8)

考试重点：

- Softmax 输出所有类别的概率。
- 所有概率之和等于 1。
- Loss 主要取决于模型给**正确类别**分配了多少概率。

Master 补充：

如果标签不是 One-Hot，而是整数标签：

.. code-block:: text

   0
   1
   2
   ...

通常使用：

``sparse_categorical_crossentropy``

二者数学目标相同，只是标签表示方式不同。

一句话记忆：

   正确类别概率越高，Categorical Crossentropy 越小。


3.7.3 Mean Squared Error (MSE)
------------------------------

**Mean Squared Error（均方误差）** 是 Regression 最常见的 ``loss function``。

模型最后通常使用：

.. code-block:: text

   Dense(1)
      ↓
   Linear Output
      ↓
   MSE

数学公式：

.. math::

   MSE=
   \frac1N
   \sum_i
   (y_i-\hat y_i)^2

计算步骤：

#. 计算每个样本的 Error。
#. 对 Error 做平方。
#. 对所有平方误差取平均。

例子：

.. list-table::
   :header-rows: 1

   * - Target
     - Prediction
     - Error²
   * - 20
     - 18
     - 4
   * - 30
     - 31
     - 1
   * - 40
     - 37
     - 9

因此：

.. math::

   MSE=\frac{4+1+9}{3}=4.67

为什么要平方？

**原因一：避免正负误差互相抵消。**

例如：

.. math::

   +5,\ -5

直接平均会得到 0，但平方后变成：

.. math::

   25,\ 25

**原因二：更严重地惩罚大误差。**

例如：

.. math::

   2^2=4

但：

.. math::

   10^2=100

因此 MSE 对 **large errors** 更敏感。

考试重点：

- Regression 通常使用 Linear Output。
- MSE 是最常见的训练 Loss。
- MAE 常作为更容易解释的 Evaluation Metric。


3.7.4 三种 Loss 的对比（必背）
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 22 34

   * - Loss
     - Output
     - 适用任务
     - 核心特点
   * - Binary Crossentropy
     - Sigmoid
     - Binary Classification
     - 自信预测错误时惩罚更大
   * - Categorical Crossentropy
     - Softmax
     - Single-label Multiclass
     - 正确类别概率越高，Loss 越小
   * - MSE
     - Linear
     - Regression
     - 对大误差惩罚更强

考试快速判断：

- ``Sigmoid + Binary Crossentropy`` → Binary Classification
- ``Softmax + Categorical Crossentropy`` → Single-label Multiclass Classification
- ``Linear + MSE`` → Regression

第四章：机器学习基础（深度笔记）
==========================

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

.. note::
   **直观例子**：一个拥有 50 万个二进制参数的模型，完全可以为 MNIST
   训练集中 5 万张图片、每张分配 10 个二进制参数，从而"完美"记住每一张
   图片对应的类别。这样的模型在训练集上准确率可以逼近 100%，但对没见过
   的新样本毫无预测能力——这正是"容量"与"泛化"之间张力的最直白体现：
   参数多不等于学得好，只等于**记得多**。

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
``0.001 * weight_coefficient_value^2`` 的惩罚项。**注意**：该惩罚项只在
训练阶段被加入损失，因此同一模型的训练损失会明显高于测试损失，这是正常
现象而非 bug。

4.4 Dropout
---------------

由 Hinton 团队提出，是最有效、应用最广泛的正则化技术之一。训练时，
以概率 :math:`p`（dropout rate，通常 0.2–0.5）随机将该层部分输出置零；
Keras 等现代框架通常采用 **inverted dropout**：训练时对保留下来的激活进行缩放，测试时不丢弃单元，也无需额外缩放。

**直观例子**：假设某层在训练时对某个输入样本本应输出向量
``[0.2, 0.5, 1.3, 0.8, 1.1]``。应用 dropout 后，其中若干条目会被随机
置零，例如变为 ``[0, 0.5, 1.3, 0, 1.1]``。测试阶段则不丢弃任何单元，
而是将输出整体按 dropout rate 缩小（如乘以 0.5），以补偿训练时"活跃
单元更少"这一事实。

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
       **反例**：试图仅凭近期股价历史预测股票未来走势通常注定失败——
       并非模型不够强，而是价格历史本身几乎不携带可用于预测的信息，
       即假设 (b) 不成立。
       注意**非平稳问题（nonstationary）**——如随季节变化的推荐系统，
       应定期用近期数据重训练，或将"时间"显式作为输入特征。

   * - 2
     - 选择成功度量
     - 度量标准决定损失函数的选择方向。平衡分类问题常用 accuracy / ROC AUC；
       类别不平衡问题常用 precision / recall；排序/多标签问题常用 mAP。
       原书建议：日常浏览 `Kaggle <https://kaggle.com>`_ 上的数据科学竞赛，
       能快速建立起对不同问题域下"成功度量"多样性的直觉。

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

6. Master 学术扩展：从教材工作流到研究级实验设计
=====================================================

本节是在原章核心内容基础上的研究生层级扩展。原章解决的是“如何建立一个可靠的
机器学习工作流”，而 Master 阶段还需要进一步回答：**如何证明实验设计本身可靠，
模型改进真实存在，而且结果可以复现。**

6.1 更完整的学习范式
----------------------

经典四分法仍然重要，但现代机器学习实践中还应认识 **半监督学习
（Semi-supervised Learning）**。它利用少量有标签数据和大量无标签数据共同训练，
特别适用于人工标注昂贵的领域，例如医学影像、遥感和工业检测。

现代视角下，可以把常见范式理解为：

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - 范式
     - 监督信号
     - 典型任务/方法
   * - Supervised
     - 人工或外部提供的标签
     - 分类、回归、检测、分割
   * - Unsupervised
     - 无显式标签
     - 聚类、降维、密度估计
   * - Self-supervised
     - 从数据自身构造训练目标
     - masked prediction、next-token prediction、contrastive learning
   * - Semi-supervised
     - 少量标签 + 大量无标签数据
     - pseudo-labeling、consistency regularization
   * - Reinforcement Learning
     - 环境反馈的 reward
     - sequential decision making、control

.. important::
   **Self-supervised Learning 的核心不是“没有 target”，而是 target 不需要人工标注。**
   现代深度学习常先通过 self-supervised pretraining 学习 representation，再针对
   downstream task 进行 fine-tuning。

6.2 Evaluation Protocol：不只是随机切分
-----------------------------------------

Hold-out 和 K-fold 是基础，但研究级实验必须根据数据结构选择切分方式。

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - 方法
     - 适用场景
     - 关键目的
   * - Hold-out
     - 数据量较充足
     - 简单、计算成本低
   * - Stratified split / K-fold
     - 类别比例不均衡
     - 保持各 fold 类别分布相近
   * - Group split / Group K-fold
     - 同一 subject/patient/user 有多个样本
     - 防止同一实体跨 train/test
   * - Time-series split
     - 时间序列或未来预测
     - 保证训练数据早于验证/测试数据
   * - Nested cross-validation
     - 小数据且需要严格模型选择
     - 将 hyperparameter tuning 与最终性能估计分离

**核心原则：数据集“互不相交”不仅意味着 row 不重复，还要考虑产生这些 row 的实体。**

例如医学影像中，一个 patient 可能有多张 X-ray：

::

   Patient A
   ├── image_1
   ├── image_2
   └── image_3

如果 ``image_1`` 在 training set，而 ``image_2`` 在 test set，即使文件不同，
模型仍可能利用 patient-specific information，导致性能被高估。因此应优先考虑
**patient-level / group-level splitting**。

6.3 Data Leakage Taxonomy
---------------------------

Data leakage 是 Master 级实验设计中必须系统掌握的概念。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 类型
     - 例子
   * - Train-test contamination
     - 测试样本或其复制品进入训练过程
   * - Preprocessing leakage
     - 使用完整数据计算 normalization 的 mean/std
   * - Feature-selection leakage
     - 在划分数据前利用所有样本选择特征
   * - Group leakage
     - 同一 patient/user 的不同记录进入不同 split
   * - Temporal leakage
     - 使用未来数据预测过去或当前
   * - Benchmark contamination
     - 模型预训练或开发过程已经接触 benchmark test information

推荐把整个 preprocessing pipeline 看成模型训练的一部分：

::

   Training data
        │
        ├── fit imputer
        ├── fit scaler
        ├── fit feature selection / PCA
        └── fit model

   Validation / Test data
        │
        └── transform only using training-fitted parameters

.. warning::
   **任何从 validation/test data 学到的参数，都可能造成 leakage。**
   这不仅包括模型 weights，也包括 mean、standard deviation、PCA components、
   feature-selection thresholds 等。

6.4 Missing Data：从“填 0”到缺失机制
--------------------------------------

将 missing value 设为 0 是某些神经网络场景中的可行策略，但不是普遍最佳实践。
研究中应先理解缺失的来源，并选择合理处理方法。

常见 missingness mechanism：

- **MCAR (Missing Completely At Random)**：缺失与已观察和未观察变量都无系统关系；
- **MAR (Missing At Random)**：在给定已观察变量后，缺失机制可解释；
- **MNAR (Missing Not At Random)**：缺失与未观察值本身仍有关。

常见处理策略包括 mean/median imputation、mode imputation、KNN/model-based
imputation、missing-indicator，以及深度学习中的 masking。

.. important::
   Imputation 也必须遵循训练/测试隔离原则。例如 median 应只从 training set
   估计，再应用于 validation/test set。

6.5 Feature Engineering 与 Representation Learning
----------------------------------------------------

传统 Feature Engineering 依赖人工先验；Deep Learning 更强调从数据中自动学习
representation。二者并非互斥。

::

   Raw Data
      │
      ├── Human-designed transformation
      │        └── Feature Engineering
      │
      └── Learned transformation
               └── Representation Learning
                         │
                         └── Pretraining → Fine-tuning

在小数据、tabular data、强领域先验或高可解释性要求下，Feature Engineering
仍然非常重要；在图像、文本、语音等高维非结构化数据中，Representation Learning
通常更具优势。

6.6 Bias–Variance Trade-off
----------------------------

Underfitting / Overfitting 可以进一步用 Bias–Variance 的视角理解：

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - 情况
     - Bias
     - Variance
     - 典型表现
   * - 模型过于简单
     - 高
     - 较低
     - Underfitting
   * - 合理复杂度
     - 适中
     - 适中
     - 较好 generalization
   * - 模型过于复杂
     - 较低
     - 高
     - Overfitting

这里的重点不是机械追求“更复杂”或“更简单”，而是利用 validation evidence
寻找能够最好泛化的模型复杂度。

6.7 现代 Regularization 工具箱
--------------------------------

原章中的更多数据、降低容量、L1/L2 和 Dropout 仍然是核心方法。现代实践还经常使用：

- **Data augmentation**：通过合理变换增加训练数据多样性；
- **Early stopping**：validation performance 不再改善时停止训练；
- **Label smoothing**：降低分类模型对 hard target 的过度自信；
- **Weight decay / AdamW**：现代 optimizer 中常见的参数约束方式。

.. note::
   Batch Normalization 的主要目的不是 regularization，而是改善网络训练行为；
   它在某些设置下可能产生 regularizing effect，但不应简单等同于 Dropout/L2。

.. note::
   对普通 SGD，L2 penalty 与 weight decay 有紧密关系；对 Adam 等 adaptive
   optimizer，两者并不应简单视为完全等价。AdamW 的关键思想正是将 weight decay
   与梯度更新解耦。

6.8 Loss Function 与 Evaluation Metric
---------------------------------------

**Loss 是训练信号；Metric 是评价标准。两者相关，但不是同一个概念。**

::

   Prediction + Target
          │
          ├── Loss
          │     └── differentiable → backpropagation → optimizer
          │
          └── Metric
                └── evaluate whether the model meets the task objective

常见指标：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 任务
     - 常见 metrics
   * - Binary / Multiclass classification
     - Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC
   * - Imbalanced classification
     - Precision, Recall, F1, PR-AUC, balanced metrics
   * - Multilabel classification
     - Micro-F1, Macro-F1, mAP, per-class AUROC
   * - Regression
     - MAE, MSE, RMSE, :math:`R^2`

.. important::
   Accuracy 在严重 class imbalance 下可能具有误导性。例如 95% 样本属于 negative
   class 时，一个永远预测 negative 的模型也可以得到 95% accuracy。

6.9 Baseline：研究不是只看一个最终分数
----------------------------------------

一个模型“表现不错”并不足以证明方法有效。研究中需要建立有意义的 baseline。

常见 baseline 层次：

1. Random baseline；
2. Majority-class / simple heuristic baseline；
3. Classical ML baseline；
4. Standard deep-learning baseline；
5. Published baseline / previous method。

研究问题通常不是：

   “我的模型 AUROC 是多少？”

而是：

   “在相同数据、相同 evaluation protocol 下，我的方法是否稳定优于合理 baseline？”

6.10 Reproducibility 与 Experimental Rigour
--------------------------------------------

可复现性（reproducibility）是学术实验的重要组成部分。实验记录至少应包含：

- random seed；
- dataset version 与 inclusion/exclusion criteria；
- train/validation/test split 方法；
- preprocessing pipeline；
- model architecture；
- optimizer、learning rate、batch size、epochs；
- hyperparameter search strategy；
- software/library versions；
- 必要时记录 hardware；
- 多次实验的 mean ± standard deviation，而不是只挑最好的一次。

.. warning::
   **Single run ≠ strong experimental evidence.**
   深度学习训练受到 initialization、batch ordering、augmentation 等随机因素影响。
   两个模型之间非常小的性能差异不一定代表真实改进。

6.11 Ablation Study
--------------------

如果提出一个包含多个新组件的方法，仅比较“完整新模型 vs baseline”无法证明每个
组件是否真正有效。Ablation Study 通过逐个移除或加入组件来回答这个问题。

例如：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Model
     - Validation AUROC
   * - CNN baseline
     - 0.840
   * - + Attention
     - 0.860
   * - + Feature Fusion
     - 0.880
   * - + Proposed Loss
     - 0.890

这类实验可以回答：

- improvement 来自哪个 component？
- 多个 component 是否存在互补作用？
- 模型复杂度增加是否值得？

6.12 Error Analysis、Robustness 与 Uncertainty
-----------------------------------------------

最终 aggregate metric 不能解释模型为什么失败。研究阶段还应检查：

- 哪些 classes 最容易预测错误？
- false positive 与 false negative 的模式是什么？
- 是否存在特定 subgroup 性能下降？
- 模型对 distribution shift 是否敏感？
- 高 confidence 的错误预测有哪些？
- probability 是否经过合理 calibration？

这些分析把“训练出一个模型”提升为“理解模型行为”。

7. Master Research Workflow
============================

在原章 7-step universal workflow 的基础上，可以扩展为更适合 dissertation /
research project 的工作流：

.. list-table::
   :header-rows: 1
   :widths: 8 32 60

   * - Step
     - 阶段
     - 核心问题
   * - 1
     - Define research question
     - 明确 prediction task、研究假设和预期贡献
   * - 2
     - Dataset & target
     - 数据是否能够回答研究问题？标签质量如何？
   * - 3
     - Define baselines
     - 建立 random / classical / deep-learning / published baseline
   * - 4
     - Evaluation design
     - 设计 leakage-free split、metrics 和最终 held-out test
   * - 5
     - Preprocessing pipeline
     - 所有需要 fit 的步骤仅使用 training data
   * - 6
     - Build baseline model
     - 验证任务具有可学习性
   * - 7
     - Scale until overfitting
     - 确认模型容量边界
   * - 8
     - Regularize & tune
     - 在 validation set 上进行模型选择
   * - 9
     - Ablation study
     - 验证 proposed components 的独立贡献
   * - 10
     - Error & robustness analysis
     - 分析失败模式、subgroup、distribution shift 和 uncertainty
   * - 11
     - Final evaluation
     - 锁定模型后仅在 held-out test set 上进行最终评价
   * - 12
     - Report & reproduce
     - 报告方法、结果、随机性、limitations 和 reproducibility 信息

8. 本章核心知识图
==================

::

   Machine Learning Problem
            │
            ▼
   Define X, Y and task
            │
            ▼
   Choose success metric
            │
            ▼
   Design leakage-free evaluation
            │
            ▼
   Prepare data
   ├── Vectorization
   ├── Normalization
   ├── Missing-data handling
   └── Feature / Representation Learning
            │
            ▼
   Establish baseline
            │
            ▼
   Train and optimize
            │
            ▼
   Underfitting ─── Appropriate Capacity ─── Overfitting
            │                 │
            │                 ▼
            │            Generalization
            │                 │
            └──── Regularization / Tuning
                              │
                              ▼
                    Final held-out evaluation
                              │
                              ▼
                 Error / Robustness Analysis
                              │
                              ▼
                  Reproducible Research Report

9. 本章要点小结
================

1. Machine Learning 的最终目标不是最小化 training loss，而是获得可靠的
   **generalization**。
2. Training / validation / test 的职责必须严格分离；任何间接使用 validation/test
   information 的行为都可能造成 **data leakage**。
3. Evaluation protocol 必须匹配数据结构：分类比例、group/patient、时间顺序都可能
   决定正确的 split 方法。
4. Preprocessing 也是学习过程的一部分；scaling、imputation、PCA、feature selection
   等参数只能从 training data 估计。
5. Feature Engineering 与 Representation Learning 是两种不同但互补的表示构建方式。
6. Underfitting / Overfitting 可以通过 model capacity、regularization 以及
   Bias–Variance Trade-off 统一理解。
7. Loss Function 用于 optimization；Evaluation Metric 用于判断模型是否满足任务目标。
8. 一个研究结果必须与合理 baseline 比较，而不是孤立报告 accuracy/AUROC。
9. Master 层级的实验还应关注 reproducibility、ablation study、error analysis、
   robustness 和 uncertainty。
10. 最终 test set 应在模型和 hyperparameters 锁定后使用，尽量保持真正的
    **held-out evaluation**。

.. seealso::
   **建议继续深入的研究生主题**

   - Bias–Variance decomposition 与 statistical learning theory；
   - Nested cross-validation 与 hyperparameter optimization；
   - Calibration、Brier score 与 Expected Calibration Error (ECE)；
   - Bootstrap confidence intervals 与模型性能比较；
   - Distribution shift、domain adaptation 与 out-of-distribution detection；
   - Explainable AI (XAI) 与模型解释可靠性；
   - Dataset shift、benchmark contamination 与 foundation-model evaluation。

