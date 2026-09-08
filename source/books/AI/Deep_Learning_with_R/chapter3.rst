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
