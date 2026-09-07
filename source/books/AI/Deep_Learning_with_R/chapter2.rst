Chapter 2. Before we begin: the mathematical building blocks of neural networks
===============================================================================

Start with MNIST dataset in Keras. (Of course!)
------------------------------------------------

.. code-block:: R

   library(keras3)

   # Loading the MNIST dataset from Keras3
   mnist <- dataset_mnist()

   str(mnist)

   train_images <- mnist$train$x
   # int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
   train_labels <- mnist$train$y
   # int [1:60000(1d)] 5 0 4 1 9 2 1 3 1 4 ...

   test_images <- mnist$test$x
   test_labels <- mnist$test$y

   str(train_labels)

Getting ready for the dataset: the training and testing images and labels.

Check out one of the image:

.. code-block:: R

   image(
     1:28,
     1:28,
     t(apply(train_images[1, , ], 2, rev)),
     col = gray.colors(256)
   )

.. image:: c2/1.png

The network architecture

.. code-block:: R

   network <- keras_model_sequential() %>%
     layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
     layer_dense(units = 10, activation = "softmax")

.. note::

   - A loss function — How the network will be able to measure its performance on the training data, and thus how it will be able to steer itself in the right direction.
   - An optimizer — The mechanism through which the network will update itself based on the data it sees and its loss function.
   - Metrics to monitor during training and testing — Here, we’ll only care about accuracy (the fraction of the images that were correctly classified).

The compilation step

.. code-block:: R

   network %>% compile(
     optimizer = "rmsprop", # gradient-based optimizer
     loss = "categorical_crossentropy",
     metrics = c("accuracy")
   )

- Optimizer（优化器）负责： 根据模型犯的错误，调整 weights。
- Loss function（损失函数）负责衡量：模型的预测到底有多错?

Preparing the image data.

.. code-block:: R

   train_images <- array_reshape(train_images, c(60000, 28 * 28)) # reshape a Matrix into an array.
   train_images <- train_images / 255 # Normalization

   test_images <- array_reshape(test_images, c(10000, 28 * 28))
   test_images <- test_images / 255

Preparing the labels.

.. code-block:: R

   train_labels <- to_categorical(train_labels)
   test_labels <- to_categorical(test_labels)

.. important:: Why we need this step?

   Our output layer looks like : [0.01, 0.02, 0.01, 0.03, 0.02, 0.85, 0.01, 0.02, 0.02, 0.03] -> prediction is 5. (0.85). VS Actual: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], then optimizer will change weights base on this loss. Moreover, we choosed "categorical_crossentropy" as loss function from beginning.

RUN

.. code-block:: R

   network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)
   /*
   Epoch 1/5
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9222 - loss: 0.2688
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9222 - loss: 0.2688
   Epoch 2/5
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9673 - loss: 0.1102
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9673 - loss: 0.1102
   Epoch 3/5
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9785 - loss: 0.0715
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9785 - loss: 0.0715
   Epoch 4/5
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9844 - loss: 0.0517
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9844 - loss: 0.0517
   Epoch 5/5
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9884 - loss: 0.0390
   469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9884 - loss: 0.0390 */

   metrics <- network %>% evaluate(test_images, test_labels)
   metrics
   /*
   $accuracy
   [1] 0.9811

   $loss
   [1] 0.06619187
   */

Data representations for neural networks
----------------------------------------

- What is a tensor?
Tensors are a generalization of vectors and matrices to an arbitrary number of dimensions (note that in the context of tensors, a *dimension* is often called an *axis*). In R, vectors are used to create and manipulate 1D tensors, and matrices are used for 2D tensors. For higher-level dimensions, *array* objects (which support any number of dimensions) are used.

.. note:: Real-world examples of data tensors

   - Vector data—2D tensors of shape (samples, features).
   - Timeseries data or sequence data—3D tensors of shape (samples, timesteps, features).
   - Images—4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width).
   - Video—5D tensors of shape (samples, frames, height, width, channels) or (samples, frames, channels, height, width).

Vector data
~~~~~~~~~~~

This is the most common case. In such a dataset, each single data point can be encoded as a vector, and thus a batch of data will be encoded as a 2D tensor (that is, an array of vectors), where the first axis is the samples axis and the second axis is the features axis.

Timeseries data or sequence data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever time matters in your data (or the notion of sequence order), it makes sense to store it in a 3D tensor with an explicit time axis. Each sample can be encoded as a sequence of vectors (a 2D tensor), and thus a batch of data will be encoded as a 3D tensor.

.. image:: c2/2.png

.. important::

   The time axis is always the second axis, by convention.

- 1. A dataset of stock prices. Every minute, we store the current price of the stock, the highest price in the past minute, and the lowest price in the past minute. Thus, every minute is encoded as a 3D vector, an entire day of trading is encoded as a 2D tensor of shape (390, 3) (there are 390 minutes in a trading day), and 250 days’ worth of data can be stored in a 3D tensor of shape (250, 390, 3). Here, each sample would be one day’s worth of data.

- 2. A dataset of tweets, where we encode each tweet as a sequence of 140 characters out of an alphabet of 128 unique characters. In this setting, each character can be encoded as a binary vector of size 128 (an all-zeros vector except for a 1 entry at the index corresponding to the character). Then each tweet can be encoded as a 2D tensor of shape (140, 128), and a dataset of 1 million tweets can be stored in a tensor of shape (1000000, 140, 128).

Image data
~~~~~~~~~~

Images typically have three dimensions: height, width, and color depth. Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in 2D tensors, by convention image tensors are always 3D, with a one-dimensional color channel for grayscale images. A batch of 128 grayscale images of size 256 × 256 could thus be stored in a tensor of shape (128, 256, 256, 1), and a batch of 128 color images could be stored in a tensor of shape (128, 256, 256, 3).

.. image:: c2/3.png

There are two conventions for shapes of images tensors:

- the channels-last convention (used by TensorFlow).(samples, height, width, color_depth)

- the channels-first convention (used by Theano).(samples, color_depth, height, width)

Video data
~~~~~~~~~~~
Video data is one of the few types of real-world data for which you’ll need 5D tensors. A video can be understood as a sequence of frames, each frame being a color image. Because each frame can be stored in a 3D tensor (height, width, color_depth), a sequence of frames can be stored in a 4D tensor (frames, height, width, color_depth), and thus a batch of different videos can be stored in a 5D tensor of shape (samples, frames, height, width, color_depth).

For instance, a 60-second, 144 × 256 YouTube video clip sampled at 4 frames per second would have 240 frames. A batch of four such video clips would be stored in a tensor of shape (4, 240, 144, 256, 3). That’s a total of 106,168,320 values! If the data type of the tensor is double, then each value is stored in 64 bits, so the tensor would represent 810 MB. Heavy! Videos you encounter in real life are much lighter, because they aren’t stored as double and they’re typically compressed by a large factor (such as in the MPEG format).

Element-wise operations
~~~~~~~~~~~~~~~~~~~~~~~~

relu

.. code-block:: R

   naive_relu <- function(x){
      for (i in nrow(x)) {
         for (j in ncol(x){
         x[i, j] <- max(x[i, j], 0)
         }
      }
      x
   }

In practice, when dealing with R arrays, these operations are available as well-optimized built-in R functions, which themselves delegate the heavy lifting to a BLAS implementation (Basic Linear Algebra Subprograms) if you have one installed (which you should). BLAS are low-level, highly parallel, efficient tensor-manipulation routines typically implemented in Fortran or C.

.. code-block:: R

   z <- x + y                1
   z <- pmax(z, 0)           2
   # 1. Element-wise addition
   # 2. Element-wise relu

Operations involving tensors of different dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R sweep() function enables you to perform operations between higher-dimension tensors and lower-dimension tensors. With sweep(), we could perform the matrix plus vector addition described earlier as follows:

.. code-block:: R

   sweep(x, 2, y, `+`)

.. note::

   sweep(x, 2, y, `+`) (原始 array x, 沿着 x 的第 2 个维度, 要加的值 y, 做什么运算：加法)

核心概念
--------

Dot product（点积，也称 tensor product）与 element-wise product（逐元素乘法）不同。

R 中：

.. code-block:: r

   # element-wise product
   z <- x * y

   # dot product
   z <- x %*% y

逐元素乘法是“对应位置相乘”；点积则会把多个输入元素组合起来。

Vector · Vector
---------------

两个一维向量做点积：

.. code-block:: r

   naive_vector_dot <- function(x, y) {
     z <- 0
     for (i in 1:length(x))
       z <- z + x[[i]] * y[[i]]
     z
   }

数学上：

.. math::

   x \cdot y = \sum_i x_i y_i

重要性质：

* ``x`` 和 ``y`` 都是 1D tensors（vectors）。
* 两个向量必须具有相同数量的元素。
* 输出是一个 scalar（标量）。

例如：

.. code-block:: text

   x = [1, 2, 3]
   y = [4, 5, 6]

   x · y
   = 1×4 + 2×5 + 3×6
   = 32

Matrix · Vector
---------------

矩阵与向量做点积时，矩阵的每一行分别与向量做一次 vector dot product。

.. code-block:: r

   naive_matrix_vector_dot <- function(x, y) {
     z <- rep(0, nrow(x))
     for (i in 1:nrow(x))
       for (j in 1:ncol(x))
         z[[i]] <- z[[i]] + x[[i, j]] * y[[j]]
     z
   }

这里：

* ``i`` 遍历矩阵的行。
* ``j`` 遍历矩阵的列。
* ``x[[i, j]] * y[[j]]`` 计算对应元素乘积。
* ``z[[i]]`` 保存第 ``i`` 行与 ``y`` 的点积。
* 输出是 vector。

也可以复用 vector dot：

.. code-block:: r

   naive_matrix_vector_dot <- function(x, y) {
     z <- rep(0, nrow(x))
     for (i in 1:nrow(x))
       z[[i]] <- naive_vector_dot(x[i,], y)
     z
   }

这说明：

.. code-block:: text

   Matrix × Vector
        =
   每一行 × Vector
        =
   多次 Vector Dot Product

注意：当至少一个 tensor 超过一维以后，点积通常不再具有交换性：

.. code-block:: text

   x %*% y  !=  y %*% x

Matrix · Matrix
---------------

两个矩阵可以相乘的条件是：

.. code-block:: text

   ncol(x) == nrow(y)

如果：

.. code-block:: text

   x shape = (a, b)
   y shape = (b, c)

那么：

.. code-block:: text

   x %*% y shape = (a, c)

每个输出元素都是：

.. math::

   z_{ij} = row_i(x) \cdot column_j(y)

Naive implementation：

.. code-block:: r

   naive_matrix_dot <- function(x, y) {
     z <- matrix(0, nrow = nrow(x), ncol = ncol(y))
     for (i in 1:nrow(x))
       for (j in 1:ncol(y)) {
         row_x <- x[i,]
         column_y <- y[,j]
         z[i, j] <- naive_vector_dot(row_x, column_y)
       }
     z
   }

记忆方法：

.. code-block:: text

   (a, b) · (b, c) -> (a, c)

   中间的 b 必须相同；
   最终保留外面的 a 和 c。

Higher-dimensional Tensor Dot
-----------------------------

同样的 shape compatibility 可以扩展到高维 tensor：

.. code-block:: text

   (a, b, c, d) · (d)    -> (a, b, c)

   (a, b, c, d) · (d, e) -> (a, b, c, e)

核心仍然是：参与点积的维度必须 compatible。


2. Tensor Reshaping（张量变形）
===============================

Reshaping 的含义
----------------

Tensor reshaping 是改变 tensor 的 shape，而不改变其中 coefficient 的总数量。

例如 MNIST 原始图像：

.. code-block:: text

   (60000, 28, 28)

每张图片是 ``28 × 28``。

为了输入 Dense layer，需要变成：

.. code-block:: text

   (60000, 784)

因为：

.. math::

   28 \times 28 = 784

R/Keras：

.. code-block:: r

   train_images <- array_reshape(
     train_images,
     c(60000, 28 * 28)
   )

重要：Keras 中优先使用 ``array_reshape()``，而不是直接修改 ``dim``。
提供的材料指出，这样可以按照与 NumPy、TensorFlow 等数值库兼容的 row-major semantics
重新解释数据。

元素数量必须保持不变
--------------------

例如：

.. code-block:: text

   (3, 2)
     ↓
   (6, 1)
     ↓
   (2, 3)

都包含：

.. math::

   6

个元素。

Transpose（转置）
-----------------

Transposition 是 reshaping 的特殊情况：交换 rows 和 columns。

R：

.. code-block:: r

   x <- matrix(0, nrow = 300, ncol = 20)

   dim(x)
   # 300 20

   x <- t(x)

   dim(x)
   # 20 300

即：

.. code-block:: text

   (300, 20)
       ↓
   transpose
       ↓
   (20, 300)


3. Tensor Operations 的几何意义
================================

Tensor 中的数值可以看成几何空间中的 coordinates。

例如：

.. code-block:: text

   A = [0.5, 1.0]

可以理解为二维空间中的一个 point，也可以理解为从 origin 指向该点的 vector。

Vector Addition
---------------

如果：

.. code-block:: text

   A = [0.5, 1.0]
   B = [1.0, 0.25]

那么 ``A + B`` 在几何上可以理解为 vector arrows 的连接。

更一般地，很多几何变换都可以通过 tensor operations 表示，例如：

* translation
* scaling
* rotation
* affine transformation

例如二维旋转可以通过向量与 ``2 × 2`` rotation matrix 做 dot product 来实现。


4. Deep Learning 的几何解释
============================

神经网络可以理解成：

**在高维空间中，对输入数据连续执行大量简单的几何变换。**

每一层：

.. code-block:: text

   input
     ↓
   transformation
     ↓
   representation

多个 layers：

.. code-block:: text

   Input
     ↓
   Layer 1
     ↓
   Layer 2
     ↓
   Layer 3
     ↓
   ...
     ↓
   Output

材料使用“揉皱的红蓝纸”作为直觉：

.. code-block:: text

   complicated / folded data
            ↓
      transformation
            ↓
      transformation
            ↓
      transformation
            ↓
   classes become separable

Deep Learning 的 ``deep`` 可以理解为：用很多层简单 transformation，
逐渐完成一个复杂的 transformation，使复杂的数据 manifold 更容易被区分。


5. Neural Layer 的核心计算
==========================

材料给出的基本形式：

.. code-block:: text

   output = relu(dot(W, input) + b)

其中：

``input``
   输入 tensor。

``W``
   weight matrix / kernel。

``b``
   bias。

``dot(W, input)``
   点积/矩阵运算。

``relu()``
   activation function。

``W`` 和 ``b`` 是 layer 的 **trainable parameters / weights**。

最重要的理解：

**模型学到的知识最终保存在这些 weight tensors 中。**


6. Random Initialization
========================

训练开始前：

.. code-block:: text

   W = small random values
   b = initial values

这称为：

**random initialization**

随机 weights 一开始不会产生有意义的 representation。

因此模型需要不断：

.. code-block:: text

   prediction
       ↓
   calculate loss
       ↓
   adjust weights
       ↓
   better prediction

这个不断调整参数的过程就是 training / learning。


7. Training Loop（训练循环）
============================

最基本的训练过程
----------------

每轮训练的核心步骤：

1. Draw a batch of training samples ``x`` and targets ``y``。
2. Forward pass：输入 ``x``，得到 ``y_pred``。
3. Compute loss：比较 ``y_pred`` 与真实 ``y``。
4. Compute gradient / backward pass。
5. Update weights，使 loss 下降。

可以记成：

.. code-block:: text

   x + y
     ↓
   Forward Pass
     ↓
   y_pred
     ↓
   Loss
     ↓
   Backward Pass
     ↓
   Gradient
     ↓
   Optimizer
     ↓
   Update W and b
     ↓
   repeat

Learning 的本质
---------------

Learning 就是：

**找到一组 model parameters，使 loss function 尽可能小。**

也就是：

.. math::

   \min_W Loss(W)


8. Loss Function
================

Loss 衡量：

**prediction 与 target 之间有多大的 mismatch。**

.. code-block:: text

   prediction close to target
          ↓
      small loss

   prediction far from target
          ↓
      large loss

训练的目标不是直接“最大化 accuracy”，而是通过 optimizer 不断最小化定义好的 loss。

MNIST 示例使用：

.. code-block:: r

   loss = "categorical_crossentropy"

因此：

.. code-block:: text

   categorical_crossentropy
            ↓
       feedback signal
            ↓
      update weights


9. Derivative（导数）
====================

考虑：

.. math::

   y = f(x)

如果 ``x`` 改变一点：

.. math::

   x + \epsilon_x

则 ``y`` 也发生小变化：

.. math::

   f(x + \epsilon_x) = y + \epsilon_y

当函数 smooth 且变化足够小时，可以局部近似为：

.. math::

   f(x + \epsilon_x) \approx y + a\epsilon_x

这里的 ``a`` 就是该点的 derivative / slope。

Derivative 的符号
-----------------

``a > 0``
   增加 ``x``，``f(x)`` 倾向增加。

``a < 0``
   增加 ``x``，``f(x)`` 倾向减少。

``|a|`` 越大
   ``f(x)`` 对 ``x`` 的局部变化越敏感。

如果目标是减少 ``f(x)``：

**向 derivative 的反方向移动。**


10. Gradient（梯度）
===================

Derivative 通常描述 scalar input。

Gradient 是 derivative 对 multidimensional / tensor input 的推广。

例如：

.. code-block:: text

   y_pred = dot(W, x)

   loss_value = loss(y_pred, y)

固定 ``x`` 和 ``y`` 后：

.. code-block:: text

   loss_value = f(W)

此时 ``W`` 是 matrix/tensor，因此：

.. code-block:: text

   gradient(f)(W)

也是一个与 ``W`` shape 相同的 tensor。

每个元素：

.. code-block:: text

   gradient[i, j]

描述：

**改变 W[i, j] 会让 loss 朝哪个方向变化，以及变化有多强。**

Gradient Descent 的核心公式
---------------------------

.. math::

   W_{new} = W_{old} - step \times gradient

为什么是 ``-``？

因为 gradient 指向局部上升方向，所以为了降低 loss，要朝反方向移动。

``step`` 对应学习过程中控制更新幅度的 scaling factor，通常与 learning rate 的概念联系起来。


11. 为什么不能一个 Weight 一个 Weight 地试？
=============================================

最 naive 的办法：

.. code-block:: text

   修改一个 weight
       ↓
   forward pass
       ↓
   看 loss
       ↓
   再修改
       ↓
   再 forward pass

但现代 neural network 可能有：

.. code-block:: text

   thousands
   millions
   tens of millions
   ... parameters

如果每个 parameter 都单独尝试，需要大量 forward passes，效率极低。

Gradient 的意义就在于：

**一次 backward computation 可以高效得到大量参数应该如何改变的信息。**


12. Stochastic Gradient Descent（SGD）
=====================================

Mini-batch SGD
--------------

训练通常不是一次使用整个 dataset，而是随机抽取一个 batch。

算法：

1. Draw random batch ``x, y``。
2. Forward pass 得到 ``y_pred``。
3. Compute loss。
4. Backward pass 得到 gradient。
5. ``W = W - learning_rate * gradient``。
6. Repeat。

``stochastic`` 的含义是：

**random / 随机。**

三种方式
--------

True SGD
   每次使用 1 个 sample。

Mini-batch SGD
   每次使用一小批 samples。

Batch SGD
   每次使用整个 dataset。

对比：

.. code-block:: text

   True SGD
   1 sample/update
   → cheap but noisy

   Mini-batch SGD
   small batch/update
   → practical compromise

   Batch SGD
   all samples/update
   → accurate update but expensive

现代 deep learning 最常见的是：

**mini-batch training**。


13. Learning Rate / Step Size
=============================

Learning rate 决定：

**每次沿 gradient 反方向走多远。**

太小：

.. code-block:: text

   tiny updates
      ↓
   slow convergence

太大：

.. code-block:: text

   huge updates
      ↓
   overshoot / unstable movement

因此 learning rate 是 neural-network training 中非常重要的 hyperparameter。


14. Local Minimum 与 Global Minimum
===================================

Local minimum
   周围看起来已经最低，但不是整个 loss landscape 的最低点。

Global minimum
   整个 loss function 中的最低点。

直觉：

.. code-block:: text

        \       /
         \__   /     <- local minimum
            \ /
             V       <- deeper/global minimum

简单 SGD 可能在某些情况下受到局部结构影响。


15. Momentum
============

Momentum 借用了物理中的“惯性”概念。

想象一个球沿 loss surface 向下滚：

.. code-block:: text

   gradient = current slope
   momentum = previous movement / velocity

因此下一次 update 不只考虑：

**当前 gradient**

还考虑：

**过去的更新方向。**

材料给出的 naive 形式：

.. code-block:: r

   past_velocity <- 0
   momentum <- 0.1

   while (loss > 0.01) {

     params <- get_current_parameters()

     w <- params$w
     loss <- params$loss
     gradient <- params$gradient

     velocity <-
       past_velocity * momentum +
       learning_rate * gradient

     w <-
       w +
       momentum * velocity -
       learning_rate * gradient

     past_velocity <- velocity

     update_parameter(w)
   }

Momentum 主要帮助理解：

* convergence speed
* local minima / ravines
* 利用过去 update 的方向信息


16. Optimizer
=============

Optimizer 决定：

**如何使用 gradient 来更新 model parameters。**

材料提到：

* SGD
* SGD with momentum
* Adagrad
* RMSProp

因此：

.. code-block:: text

   Gradient
      ↓
   Optimizer
      ↓
   rules for updating parameters
      ↓
   New weights

Optimizer 与 Loss 是两个不同概念：

``Loss``
   告诉模型“现在错得有多严重”。

``Optimizer``
   决定“根据 gradient 应该怎么修改 weights”。


17. Backpropagation（反向传播）
==============================

神经网络是很多 operations 的 chain。

例如：

.. code-block:: text

   f(W1, W2, W3)
       =
   a(W1, b(W2, c(W3)))

为了知道每个 weight 对最终 loss 的影响，需要使用：

**Chain Rule（链式法则）**

基本形式：

.. math::

   \frac{d}{dx}f(g(x))
   =
   f'(g(x))g'(x)

Backpropagation 的思想：

.. code-block:: text

   Final Loss
       ↑
   Layer 3
       ↑
   Layer 2
       ↑
   Layer 1

从最终 loss 开始向后传播，利用 chain rule 计算每个 parameter 对 loss 的贡献。

因此：

.. code-block:: text

   Forward Pass
       ↓
   Prediction
       ↓
   Loss
       ↓
   Backpropagation
       ↓
   Gradients
       ↓
   Optimizer
       ↓
   Weight Update

现代 framework 会自动完成 gradient computation，因此通常不需要手写 backpropagation。


18. MNIST 数据预处理
====================

加载数据：

.. code-block:: r

   library(keras)

   mnist <- dataset_mnist()

   train_images <- mnist$train$x
   test_images <- mnist$test$x

Reshape：

.. code-block:: r

   train_images <-
     array_reshape(train_images, c(60000, 28 * 28))

   test_images <-
     array_reshape(test_images, c(10000, 28 * 28))

Normalize：

.. code-block:: r

   train_images <- train_images / 255
   test_images <- test_images / 255

最终 shape：

.. code-block:: text

   Training:
   (60000, 784)

   Testing:
   (10000, 784)


19. MNIST Neural Network
========================

模型：

.. code-block:: r

   network <- keras_model_sequential() %>%
     layer_dense(
       units = 512,
       activation = "relu",
       input_shape = c(28 * 28)
     ) %>%
     layer_dense(
       units = 10,
       activation = "softmax"
     )

第一层
------

.. code-block:: text

   Input: 784 features
        ↓
   Dense: 512 units
        ↓
   ReLU

其核心可以抽象成：

.. code-block:: text

   output = relu(dot(W, input) + b)

第二层
------

.. code-block:: text

   512
    ↓
   Dense(10)
    ↓
   Softmax
    ↓
   10 class probabilities

这里 10 个输出对应 MNIST 的数字类别 ``0–9``。


20. Weight Tensor = 模型的“知识”
================================

Dense layer 中包含 trainable weights。

训练前：

.. code-block:: text

   random weights

训练中：

.. code-block:: text

   loss
    ↓
   gradient
    ↓
   optimizer
    ↓
   update weights

训练后：

.. code-block:: text

   learned weights

因此材料强调：

**模型学习到的信息持续存在于 weight tensors 中。**


21. R Pipe Operator ``%>%``
===========================

``%>%`` 来自 ``magrittr``。

它把左边的对象作为右边 function 的第一个 argument。

例如：

.. code-block:: r

   network <- keras_model_sequential() %>%
     layer_dense(units = 512, activation = "relu") %>%
     layer_dense(units = 10, activation = "softmax")

概念上相当于：

.. code-block:: r

   network <- keras_model_sequential()

   layer_dense(
     network,
     units = 512,
     activation = "relu"
   )

   layer_dense(
     network,
     units = 10,
     activation = "softmax"
   )

优势：

* 更 compact。
* 更 readable。
* 清楚表达 operation chain。


22. Keras Model 的 In-place Modification
========================================

材料特别指出，Keras model 与很多普通 R object 不同。

例如：

.. code-block:: r

   network %>% compile(...)

并不是：

.. code-block:: text

   old network
       ↓
   create independent new network

而是对现有的 ``network`` object 进行修改。

Keras model 是 layers 组成的 directed acyclic graph，其 state 会在 training 中更新。


23. Compile
===========

代码：

.. code-block:: r

   network %>% compile(
     optimizer = "rmsprop",
     loss = "categorical_crossentropy",
     metrics = c("accuracy")
   )

三个重要参数：

``optimizer``
   如何利用 gradient 更新 weights。

``loss``
   模型训练时需要最小化的目标。

``metrics``
   训练/评估过程中希望观察的指标，例如 accuracy。

关系：

.. code-block:: text

   categorical_crossentropy
           ↓
         Loss
           ↓
      Backpropagation
           ↓
        Gradient
           ↓
        RMSProp
           ↓
      Update weights


24. fit()
=========

代码：

.. code-block:: r

   network %>% fit(
     train_images,
     train_labels,
     epochs = 5,
     batch_size = 128
   )

``batch_size = 128``
   每次 gradient update 使用 128 个 samples。

``epochs = 5``
   整个 training dataset 被完整遍历 5 次。

Epoch
-----

**一个 epoch = 模型完整看一遍全部 training data。**

材料中的 MNIST：

.. code-block:: text

   60000 training samples
   batch size = 128

约：

.. code-block:: text

   469 updates / epoch

5 epochs：

.. code-block:: text

   469 × 5
   =
   2345 gradient updates


25. 从 ``fit()`` 看完整训练过程
===============================

当你执行：

.. code-block:: r

   network %>% fit(...)

背后发生的是：

.. code-block:: text

   Training Data
        ↓
   Mini-batch (128)
        ↓
   Forward Pass
        ↓
   Prediction
        ↓
   categorical_crossentropy
        ↓
   Loss
        ↓
   Backpropagation
        ↓
   Gradients
        ↓
   RMSProp
        ↓
   Update W and b
        ↓
   Next Mini-batch
        ↓
       ...
        ↓
   Complete one epoch
        ↓
   Repeat for 5 epochs


26. 本章最重要的逻辑链
======================

如果只记一张图，记下面这一张：

.. code-block:: text

   INPUT DATA
       ↓
   TENSORS
       ↓
   DOT PRODUCT / RESHAPING / OTHER TENSOR OPERATIONS
       ↓
   NEURAL NETWORK LAYERS
       ↓
   W · x + b
       ↓
   ACTIVATION
       ↓
   PREDICTION
       ↓
   LOSS
       ↓
   DERIVATIVE
       ↓
   GRADIENT
       ↓
   BACKPROPAGATION
       ↓
   OPTIMIZER
       ↓
   UPDATE WEIGHTS
       ↓
   LOWER LOSS
       ↓
   REPEAT
       ↓
   LEARNING


27. 必须掌握的术语
==================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Term
     - 核心含义
   * - Tensor
     - 多维数值数据结构。
   * - Shape
     - Tensor 每个 axis 的大小。
   * - Dot product
     - 将 tensor 中的元素组合起来的核心运算。
   * - Reshaping
     - 改变 shape，但保持 coefficient 总数不变。
   * - Transpose
     - 交换 matrix 的 rows 和 columns。
   * - Weight
     - 模型通过 training 学习的参数。
   * - Bias
     - Layer 中另一类 trainable parameter。
   * - Forward pass
     - 从 input 计算 prediction。
   * - Loss
     - prediction 与 target mismatch 的度量。
   * - Derivative
     - 一维函数在局部的变化率/斜率。
   * - Gradient
     - Derivative 对 tensor/multidimensional parameters 的推广。
   * - Learning rate
     - 控制每次 parameter update 的步长。
   * - SGD
     - 利用随机样本/批次的 gradient 更新 parameters。
   * - Mini-batch
     - 每次训练使用的一小批 samples。
   * - Epoch
     - 完整遍历一次 training dataset。
   * - Momentum
     - 更新时同时利用过去 movement 的信息。
   * - Optimizer
     - 定义如何根据 gradient 更新 parameters。
   * - Backpropagation
     - 用 chain rule 从 loss 向后计算 gradients。
   * - Trainable parameter
     - 可以通过 training 更新的 parameter。
   * - Random initialization
     - 训练开始时用小随机值初始化 weights。


28. 考试/复习版：十句话记住本章
==============================

1. Neural networks operate on **tensors**。
2. Dense layers heavily rely on **dot products**。
3. ``x %*% y`` 是 R 中的 dot/matrix product，而 ``x * y`` 是 element-wise product。
4. Matrix multiplication 要求左矩阵 columns 数等于右矩阵 rows 数。
5. **Reshaping changes shape, not the total number of coefficients.**
6. Neural network layers can be viewed as geometric transformations。
7. **Weights contain the information learned by the network.**
8. Training 的目标是通过 gradient-based optimization **minimize loss**。
9. Backpropagation 利用 **chain rule** 高效计算每个 parameter 的 gradient。
10. Optimizer 使用 gradients 更新 weights；反复执行后模型逐渐学会 input → target mapping。


29. 一句话理解神经网络训练
==========================

.. important::

   **Deep learning = 用一连串可微分的 tensor transformations 产生 prediction，
   用 loss 衡量错误，通过 backpropagation 得到 gradients，再由 optimizer
   不断修改 weights，使 loss 越来越小。**


30. 下一阶段学习时要特别关注
============================

在继续后面的 Deep Learning 内容时，看到任何模型都可以主动问下面几个问题：

1. **Input tensor 的 shape 是什么？**
2. **这一层对 tensor 做了什么 operation？**
3. **这一层的 output shape 是什么？**
4. **哪些 tensors 是 trainable weights？**
5. **Activation function 是什么？**
6. **Loss function 是什么？**
7. **Optimizer 是什么？**
8. **Batch size 和 epochs 是多少？**
9. **Gradient 如何从 loss 传播回 weights？**
10. **最终 prediction 如何对应真实 target？**

如果这十个问题都能回答，通常就已经理解了这个 neural network 的基本训练逻辑。
"""

out = Path("/mnt/data/deep_learning_chapter2_summary.rst")
out.write_text(rst, encoding="utf-8")
print(f"Created: {out}")
print(f"Lines: {len(rst.splitlines())}")

STDOUT/STDERR
Created: /mnt/data/deep_learning_chapter2_summary.rst
Lines: 1345

31. Why ``dz/dw1 = x1`` in Gradient Descent?
=============================================

Consider a single neuron before applying the activation function:

.. math::

   z = w_1x_1 + w_2x_2 + b

where:

``x1, x2``
   Input features.

``w1, w2``
   Trainable weights.

``b``
   Bias.

``z``
   The neuron's linear output before the activation function.


Partial Derivative with Respect to ``w1``
------------------------------------------

We want to know:

.. math::

   \frac{\partial z}{\partial w_1}

This means:

**If ``w1`` changes slightly, how much will ``z`` change?**

When taking the partial derivative with respect to ``w1``, all other variables are treated as constants:

.. math::

   \frac{\partial z}{\partial w_1}
   =
   \frac{\partial}{\partial w_1}
   (w_1x_1 + w_2x_2 + b)

Separate the terms:

.. math::

   \frac{\partial z}{\partial w_1}
   =
   \frac{\partial(w_1x_1)}{\partial w_1}
   +
   \frac{\partial(w_2x_2)}{\partial w_1}
   +
   \frac{\partial b}{\partial w_1}

Because ``x1`` is treated as a constant:

.. math::

   \frac{\partial(w_1x_1)}{\partial w_1}
   =
   x_1
   \frac{\partial w_1}{\partial w_1}

and:

.. math::

   \frac{\partial w_1}{\partial w_1} = 1

Therefore:

.. math::

   \frac{\partial(w_1x_1)}{\partial w_1}
   =
   x_1

The other terms do not contain ``w1``:

.. math::

   \frac{\partial(w_2x_2)}{\partial w_1} = 0

and:

.. math::

   \frac{\partial b}{\partial w_1} = 0

Therefore:

.. important::

   .. math::

      \boxed{
      \frac{\partial z}{\partial w_1} = x_1
      }


Intuitive Example
-----------------

Suppose:

.. math::

   z = 3w_1 + 2w_2 + 1

and keep:

.. math::

   w_2 = 4

Then:

.. math::

   z = 3w_1 + 9

Try different values of ``w1``:

.. code-block:: text

   w1 = 1  ->  z = 12

   w1 = 2  ->  z = 15

   w1 = 3  ->  z = 18

Every time ``w1`` increases by ``1``, ``z`` increases by ``3``.

Therefore the slope of ``z`` with respect to ``w1`` is:

.. math::

   \frac{\partial z}{\partial w_1} = 3

Because:

.. math::

   x_1 = 3

we again get:

.. math::

   \boxed{
   \frac{\partial z}{\partial w_1} = x_1
   }


Meaning in a Neural Network
---------------------------

The equation:

.. math::

   \frac{\partial z}{\partial w_1} = x_1

tells us how sensitive the neuron's output ``z`` is to changes in the weight ``w1``.

If:

.. math::

   x_1 = 100

then:

.. math::

   \frac{\partial z}{\partial w_1} = 100

A small change in ``w1`` can cause a relatively large change in ``z``.

But if:

.. math::

   x_1 = 0

then:

.. math::

   \frac{\partial z}{\partial w_1} = 0

because:

.. math::

   w_1x_1 = w_1(0) = 0

Changing ``w1`` has no immediate effect on ``z`` through this connection.


Connection to Gradient Descent
------------------------------

Gradient Descent is usually not directly interested in:

.. math::

   \frac{\partial z}{\partial w_1}

Instead, it needs:

.. math::

   \frac{\partial Loss}{\partial w_1}

because the goal of training is to determine:

**How should ``w1`` change in order to reduce the loss?**

The computation follows the path:

.. code-block:: text

   w1
    |
    v
   z = w1*x1 + w2*x2 + b
    |
    v
   Activation Function
    |
    v
   Prediction
    |
    v
   Loss


Chain Rule
----------

Because ``w1`` affects ``z``, ``z`` affects the prediction, and the prediction affects the loss,
we use the **chain rule**:

.. math::

   \frac{\partial Loss}{\partial w_1}
   =
   \frac{\partial Loss}{\partial prediction}
   \times
   \frac{\partial prediction}{\partial z}
   \times
   \frac{\partial z}{\partial w_1}

We already know:

.. math::

   \frac{\partial z}{\partial w_1} = x_1

Therefore:

.. math::

   \frac{\partial Loss}{\partial w_1}
   =
   \frac{\partial Loss}{\partial prediction}
   \times
   \frac{\partial prediction}{\partial z}
   \times
   x_1


Updating the Weight
-------------------

Once the gradient has been calculated, Gradient Descent updates ``w1``:

.. math::

   w_1^{new}
   =
   w_1^{old}
   -
   \eta
   \frac{\partial Loss}{\partial w_1}

where:

``η``
   Learning rate.

``∂Loss/∂w1``
   The gradient telling us how changing ``w1`` affects the loss.

The minus sign means that the weight moves in the direction opposite to the gradient,
because Gradient Descent attempts to reduce the loss.


Key Idea
--------

.. important::

   For

   .. math::

      z = w_1x_1 + w_2x_2 + b

   we have:

   .. math::

      \frac{\partial z}{\partial w_1} = x_1

   because ``x1`` is constant with respect to ``w1``, while
   ``w2*x2`` and ``b`` do not depend on ``w1``.

This forms one small part of the larger Backpropagation process:

.. code-block:: text

   Weight
     ↓
   Linear operation
     ↓
   Activation
     ↓
   Prediction
     ↓
   Loss
     ↓
   Chain Rule
     ↓
   Gradient
     ↓
   Gradient Descent
     ↓
   Update Weight