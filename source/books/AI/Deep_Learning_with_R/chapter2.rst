Chapter 2. Before we begin: the mathematical building blocks of neural networks
===============================================================================

Start with MNIST dataset in Keras. (Of course!)

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

   test_images <- array_reshape(test_images, c(60000, 28 * 28))
   test_images <- test_images / 255

Preparing the labels.

.. code-block:: R

   train_labels <- to_categorical(train_labels)
   test_labels <- to_categorical(test_labels)

.. important:: Why we need this step?

   Our output layer looks like : [0.01, 0.02, 0.01, 0.03, 0.02, 0.85, 0.01, 0.02, 0.02, 0.03] -> prediction is 5. (0.85). VS Actual: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], then optimizer will change weights base on this loss. Moreover, we choosed "categorical_crossentropy" as loss function from beginning.

