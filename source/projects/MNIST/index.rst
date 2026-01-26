MNIST Classification
==============================

This project demonstrates how to build a **handwritten digit classifier** using PyTorch.
The model is trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9).

The main goals of this project are:

- Understand how to load and preprocess image data in PyTorch.
- Build and train a neural network for image classification.
- Evaluate model performance on unseen test data.

Features
--------

- Fully connected neural network (MLP) for MNIST digit classification.
- Optional Convolutional Neural Network (CNN) for higher accuracy.
- Easy-to-run scripts with training and testing routines.
- GPU support for faster computation (if available).

Data Loader
-----------

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import torchvision
   import torchvision.transforms as transforms

   transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
      ])