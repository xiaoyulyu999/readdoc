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

   # Download the dataset
   train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

   # dataloader
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

MLP
---

.. code-block:: python

   class MNISTModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.flatten = nn.Flatten()
           self.fc1 = nn.Linear(28 * 28, 128)
           self.fc2 = nn.Linear(128, 64)
           self.fc3 = nn.Linear(64, 10)
           self.relu = nn.ReLU()

       def forward(self, x):
           x = self.flatten(x)
           x = self.relu(self.fc1(x))
           x = self.relu(self.fc2(x))
           x = self.fc3(x)
           return x


This model is a **Multi-Layer Perceptron (MLP)** designed to classify handwritten digits from the MNIST dataset.
Each MNIST image has size **28 × 28** pixels and is converted into a vector of **784 values** before being processed by the network.

The goal of the model is to learn a mapping from an input image to one of the **10 digit classes (0–9)**.

Overall structure::

    Image → Vector → Feature Extraction → Digit Classification


Flatten Layer
~~~~~~~~~~~~~

The MNIST images are provided in the shape::

    (batch_size, 1, 28, 28)

MLP layers only accept 1D vectors, so the input is flattened into::

    (batch_size, 784)

Each pixel becomes one input feature to the neural network.


First Fully Connected Layer (fc1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first linear layer is defined as::

    nn.Linear(28*28, 128)

This layer maps **784 input pixels** to **128 learned features**.
Each neuron learns a weighted combination of pixels, allowing the network to detect simple patterns such as:

- Vertical strokes
- Horizontal strokes
- Bright regions
- Edges and curves

This is the first stage of feature extraction.


ReLU Activation
~~~~~~~~~~~~~~~~~~~~~~~~~~

After each linear layer, a ReLU activation is applied::

    ReLU(x) = max(0, x)

ReLU introduces **non-linearity**, which allows the network to model complex shapes and patterns.
Without ReLU, the entire network would behave like a single linear transformation and would not be powerful enough to recognize digits.


Second Fully Connected Layer (fc2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second linear layer is::

    nn.Linear(128, 64)

This layer compresses the 128 extracted features into **64 more abstract features**.
These neurons learn higher-level digit components such as loops, corners, and curves by combining the simpler features learned in the first layer.


Output Layer (fc3)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The final layer is::

    nn.Linear(64, 10)

This maps the 64 high-level features into **10 output values**, one for each digit (0–9).
These values are called **logits** and represent how strongly the model believes the input belongs to each digit class.
The class with the highest logit is chosen as the predicted digit.

Intuition
~~~~~~~~~~~~~

The model can be understood as::

    784 pixel sensors → 128 pattern detectors → 64 digit components → 10 digit scores

This hierarchical processing is why this MLP achieves around **97% accuracy** on MNIST.

Define loss function and optimizer
----------------------------------

.. code-block:: python

   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

Training the Model
------------------

.. code-block:: python

   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   model.to(device)

   epochs = 5
   for epoch in range(epochs):
       model.train()
       running_loss = 0.0
       for images, labels in train_loader:
           images, labels = images.to(device), labels.to(device)

           optimizer.zero_grad()           # 1
           outputs = model(images)         # 2
           loss = criterion(outputs, labels)  # 3
           loss.backward()                 # 4
           optimizer.step()                # 5

           running_loss += loss.item()

       print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

.. admonition:: 1.

   PyTorch accumulates gradients by default. If you don't clear them:
   - gradients pile up
   - learning becomes wrong

.. admonition:: 2.

   runs in the model: images → flatten → fc1 → ReLU → fc2 → ReLU → fc3 → logits

.. admonition:: 3.

   inputs : logits & ture labels
   - logits -> softmax -> log -> compare with labels
   - Loss answers: How wrong is the model right now?

.. admonition:: 4.

   This is where learning signals are computed. Computes gradients for every weight.

.. admonition:: 5.

   - new_weight = old_weight - learning_rate * gradient
   - This is actual learning step. Learning rate is adaptive. Momentum is applied.

.. code-block:: python
   '''
   Epoch [1/5], Loss: 0.3845
   Epoch [2/5], Loss: 0.1809
   Epoch [3/5], Loss: 0.1325
   Epoch [4/5], Loss: 0.1036
   Epoch [5/5], Loss: 0.0880
   '''

Evaluate
---------

.. code-block:: python

   model.eval()
   correct = 0
   total = 0

   with torch.no_grad():
       for images, labels in test_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

   print(f"Test Accuracy: {100 * correct / total:.2f}%")

   # Test Accuracy: 96.71%

Can make it better?
-------------------

1. MLP loses spatial information:

   28 × 28 image → 784 vector → Linear layers → output

   - Every pixel is treated as an independent feature

   - The model does not know which pixels are next to each other

   - Local patterns like vertical/horizontal strokes are not explicitly detected

   - The network has to learn relationships from scratch, which is harder

2. CNN preserves spatial structure:

   Conv2d + ReLU → detects edges and small patterns
   Pooling → focuses on important regions
