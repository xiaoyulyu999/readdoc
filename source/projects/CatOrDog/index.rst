Cat and Dog Image Classification
==============================

This project demonstrates how to build a **cat vs dog image classifier** using **PyTorch**.
It is based on a practical deep learning workflow including data preprocessing, model
construction, training, and evaluation.

Project Info
----------------------------
- Series: *Deep Learning 100 Examples – PyTorch Edition*
- Dataset: https://pan.baidu.com/s/1YREL1omT9YJrp9B1PBPTfQ
- Extraction Code: ionw

Environment
----------------------------

- Python 3.8
- Jupyter Lab
- torch==1.10.0+cu113
- torchvision==0.11.1+cu113

Import Libraries
----------------------------

.. code-block:: python

   import torch
   from torch import nn
   from torch.utils.data import DataLoader
   from torchvision import datasets
   from torchvision.transforms import ToTensor
   import torchvision.transforms as transforms
   import matplotlib.pyplot as plt
   import numpy as np

Dataset and Preprocessing
----------------------------

Dataset directories:

.. code-block:: python

   train_datadir = './1-cat-dog/train/'
   test_datadir  = './1-cat-dog/val/'

Data Augmentation Experiments
----------------------------

Different data augmentation strategies were tested:

=============================== ========
Method                          Accuracy
=============================== ========
No augmentation                 79.2%
Random rotation                 80.8%
Rotation + Gaussian blur        83.3%
Random vertical flip            73.3%
=============================== ========

Some augmentations improve performance, but some (e.g., vertical flip) may reduce accuracy.

Transforms
-----------

.. code-block:: python

   train_transforms = transforms.Compose([
       transforms.Resize([224, 224]),
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])

   test_transforms = transforms.Compose([
       transforms.Resize([224, 224]),
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])

Dataset Loading
----------------------------

.. code-block:: python

   train_data = datasets.ImageFolder(train_datadir, transform=train_transforms)
   test_data  = datasets.ImageFolder(test_datadir, transform=test_transforms)

   train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1)
   test_loader  = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=1)

Sample batch shape:

.. code-block:: text

   X: [4, 3, 224, 224]
   y: [4]

Model Architecture
----------------------------

The model is based on **LeNet**, adapted for RGB images and 224×224 resolution.

.. code-block:: python

   import torch.nn.functional as F

   device = "cuda" if torch.cuda.is_available() else "cpu"

   class LeNet(nn.Module):
       def __init__(self):
           super(LeNet, self).__init__()
           self.conv1 = nn.Conv2d(3, 6, 5)
           self.conv2 = nn.Conv2d(6, 16, 5)
           self.pool = nn.MaxPool2d(2, 2)

           self.fc1 = nn.Linear(16 * 53 * 53, 120)
           self.fc2 = nn.Linear(120, 84)
           self.fc3 = nn.Linear(84, 2)

       def forward(self, x):
           x = F.relu(self.conv1(x))
           x = self.pool(x)
           x = F.relu(self.conv2(x))
           x = self.pool(x)
           x = x.view(x.size(0), -1)
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)
           return x

   model = LeNet().to(device)

Loss Function and Optimizer
----------------------------

.. code-block:: python

   loss_fn = nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

Training Function
----------------------------

.. code-block:: python

   def train(dataloader, model, loss_fn, optimizer):
       size = len(dataloader.dataset)
       model.train()

       for batch, (X, y) in enumerate(dataloader):
           X, y = X.to(device), y.to(device)
           pred = model(X)
           loss = loss_fn(pred, y)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           if batch % 100 == 0:
               loss, current = loss.item(), batch * len(X)
               print(f"loss: {loss:>7f} [{current}/{size}]")

Testing Function
----------------------------

.. code-block:: python

   def test(dataloader, model, loss_fn):
       size = len(dataloader.dataset)
       num_batches = len(dataloader)
       model.eval()

       test_loss, correct = 0, 0
       with torch.no_grad():
           for X, y in dataloader:
               X, y = X.to(device), y.to(device)
               pred = model(X)
               test_loss += loss_fn(pred, y).item()
               correct += (pred.argmax(1) == y).type(torch.float).sum().item()

       test_loss /= num_batches
       correct /= size
       print(f"Accuracy: {100*correct:.1f}%, Avg loss: {test_loss}")

Training Process
----------------------------

.. code-block:: python

   epochs = 20
   for t in range(epochs):
       print(f"Epoch {t+1}")
       train(train_loader, model, loss_fn, optimizer)
       test(test_loader, model, loss_fn)

Final Result
----------------------------

After 20 epochs, the model reached:

- **Best accuracy: ~75.8%**
- **Final accuracy: ~73.3%**

This shows that even a simple CNN like LeNet can effectively perform
binary image classification on cats and dogs when combined with
proper preprocessing and training.

Problem
--------

.. code-block:: python

   self.fc1 = nn.Linear(16 * 53 * 53, 120)

The first fc has 5, 393, 280 parameters! The LeNet was designed for reading the image size 32 * 32
To fix this:

.. note::

   Add more pooling

   .. code-block:: python

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.pool(x) <--
   Too shallow, The model will learn nothing. The result accuracy is 50% means it just guesses.

.. note::

   Resize images to 32 * 32.

   .. code-block:: python

      transform.Resize((32, 32))
      //////////////////////////

      train_transforms = transforms.Compose([
          transforms.Resize([32, 32]),   # <- change here
          transforms.ToTensor(),
      ])

      test_transforms = transforms.Compose([
          transforms.Resize([32, 32]),   # <- change here
          transforms.ToTensor(),
      ])
      //////////////////////////
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
   The size of images is smaller but loss a lot of detail


The best choice can be changing the model from LeNet (designed for 32 * 32) to ResNet18 (convolutional neural network designed for 224×224 images)

.. code-block:: python

   import torchvision.models as models
   device = "mps" if torch.backends.mps.is_available() else "cpu"

   model = models.resnet18(pretrained=True)
   model.fc = nn.Linear(512, 2)
   model = model.to(device)
   '''
   Epoch 1
   loss: 0.831960 [0/480]
   loss: 0.327696 [400/480]
   Accuracy: 100.0%, Avg loss: 0.13268096558749676
   Epoch 2
   loss: 0.142731 [0/480]
   loss: 0.200979 [400/480]
   Accuracy: 100.0%, Avg loss: 0.06832013993213575
   Epoch 3
   loss: 0.027676 [0/480]
   loss: 0.037132 [400/480]
   Accuracy: 100.0%, Avg loss: 0.042580808202425637
   Epoch 4
   loss: 0.015773 [0/480]
   loss: 0.159375 [400/480]
   Accuracy: 100.0%, Avg loss: 0.02732905512675643
   Epoch 5
   loss: 0.145204 [0/480]
   loss: 0.076785 [400/480]
   Accuracy: 100.0%, Avg loss: 0.02395119732245803
   Epoch 6
   loss: 0.085170 [0/480]
   loss: 0.084924 [400/480]
   Accuracy: 100.0%, Avg loss: 0.020820383230845133
   Epoch 7
   loss: 0.006454 [0/480]
   loss: 0.005942 [400/480]
   Accuracy: 100.0%, Avg loss: 0.015191611143139501
   Epoch 8
   loss: 0.078428 [0/480]
   loss: 0.009044 [400/480]
   Accuracy: 100.0%, Avg loss: 0.010916504779985795
   Epoch 9
   loss: 0.038217 [0/480]
   loss: 0.061903 [400/480]
   Accuracy: 100.0%, Avg loss: 0.014720110261502365
   Epoch 10
   loss: 0.007591 [0/480]
   loss: 0.063153 [400/480]
   Accuracy: 100.0%, Avg loss: 0.012034913327079266
   Epoch 11
   loss: 0.798543 [0/480]
   loss: 0.013305 [400/480]
   Accuracy: 100.0%, Avg loss: 0.011147843091748655
   Epoch 12
   loss: 0.025731 [0/480]
   loss: 1.220648 [400/480]
   Accuracy: 100.0%, Avg loss: 0.009033751945632199
   Epoch 13
   loss: 0.047058 [0/480]
   loss: 0.709167 [400/480]
   Accuracy: 100.0%, Avg loss: 0.00851133301621303
   Epoch 14
   loss: 0.017307 [0/480]
   loss: 0.891749 [400/480]
   Accuracy: 100.0%, Avg loss: 0.010462252660848511
   Epoch 15
   loss: 1.194395 [0/480]
   loss: 0.013166 [400/480]
   Accuracy: 100.0%, Avg loss: 0.013428901451212975
   Epoch 16
   loss: 0.013937 [0/480]
   loss: 0.001417 [400/480]
   Accuracy: 100.0%, Avg loss: 0.0060176165619244175
   Epoch 17
   loss: 0.019566 [0/480]
   loss: 0.018773 [400/480]
   Accuracy: 100.0%, Avg loss: 0.006776365474797786
   Epoch 18
   loss: 0.001461 [0/480]
   loss: 0.053924 [400/480]
   Accuracy: 100.0%, Avg loss: 0.006957509882825738
   Epoch 19
   loss: 0.009045 [0/480]
   loss: 0.002122 [400/480]
   Accuracy: 100.0%, Avg loss: 0.006715904110266516
   Epoch 20
   loss: 0.002422 [0/480]
   loss: 0.001453 [400/480]
   Accuracy: 100.0%, Avg loss: 0.011973593776929192
   '''

.. warning:: NOT GOOD

   ResNet18 is showing 100% training and test accuracy after just a few epochs. While that looks impressive, it’s a classic sign of overfitting or a dataset/setup issue, not necessarily a perfect model.

