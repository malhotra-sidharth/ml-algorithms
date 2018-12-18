# References:
# https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/

import torch
import torch.nn as nn
import torchvision

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    # 1 input channel image (black/white images), 6 output channels, 3x3 kernel
    self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
    self.pool = nn.MaxPool2d(2)
    self.relu = nn.ReLU()
    self.input_size = 6*141*141
    self.fc1 = nn.Linear(self.input_size, 8)

  def forward(self, x):
    # output of first convolution
    # (3, 285, 285) => (6, 283, 283)
    x = self.relu(self.conv1(x))

    # max pooling
    # (6, 283, 283) => (6, 141, 141)
    x = self.pool(x)

    # print(x.shape)
    # reshape tensor
    x = x.view(-1, self.input_size)

    x = self.relu(x)

    return x