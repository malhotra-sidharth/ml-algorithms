import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# Reference
# https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a-fashion-clothes-dataset-e589682df0c5
# https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

# hyper parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# download dataset
train_dataset = datasets.FashionMNIST(root='./data',
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)

test_dataset = datasets.FashionMNIST(root='./data',
                                      train=False,
                                      transform=transforms.ToTensor())

# load dataset
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=test_dataset,
                          shuffle=False,
                          batch_size=batch_size)


# Neural Network Class
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cLayer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=5, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(2))

    self.cLayer2 = nn.Sequential(
      nn.Conv2d(16, 32, kernel_size=5, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2))

    self.fc = nn.Linear(7*7*32, 10)

  def forward(self, x):
    out = self.cLayer1(x)
    out = self.cLayer2(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out