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
      nn.Conv2d(1, 16)
    )