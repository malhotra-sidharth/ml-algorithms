# references
# https://www.superdatascience.com/opencv-face-detection/
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# dataset
# https://github.com/muxspace/facial_expressions
#
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from facecrop import CropFace
from network import CNN

# extract images names from folder
imgs = os.listdir('./data/images/cropped')

# get the emotions of each image
labls = pd.read_csv('./data/legend.csv')

# save cropped images
# UNCOMMENT BELOW LINES TO RE-CROP ALL IMAGES
cf = CropFace()
# cf.crop_multiple_images(images, './data/images')

# separate cropped files into respective class folders
# to be used by PyTorch Dataloader
# UNCOMMENT THIS LINE TO RE-SEPARATE THE DATASET FOR DATALOADER
# cf.separate_classes_for_dataloader('./data/images/cropped', labels, images)


# train neural network
image_data = ImageFolder('./data/images/cropped/dataset',
                         transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(image_data,
                                           shuffle=True,
                                           batch_size=500)

dataiter = iter(train_loader)
# images, labels = dataiter.next()

net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for epoch in range(2):
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 5 == 0:    # print every 2000 mini-batches
      print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print("Finished Training")    
    





