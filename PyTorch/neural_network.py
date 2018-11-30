import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

# References:
# https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c

# hyper parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# download dataset
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())


# load dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


# Neural Network Class
class Net(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out


net = Net(input_size, hidden_size, num_classes)

# enable gpu if available
if torch.cuda.is_available():
  net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = Variable(images.view(-1, 28*28))
    labels = Variable(labels)

    optimizer.zero_grad()
    output = net(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:  # Logging
      print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
            % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

correct = 0
total = 0
for images, labels in test_loader:
  images = Variable(images.view(-1, 28 * 28))
  outputs = net(images)
  _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
  total += labels.size(0)  # Increment the total count
  correct += (predicted == labels).sum()  # Increment the correct count

print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))