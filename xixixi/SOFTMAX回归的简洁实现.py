import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data as Data
import sys

num_inputs = 28*28
num_outputs = 10
batch_size = 256

mnist_train = torchvision.datasets.FashionMNIST("E:/data/", train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST("E:/data/", train=False, download=False, transform=transforms.ToTensor())

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.layer1 = nn.Linear(num_inputs, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_outputs)
        self.Relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
     #   x = self.Relu(x)
        x = self.layer2(x)
        #x = self.Relu(x)
        x = self.layer3(x)
        return x

from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, num_outputs))
# ]))

net = LinearNet(num_inputs, num_outputs)

torch.nn.init.normal_(net.layer1.weight, mean=0, std=0.01)
torch.nn.init.constant_(net.layer1.bias, val=0)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.03)

num_epochs = 10
print(net)

def evaluate_accuracy(data_iter, net) -> torch.float32:
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

for epoch in range(num_epochs):
    train_l_sum, n, train_acc_sum = 0.0, 0, 0.0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        n += y.shape[0]
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
    test_acc = evaluate_accuracy(test_iter, net)
    print('epoch %d, loss %f, train acc %f, test acc %f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

