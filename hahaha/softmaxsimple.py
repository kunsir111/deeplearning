import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
import torch.nn as nn
import torch.utils.data as Data
from collections import OrderedDict

mnist_train = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

batch_size = 32
num_inputs = 28 * 28
num_outputs = 10

train_iter = Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=3)
test_iter = Data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=3)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self,x):
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    OrderedDict([
        ('falttenlayer', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

def train(epoch):
    train_l_sum, train_acc, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    print('epoch %d, train loss %lf, train acc %lf' % (epoch + 1, train_l_sum / n, train_acc / n))

def test():
    test_acc, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            test_acc += (net(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        print('test acc %lf' % (test_acc / n))

num_epochs = 10
if __name__ == '__main__':
    for epoch in range(num_epochs):
        train(epoch)
        test()
