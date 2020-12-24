import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
import visdom
import numpy as np

mnist_train = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

batch_size = 128

train_iter = Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0)
test_iter = Data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=0)

num_inputs = 28 * 28
num_outputs = 10

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0
            )
        )
        self.Polling1 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            )
        )
        self.Polling2 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=120,
                kernel_size=4,
                stride=1,
                padding=0
            )
        )
        self.fc = nn.Linear(120, 128)
        self.linear = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_outputs)
    def forward(self, x):
        x = self.conv1(x)
        x = self.Polling1(x)
        x = self.conv2(x)
        x = self.Polling2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.linear(x)
        x = self.out(x)
        return x

net = Convnet()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

acc_train = []
acc_test = []
def train(epoch):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    acc_train.append(train_acc_sum / n)
    print('epoch %d, train loss %lf, train acc %lf' % (epoch + 1, train_l_sum / n, train_acc_sum / n))

def test():
    test_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            test_acc_sum += (net(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        print('test acc %lf' % (test_acc_sum / n))
        acc_test.append(test_acc_sum / n)
num_epochs = 10
def print_loss():
    train_acc = torch.tensor(acc_train).view(-1, 1)
    test_acc = torch.tensor(acc_test).view(-1, 1)
    # train_acc = torch.tensor([1, 3, 4, 5, 6]).view(-1, 1)
    # test_acc = torch.tensor([2, 3, 4, 5, 6]).view(-1, 1)
    x = torch.tensor([x for x in range(0, num_epochs)]).view(-1, 1)
    ploty = torch.cat((train_acc, test_acc), dim=1)
    plotx = torch.cat((x, x), dim=1)
    print(ploty.shape)
    print(plotx.shape)
    visdom.Visdom().line(Y=ploty, X=plotx, win='line plo', env='main',
                       opts=dict(dash=np.array(['solid', 'solid']),legend=['train acc', 'test acc'], title='LeNet准确率'))
    # x = torch.linspace(6, -6, 100).view(-1, 1)
    # sigmoid = torch.nn.Sigmoid()
    # sigmoidy = sigmoid(x)
    # tanh = torch.nn.Tanh()
    # tanhy = tanh(x)
    # ploty = torch.cat((sigmoidy, tanhy), dim=1)
    # plotx = torch.cat((x, x), dim=1)
    # print(plotx.shape)
    # print(ploty.shape)
    # visdom.Visdom().line(Y=ploty, X=plotx, win='line plot', env='main',
    #          opts=dict(dash=np.array(['solid', 'dash']),
    #                    legend=['Sigmoid', 'Tanh'],
    #                    title='常用激活函数'))



if __name__ == '__main__':
    for epoch in range(num_epochs):
        lossNum = []
        train(epoch)
        test()
    print_loss()
    # for X, y in train_iter:
    #     print(X.size(), y.size())
    #     break