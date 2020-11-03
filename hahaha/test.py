import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
from collections import OrderedDict
import hiddenlayer as hl

mnist_train = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

batch_size = 128
num_inputs = 28 * 28
num_outputs = 10

train_iter = Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=3)
test_iter = Data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=3)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.out(x)
        return output

MyConvnet = ConvNet()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MyConvnet.parameters(), lr=0.03)

# print(MyConvnet)

# init.normal_(MyConvnet.conv1..weight, mean=0, std=0.01)
# init.normal_(MyConvnet.conv2.weight, mean=0, std=0.01)
# init.normal_(MyConvnet.fc.weight, mean=0, std=0.01)
# init.normal_(MyConvnet.out.weight, mean=0, std=0.01)
# init.constant_(MyConvnet.conv1.bias, val=0)
# init.constant_(MyConvnet.conv2.bias, val=0)
# init.constant_(MyConvnet.fc.bias, val=0)
# init.constant_(MyConvnet.out.bias, val=0)

def train(epoch):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = MyConvnet(X)
        l = loss(y_hat, y).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    print('epoch %d, train loss %lf, train acc %lf' % (epoch + 1, train_l_sum / n, train_acc_sum / n))

def test():
    test_acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in test_iter:
            test_acc_sum += (MyConvnet(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        print('test acc %lf' % (test_acc_sum / n))
num_epochs = 10
if __name__ == '__main__':
    # for X, y in train_iter:
    #     print(X.shape, y.shape)
    #     break
    for epoch in range(num_epochs):
        train(epoch)
        test()
# hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
# hl_graph.theme = hl.graph.THEMES['blue'].copy()
# hl_graph.save("E:\\deeplearning\\hahaha\\data/MyConvnet_hl.png", format="png")