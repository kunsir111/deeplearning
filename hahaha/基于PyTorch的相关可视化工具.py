import torch
import torchvision
import torchvision.transforms as transtorms
import torch.utils.data as Data
import torch.nn as nn
import torch
import torch.nn as nn
import torchvision
from visdom import Visdom
import numpy as np


mnist_train = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transtorms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./data/', train=False, download=True, transform=transtorms.ToTensor())

batch_size = 128

train_iter = Data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=3)
test_iter = Data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=3)

class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.out(x)
        return output

MyConvnet = Convnet()


if __name__ == '__main__':
    vis = Visdom()
    x = torch.linspace(6, -6, 100).view(-1, 1)
    sigmoid = torch.nn.Sigmoid()
    sigmoidy = sigmoid(x)
    tanh = torch.nn.Tanh()
    tanhy = tanh(x)
    relu = torch.nn.ReLU()
    reluy = relu(x)
    ploty = torch.cat((sigmoidy, tanhy, reluy), dim=1)
    plotx = torch.cat((x, x, x), dim=1)
    vis.line(Y=ploty, X=plotx, win='line plot', env='main',
             opts=dict(dash=np.array(['solid', 'dash', 'dashdot']),
                       legend=['Sigmoid', 'Tanh', 'ReLU'],
                        title='常用激活函数'))

    y1 = torch.sin(x)
    y2 = torch.cos(x)
    ploty = torch.cat((y1, y2), dim=1)
    plotx = torch.cat((x, x), dim=1)
    vis.stem(X=plotx, Y=ploty, win='stem plot', env='main1',
             opts=dict(legend=['sin', 'cos'],
                       title='茎叶图',
                    ))

    for step, (b_x, b_y) in enumerate(train_iter):
        print(b_x.shape)
        print(b_y.shape)
        break
    vis.image(b_x[0, :, :, :], win='one image', env='MyimagePlot',
              opts=dict(title='一张图像'))
    vis.images(b_x, win='my batch image', env='MyimagePlot',
               nrow=16, opts=dict(title='一个批次的图像'))

    texts = """A flexible tool for creating, organizing, and sharing visualization of live,rich data. Supports Torch and Numpy."""
    vis.text(texts, win='text plot', env='MyimagePlot',
             opts=dict(title='可视化文本'))
    vis.text()