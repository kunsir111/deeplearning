import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs))).type(torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())).type(torch.float32)


batch_size = 10
# TensorDataset 可以用来对 tensor 进行打包，就好像 python 中的 zip 功能。该类通过每
# 一个 tensor 的第一个维度进行索引。因此，该类中的 tensor 第一维度必须相等。
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True)

# for X, y in data_iter:
#     print(X, y)
#     break

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        nn.Linear(n_feature, 1)

    def forward(self, x):
        return self.linear(x)

net = LinearNet(num_inputs)
print(net)

# net = nn.Sequential(
#     nn.Linear(num_inputs, 1)
# )
# print(net)
# print(net[0])
#
# print("="*80)
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# print(net)
# print(net[0])
#
# print("="*80)
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear', nn.Linear(num_inputs, 1)),
# ]))
# print(net)
# print(net[0])

for param in net.parameters():
    print(param)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr = 0.03)
print(optimizer)

num_epochs = 3

for epoch in range(num_epochs):
    train_l_sum, train_acc, n = 0.0, 0.0, 0
    for X, y in data_iter:
        y_hat = net(X)
        l = loss(y_hat, y.view(-1, 1)).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        n += X.shape[0]
        train_l_sum += l.item()

    print('epoch %d: train loss %f' % (epoch + 1, train_l_sum / n))

# print(true_w, "\n", net[0].weight)
# print(true_b, "\n", net[0].bias)

print("\n")
for param in net.parameters():
    print(param)
