import random

import torch
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

num_input = 2;
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features =torch.from_numpy(np.random.normal(0, 1, (num_examples, num_input))).type(torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size=labels.size())).type(torch.float32)

# print(features[0], labels[0])

def use_svg_display():
    # ⽤⽮量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break
w = torch.tensor(np.random.normal(0, 0.01, (num_input, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

lr = 0.03
num_epoch = 10
net = linreg
loss = square_loss

for epoch in range(num_epoch):
    train_l_sum, n = 0.0, 0
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
        n += X.shape[0]
        train_l_sum += l.item()
    # train_l = loss(net(features, w, b), labels)
    print('epoch % d, loss %f' % (epoch + 1, train_l_sum / n))

print(true_w, "\n", w)
print(true_b, "\n", b)