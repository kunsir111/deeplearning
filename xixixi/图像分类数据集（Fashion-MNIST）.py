import torch
import torch.utils.data as Data
import sys
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np

mnist_train = torchvision.datasets.FashionMNIST(root="E:/data/", train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root="E:/data/", train=False, download=False, transform=transforms.ToTensor())

# print(type(mnist_train[0]))
# print(len(mnist_train), len(mnist_test))
# feature, label = mnist_train[0]
# print(feature.shape, label)

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

batch_size = 256

train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print(time.time() - start)

num_inputs  = 28 * 28
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01,(num_inputs, num_outputs)), dtype=torch.float32)
b = torch.zeros(num_outputs, dtype=torch.float32)

W.requires_grad = True
b.requires_grad = True

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim = 1, keepdim = True)
    return X_exp / partition

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

def evalute_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def sgd(params, batch_size, lr):
    for param in params:
        param.data -= lr * param.grad / batch_size


loss = cross_entropy
num_epochs, lr = 5, 0.01

for epoch in range(num_epochs):
    train_acc_sum, train_l_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()
        l.backward()
        sgd([W, b], batch_size, lr)
        train_l_sum += l.item()
        n += y.shape[0]
    train_acc_sum = evalute_accuracy(train_iter, net)
    print('epoch %d, train loss %f train acc %d' % (epoch + 1, train_l_sum / n, train_acc_sum / n))
