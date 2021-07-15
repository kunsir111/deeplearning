import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

# float32一般的显卡只支持32位的浮点数
xy = np.loadtxt("E:/data/diabetes/diabetes.csv.gz", delimiter=',', dtype=np.float32)
# torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
x_data = torch.from_numpy(xy[:, :-1])
# print(x_data.size())

# 中括号作用保持最后一列是数组
y_data = torch.from_numpy(xy[:, [-1]])
# print(y_data.shape)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# for epoch in range(100):
#     # 1 Forward
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#
#     # 2 Backward
#     optimizer.zero_grad()
#     loss.backward()
#     # 3 .Update
#     optimizer.step()
#
#     print(loss.item())

class diabetesDataset(Data.Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
dataset = diabetesDataset("E:/data/diabetes/diabetes.csv.gz")
train_loader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
if __name__ == '__main__':

    for epoch in range(10 ):
        train_loss_sum = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            train_loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:", epoch, train_loss_sum)