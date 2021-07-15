import torch
import numpy as np

# # torch.from_numpy()的用法和解释
# xy = np.array([1, 2, 3])
# print(xy)
# # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
# xy1 = torch.from_numpy(xy)
# xy1[0] = 2
# print(xy)
# print(xy1)

x = np.array([[1, 2],
              [3, 4],
              [5, 6],
              ])
xy = np.array([[1, 2, 3],
               [4, 5, 6],
               ])
# xy1 = torch.from_numpy(xy)
# print(xy1)
# print(xy1[:, :-1])
# print(xy1[:,[-1]])
#
# list = [xy, xy]
# print(list)
# print(list.data)

# xy  = torch.tensor(xy, dtype = torch.float32)
# xy.requires_grad = True
# x = torch.tensor(x, dtype=torch.float32)
# y = torch.empty(3, 3)
# print(xy.size(), x.size())
# y = torch.mm(xy, x)
# print(y)
# y.backward()
# print(y)
x = np.array([[1, 2],
              [3, 4],
              [5, 6],
              ])

#id_before = id(x)
# x = torch.from_numpy(x)
# x = x.type(torch.float32)
# id_before = id(x)
# print(x)
# y = x.view(2, 3)
# print(x)
# print(id_before == id(x))
# print(x)
# x[0][1] = 10
# print(x)
# print(y)
# z = x.reshape(3, 2)
# print(z)

# x = torch.ones(2, 2)
# x.requires_grad = True
# y = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(y)
# y = x + 2
# print(x)
# print(x.grad_fn)
# print(y)
# print(y.grad_fn)
# print(x.is_leaf, y.is_leaf)
# z = y * y * 3
# out = z.mean()
# print(z, out)
# out.backward()
# print(x.grad)

li = ['0'] * 51

list = [[1,2],
        [3, 4],
        [5, 6]]
for l in list:
    li[l[0]:l[1] + 1] = '1' * (l[1] + 1 - l[0])
print(li)
List = []

