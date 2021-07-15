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

xy = np.array([[1, 2, 3],
               [4, 5, 6],
               ])
xy1 = torch.from_numpy(xy)
print(xy1)
print(xy1[:, :-1])
print(xy1[:,[-1]])