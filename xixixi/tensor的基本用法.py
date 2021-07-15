# 在PyTorch中， torch.Tensor 是存储和变换数据的主要⼯具。如果你之前⽤过NumPy，你会发现
# Tensor 和NumPy的多维数组⾮常类似。然⽽， Tensor 提供GPU计算和⾃动求梯度等更多功能，这
# 些使 Tensor 更加适合深度学习。
import numpy as np
import torch

# 创建⼀个5x3的未初始化的 Tensor ：
x = torch.empty(5, 3)
print(x)

# 创建⼀个5x3的随机初始化的 Tensor :
x = torch.rand(5, 3)
print(x)

#创建⼀个5x3的long型全0的 Tensor :
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 还可以直接根据数据创建:
x = torch.Tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.float32)
print(x)

# 通过 shape 或者 size() 来获取 Tensor 的形状
print(x.size())
print(x.shape)

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

res = torch.empty(5, 3)
torch.add(x, y, out=res)
print(res)
y.add_(x)
print(y)

# 注意 view() 返回的新tensor与源tensor共享内存（其实是同⼀个tensor），也即更改其中的⼀个，另
# 外⼀个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察⻆度)

# 所以如果我们想返回⼀个真正新的副本（即不共享内存）该怎么办呢？Pytorch还提供了⼀
# 个 reshape() 可以改变形状，但是此函数并不能保证返回的是其拷⻉，所以不推荐使⽤。推荐先
# ⽤ clone 创造⼀个副本然后再使⽤ view 。

x = torch.rand(5, 3)
x1 = x.view(3, 5)

print(x)

x1[2, 3] = 8
print(x)
print(x1)

x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)


# Tensor 转NumPy
# 使⽤ numpy() 将 Tensor 转换成NumPy数组:
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a += 1
print(a, b)
b += 1
print(a, b)

# NumPy数组转 Tensor
# 使⽤ from_numpy() 将NumPy数组转换成 Tensor :
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

c = torch.tensor(a)
a += 1
print(a, c)