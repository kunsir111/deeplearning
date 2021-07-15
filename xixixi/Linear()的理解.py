import torch
import torch.nn as nn

m = nn.Linear(20,30)
# 查看生成的权重矩阵的shape，会发现权重的shape为30*20
print(m.weight.size())

# 生成一个128*20的矩阵
input = torch.randn(128, 20)
# 查看生成矩阵的大小
print(input.size())

# 传入input后，线性变换得到的shape为128*20
print(m(input).size())
