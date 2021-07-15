import torch
import copy

input = torch.randn(3, 3)
input1 = copy.deepcopy(input)
input = torch.sigmoid(input)
print(input)

m = torch.nn.Sigmoid()
print(m(input))

target = torch.FloatTensor([[0, 1, 1],
                            [0, 0 , 1],
                            [1, 0, 1]
                            ])

# 对于二分类问题的三个训练样本，假设我们得到了模型的预测值pred=[3,2,1]，而真实标签对应的是[1,1,0]，如果要使用
# BCELoss，要求样本必须在0~1之间，也就是需要调用sigmoid。
# 而采用BECWithLogitsLoss时，相当于把BCELoss和sigmoid融合了，也就是说省略了sigmoid这个步骤。
lo = abs(target * torch.log(input) + (1 - target) * torch.log(1 - input))
print(lo.mean().item())

lo = torch.nn.BCELoss()

print(lo(input, target).item())

lo = torch.nn.BCEWithLogitsLoss()(input1, target)

print(lo.item())