import cv2
import numpy as np

# img = cv2.imread("E:/data/Salt body recognition/tgs-salt-identification-challenge/train/images/000e218f21.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, dsize=(202, 202))
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# a = [1, 2, 3, 4, 5]
# print(np.pad(a, (1, 2), 'edge'))

# import torchvision
#
# base_model = torchvision.models.resnet18(pretra ined=True)
# print(list(base_model.children())[7])
#
# import torch
# w = torch.tensor([1.0])
# w.requires_grad = True
# print(w)
# import torch
#
# x1 = torch.tensor([1, 2, 3, 4]).view(4, 1)
# x2 = torch.tensor([1, 2, 4, 5]).view(4, 1)
# print(x1.size())
# print(x1.size())
# print()
# print((x1 == x2).sum().item())

# import torch
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim = True))
# print(X.sum(dim=1, keepdim = True))

# import pandas as pd
# df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
#                   index=['cobra', 'viper', 'sidewinder'],
#                   columns=['max_speed', 'shield'])
# print(df)
#
# df.loc[['viper', 'sidewinder'], ['shield']] = 50
# print(df)
#
# df.loc['viper'] = 10
# print(df)
#
# df.loc[df['shield']== 10] = 0
# print(df)

# import numpy as np
# x1 = np.array([1, 2, 3, 4])
# x2 = np.array([3, 4, 5, 6])
# print(np.setdiff1d(x1, x2))

# import time
# import tqdm
# for i in tqdm.tqdm(range(1000)):
#     time.sleep(0.001)
#
# from tqdm import trange
#
# for i in trange(1000):
#     time.sleep(0.001)

import numpy as np
a = np.arange(10).reshape(1, 10)
print(a)
print(a.shape)

b = a.squeeze(0)
print(b)