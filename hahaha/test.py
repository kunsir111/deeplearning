import cv2
import numpy as np

# img = cv2.imread("E:/data/Salt body recognition/tgs-salt-identification-challenge/train/images/000e218f21.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, dsize=(202, 202))
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# a = [1, 2, 3, 4, 5]
# print(np.pad(a, (1, 2), 'edge'))

import torchvision

base_model = torchvision.models.resnet18(pretrained=True)
print(list(base_model.children())[7])
