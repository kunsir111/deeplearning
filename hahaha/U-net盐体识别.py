import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
import torch.nn as nn
import torchvision

save_weight = 'weight/'
# os.path.isdir() 判断路径是否为目录
if not os.path.isdir(save_weight):
    # s.mkdir()创建单个目录
    os.mkdir(save_weight)

train_image_dir = f'E:/data/Salt body recognition/tgs-salt-identification-challenge/train/images'
train_mask_dir = f'E:/data/Salt body recognition/tgs-salt-identification-challenge/train/masks'
test_image_dir = f'E:/data/Salt body recognition/tgs-salt-identification-challenge/test/images'

# pd.read_csv()读取的文件的数据格式，例如：
# df_1 = pd.DataFrame({'姓名': ["小明","小红","小刚"],
#                    '年纪': [10,9,12],
#                    '城市': ['上海','北京','深圳']})
depths = pd.read_csv('E:/data/Salt body recognition/tgs-salt-identification-challenge/depths.csv')


# inplace功能：就地执行排序，不用重新创建一个新的元素(DataFrame) 相当于 depths = depths.sort_values('z')
depths.sort_values('z', inplace=True)
# print(depths)

# labels是指要删除的标签，一个或者是列表形式的多个，axis是指处哪一个轴0是行，1是列
depths.drop('z', axis=1, inplace=True)
# print(depths)

# depths.shape[0]返回的是depths行数，有几行
# depths.shape[1]返回的是depths列数，有几列
# list(range(0, 5)) 生成0-4的列表
# ????? 为什么生成‘fold’列
depths['fold'] = (list(range(0, 5)) * depths.shape[0])[:depths.shape[0]]
# print(depths)

train_df = pd.read_csv('E:/data/Salt body recognition/tgs-salt-identification-challenge/train.csv')
# print(train_df['id'])

# merge是pandas中用来合并数据的函数，不像concat是按照某行或某列来合并，而是按照数据中具体的某一字段来连接数据。
# pd.merge(DateFrame1,DateFrame2,on = ' ',how = ' ')
# 或 DateFrame1.merge(DateFrame2,on = ' ',how = ' ')
# on表示按照哪个特征来找相同的字段，没有on的话，就自动找相同的字段
# how是指两个DateFrame的拼接方式。
#
# how = ‘outer’:外置，相当于两个DateFrame求并集
# how = ‘right’: 右置，合并后，按照最右边不为空的样本显示
# how = ‘left’：左置，合并后，按照最左边不为空的样本显示
# how = ‘inner’：只显示匹配到的字段的样本
#
# 例子：
# df_1 = pd.DataFrame({'姓名': ["小明","小红","小刚"],
#                    '年纪': [10,9,12],
#                    '城市': ['上海','北京','深圳']})
# df_2 = pd.DataFrame({'零花钱': [50,200,600,400,80],
#                    '城市': ['苏州','北京','上海','广州','重庆']})
# result = pd.merge(df_1,df_2, on = '城市', how = 'outer')

# 自动找到train.csv和depths.csv中的相同字段合并
train_df = train_df.merge(depths)
# print(train_df)

dist = []

# print(train_df.id.values)
for id in train_df.id.values:
    # print(id)

    # 读入图像：使用函数cv2.imread(filepath,flags)读入一副图片
    # filepath：要读入图片的完整路径
    # flags：读入图片的标志
    #       cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
    #       cv2.IMREAD_GRAYSCALE：读入灰度图片
    #       cv2.IMREAD_UNCHANGED：读入完整图片，包括alpha通道
    # 注意：openCV中的文件路径不能有中文
    img = cv2.imread(f'E:/data/Salt body recognition/tgs-salt-identification-challenge/train/images/{id}.png', cv2.IMREAD_GRAYSCALE)

    # np.unique() 返回的是一个无元素重复的数组或列表
    dist.append(np.unique(img).shape[0])

train_df['unique_pixels'] = dist

# 显示所有列
# pd.set_option('display.max_columns', None)

# 显示所有行
# pd.set_option('display.max_rows', None)
# print(train_df)


# 更改图片的
def do_resize2(image, mask, W, H):
    # 在图像处理过程中，有时需要把图像调整到同样大小，便于处理，这时需要用到图像resize()
    image = cv2.resize(image, dsize=(W, H))
    mask = cv2.resize(mask, dsize=(W, H))
    return image, mask

def do_center_pad(image, pad_left, pad_right):
    return np.pad(image, (pad_left, pad_right), 'edge')

def do_center_pad2(image, mask, pad_left, pad_right):
    image = do_center_pad(image, pad_left, pad_right)
    mask = do_center_pad(mask, pad_left, pad_right)
    return image, mask

# 无论是官方给出的数据集如torchvision.datasets.MNIST等，还是我们在做实验时需要
# 使用自己的数据集，都要继承Dataset类，在继承过程中，须重载的函数包括：
class SaltDataset(Dataset):
    # 构造函数
    # ??? 为什么要把图片大小改为202 * 202
    def __init__(self, image_list, mode, mask_list, fine_size=202, pad_left=0, pad_right=0):
        self.imageList = image_list
        self.mode = mode
        self.maskList = mask_list
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right

    # sampler（如SequentialSampler()类）中有调用len()函数：
    def __len__(self):
        return len(self.imageList)

    # _DataLoaderIter()类中有调用：
    def __getitem__(self, idx):
        # 深复制：即将被复制对象完全再复制一遍作为独立的新个体单独存在。所以改变原有被复
        # 制对象不会对已经复制出来的新对象产生影响。
        image = copy.deepcopy(self.imageList(idx))
        if self.mode == 'train':
            mask = copy.deepcopy(self.maskList[idx])

            # numpy.where调用方式为numpy.where(condition,1,2)
            # 满足条件的位置上返回结果1，不满足的位置上返回结果2

            label = np.where(mask.sum() == 0, 1.0, 0.0).astype(np.float32)
            if self.fine_size != image.shape[0]:
                #
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                # 边缘填充
                image, mask = do_center_pad2(image, mask, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            return image, mask, label

        if self.mode == 'val':
            mask = copy.deepcopy(self.maskList[idx])

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask

            return image, mask

        if self.mode == 'test':

            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)
            return image

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class Unet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2]
        )
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder()
        self.decode3 = Decoder()
        self.decode2 = Decoder()
        self.decode1 = Decoder()
        self.decode0 = nn.Sequential(
            nn.Upsample(),
            nn.Conv2d(),
            nn.Conv2d()
        )
        self,conv_last = nn.Conv2d()

    def forward(self, input):
        pass