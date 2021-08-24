import torch
import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

n_fold = 5
pad_left = 27
pad_right = 27
fine_size = 202
batch_size = 18
epoch = 25
snapshot = 6
max_lr = 0.012
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
n_fold = 5
device = torch.device('cuda')
save_weight = './weights/'

if not os.path.isdir(save_weight):
    os.mkdir(save_weight)
weight_name = 'model_' + str(fine_size + pad_left + pad_right) + '_res18'

train_image_dir = './tgs-salt-identification-challenge/train/images'
train_mask_dir = './tgs-salt-identification-challenge/train/masks'
test_image_dir = './tgs-salt-identification-challenge/test/images'

depths = pd.read_csv('./tgs-salt-identification-challenge/depths.csv')
# print(depths)

depths.sort_values('z', inplace=True)
depths.drop('z', axis=1, inplace=True)
depths['fold'] = (list(range(0, 5)) * depths.shape[0])[:depths.shape[0]]
# print(depths)
train_df = pd.read_csv('./tgs-salt-identification-challenge/train.csv')
# print(train_df)

# 按照train_df中的第一个键值合并train_df和depth
train_df = train_df.merge(depths)
# print(train_df)

dist = []
# print(train_df.id.values)

# cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
# cv2.IMREAD_GRAYSCALE：读入灰度图片
# cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
for id in train_df.id.values:
    img = cv2.imread(os.path.join(train_image_dir, id + '.png') , cv2.IMREAD_GRAYSCALE)
    # print(img.shape)
    # print(np.unique(img).shape[0])
    dist.append(np.unique(img).shape[0])
    # break
train_df['unique_pixels'] = dist

# print(train_df)

test_image_id = './tgs-salt-identification-challenge/train/images/0a7e067255.png'

# 读取训练集图片并归一化
def trainImageFetch(image_id):
    image_train = np.zeros((image_id.shape[0], 101, 101), dtype = np.float32)
    mask_train = np.zeros((image_id.shape[0], 101, 101), dtype = np.float32)

    for idx, image_id in tqdm(enumerate(image_id), total = image_id.shape[0]):
        image_path = os.path.join(train_image_dir, image_id + '.png')
        mask_path = os.path.join(train_mask_dir, image_id + '.png')

        # 将像素值归一化
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

        image_train[idx] = image
        mask_train[idx] = mask
    # 返回图像大小为101 * 101
    return image_train, mask_train

# train_data  = trainImageFetch(train_df['id'])

def testImageFetch(test_id):
    image_test = np.zeros((len(test_id), 101, 101), dtype=np.float32)
    for idx, image_id in tqdm(enumerate(test_id), total=len(test_id)):
        image_path = os.path.join(test_image_dir, image_id + '.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        image_test[idx] = image

    # 返回测试集的图片的大小为101 * 101
    return image_test

def do_resize2(image, mask, H, W):
    image = cv2.resize(image, dsize=(W, H))
    mask = cv2.resize(mask, dsize=(W, H))
    return image, mask

def do_center_pad(image, pad_left, pad_right):
    return np.pad(image, (pad_left, pad_right), 'edge')

def do_center_pad2(image, mask, pad_left, pad_right):
    image = do_center_pad(image, pad_left, pad_right)
    mask = do_center_pad(mask, pad_left, pad_right)
    return image, mask

class SaltDataset(Dataset):
    def __init__(self, image_list, mode, mask_list = None, fine_size = 202, pad_left = 0, pad_right = 0):
        self.imagelist = image_list
        self.mode = mode
        self.masklist = mask_list
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            # mask代表背景掩码，白色（1）代表有盐体，黑色（0 ）代表没有
            #
            # 原始图像大小为101 * 101， 转换为fine_size，变成202 * 202
            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            # 图像转换为fine_size后，对边缘进行填充，填充后图片大小为256 * 256
            if self.pad_left != 0:
                image, mask = do_center_pad2(image, mask, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            # 返回 1 * 256 * 256
            return image, mask


        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            return image, mask

        elif self.mode == 'test':
            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])

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
        super(Unet, self).__init__()
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7 ,7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2]
        )
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 256 + 128, 256)
        self.decode2 = Decoder(256, 128 + 64, 128)
        self.decode1 = Decoder(128, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        f = self.layer5(e4)
        # print(f.shape, e4.shape)
        d4 = self.decode4(f, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        d1 = self.decode1(d2, e1)
        d0 = self.decode0(d1)
        out = self.conv_last(d0)
        # print(out.shape)
        return out

net = Unet
print(net(1))

def train(train_loader, model):
    running_loss = 0.0
    data_size = len(train_data)
    # 在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out。
    model.train()

    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):

            # inputs维度 batch_size * 1 * 256 * 256
            #print(inputs.shape)
            logit = model(inputs)
            #print(logit.shape)
            # squeeze 参数:axis: 选择数组中的某一维度移除
            loss = nn.BCEWithLogitsLoss()(logit.squeeze(1), masks.squeeze(1))

            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.shape[0]

    epoch_loss = running_loss / data_size
    return epoch_loss

def test(test_loader, model):
    running_loss = 0.0
    data_size = len(val_data)
    predicts = []
    truths = []

    model.eval()

    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = outputs[:, :, pad_left:pad_left +fine_size, pad_left:pad_left +fine_size]
            loss = nn.BCEWithLogitsLoss()(outputs.squeeze(1), masks.squeeze(1))

        predicts.append(torch.sigmoid(outputs).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)
    predicts = np.concatenate(predicts).squeeze()

    truths = np.concatenate(truths).squeeze()
    # print(predicts.shape, truths.shape)
    precision = do_kaggle_metric(predicts, truths, 0.5)
    precision = precision.mean()

    epoch_loss = running_loss / data_size
    return epoch_loss, precision

def do_kaggle_metric(predicts, truths, k):
    acc = []
    for x, y in zip(predicts, truths):
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        acc.append((x == y).sum() / (202 * 202))
    acc = np.array(acc)
    return acc

def rle_encode(im):
    # flatten()函数用法
    # flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
    # flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用！。
    # a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降 。
    # a.flatten().A：a是个矩阵，降维后还是个矩阵，矩阵.A（等效于矩阵.getA()）变成了数组。具体看下面的例子：
    # >>> from numpy import *
    # >>> a=array([[1,2],[3,4],[5,6]])
    # >>> a
    # array([[1, 2],
    #     [3, 4],
    #     [5, 6]])
    # >>> a.flatten() #默认按行的方向降维
    # array([1, 2, 3, 4, 5, 6])
    # >>> a.flatten('F') #按列降维
    # array([1, 3, 5, 2, 4, 6])
    # >>> a.flatten('A') #按行降维
    # array([1, 2, 3, 4, 5, 6])
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # print(runs)
    runs[1::2] -= runs[::2]
    s = ' '.join(str(x) for x in runs)
    # print(s)
    return  ' '.join(str(x) for x in runs)


all_id = train_df['id'].values
fold = []
for i in range(5):
    fold.append(train_df.loc[train_df['fold'] == i, 'id'].values)
# print(fold)

salt = Unet(1)
salt.to(device)

for idx in range(5):
    if idx == 1:
        break
    scheduler_step =epoch

    optimizer = torch.optim.SGD(salt.parameters(), lr = max_lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, min_lr)

    # setdiff1d(x, y)这个函数返回两个值：
    # 第一个是：在x中出现，但是在y中没有出现的的元素
    # 第二个是：这些元素在x中的索引（也就是下标或者位置）
    train_id = np.setdiff1d(all_id, fold[idx])
    val_id = fold[idx]

    X_train, y_train = trainImageFetch(train_id) # 图像大小为101*101
    X_val, y_val = trainImageFetch(val_id)

    train_data = SaltDataset(X_train, 'train', y_train, pad_left = 27, pad_right = 27) # 图像大小为
    val_data = SaltDataset(X_val, 'val', y_val, pad_left = 27, pad_right = 27)

    train_loader = DataLoader(train_data, shuffle=RandomSampler(train_data), batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    num_snapshot = 1
    best_acc = 0

    for epoch_ in range(epoch):
        train_loss = train(train_loader, salt)
        val_loss, accuracy = test(val_loader, salt)
        lr_scheduler.step()

        if accuracy > best_acc:
            best_acc = accuracy
            best_param = salt.state_dict()

        if (epoch_ + 1) % scheduler_step == 0:
            # print(save_weight + weight_name + str(idx) + str(num_snapshot) + '.ph')
            torch.save(best_param, './weights/' + weight_name + str(idx) + str(num_snapshot) + '.ph')
            optimizer = torch.optim.SGD(salt.parameters(), lr = max_lr, momentum=momentum, weight_decay=weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, min_lr)
            num_snapshot += 1
            best_acc = 0

        print('epoch: {} train_loss: {:.3f} val_loss: {:.3f} val_accuracy: {:.3f}'.format(epoch_ + 1, train_loss, val_loss, accuracy))

test_id  = [x[:-4] for x in os.listdir(test_image_dir) if x[-4:] == '.png']
image_test = testImageFetch(test_id)
overall_pred_101 = np.zeros((len(test_id), 101, 101), dtype=np.float32)

for step in range(1, 6):
    if(step == 2):
        break
    print('Predicting Snapshot', step)
    pred_null = []
    # print('./weights/' +weight_name + '0' + str(step) + '.ph')
    param = torch.load(save_weight +weight_name + '0' + str(step) + '.ph')
    salt.load_state_dict(param)

    test_data = SaltDataset(image_test, mode='test', fine_size=fine_size, pad_left=pad_left, pad_right = pad_right)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    salt.eval()
    for images in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        with torch.set_grad_enabled(False):
            pred = salt(images)
            pred = torch.sigmoid(pred).squeeze(1).cpu().numpy()
            pred = pred[:, pad_left : pad_left + fine_size, pad_left: pad_left + fine_size]
            # print(pred)
            pred_null.append(pred)

    idx = 0
    for i  in range(len(pred_null)):
        for j in range(batch_size):
            overall_pred_101[idx] += cv2.resize(pred_null[i][j], dsize = (101, 101))
            idx += 1

submission = pd.DataFrame({'id': test_id, 'rle_mask' : list(overall_pred_101)})
# print(submission)

# lambda表达式 分号前边代表变量，后边代表式子
submission['rle_mask'] = submission['rle_mask'].map(lambda x : rle_encode(x > 0.5 * 0.5))
# print(submission)

submission.set_index('id', inplace=True)

sample_submission = pd.read_csv('./tgs-salt-identification-challenge/sample_submission.csv')
# print(sample_submission)

# set_index( ) 将 DataFrame 中的列转化为行索引。
sample_submission.set_index('id', inplace=True)
# print(sample_submission)

# reindex()是pandas对象的一个重要方法，其作用是创建一个新索引的新对象。
submission = submission.reindex(sample_submission.index)
# print(submission)

# 数据清洗时，会将带空值的行删除，此时DataFrame或Series类型的数据不再是连续的索引，可以使用reset_index()重置索引。
submission.reset_index(inplace=True)
# print(submission)

submission.to_csv('submission.csv', index = False)
