import os
import pandas as pd

save_weight = 'weight/'
# os.path.isdir() 判断路径是否为目录
if not os.path.isdir(save_weight):
    # s.mkdir()创建单个目录
    os.mkdir(save_weight)

train_image_dir = f'E:/data/盐体识别/tgs-salt-identification-challenge/train/images'
train_mask_dir = f'E:/data/盐体识别/tgs-salt-identification-challenge/train/masks'
test_image_dir = f'E:/data/盐体识别/tgs-salt-identification-challenge/test/images'
depths = pd.read_csv('E:/data/盐体识别/tgs-salt-identification-challenge/depths.csv')


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

train_df = pd.read_csv('E:/data/盐体识别/tgs-salt-identification-challenge/train.csv')
print()
train_df = train_df.merge(depths)
print(train_df)