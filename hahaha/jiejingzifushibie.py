import os, sys, glob, shutil, json
os.environ['DUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as Data
import torchvision.models as models
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
train_path = glob.glob('./data/jiejingzifushibie/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('./data/jiejingzifushibie/mchar_train.json'))
train_label = [train_json[x]['top'] for x in train_json]
print(len(train_path), len(train_label))
print(type(Data.Dataset))
use_cuda = False
class SVHNDataset(Data.Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))
    def __len__(self):
        return len(self.img_path)
# class SVHNDataset(Data.dataset):
#     def __init__(self, img_path, img_label, transform=None):
#         self.img_path = img_path
#         self.img_label = img_label
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = None
#     def __getitem__(self, index):
#         img = Image.open(self.img_path[index]).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         lbl = np.array(self.img_label[index], dtype=np.int)
#         lbl = list(lbl) + (5 - len(lbl)) * [10]
#         return img, torch.from_numpy(np.array(lbl[:5]))
#     def __len__(self):
#         return len(self.img_path)

train_loader = Data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])),
    batch_size=10,
    shuffle=True,
    num_workers=0,
)

val_path = glob.glob('./data/jiejingzifushibie/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('./data/jiejingzifushibie/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]
print(len(val_path), len(val_label))
val_loader = Data.DataLoader(SVHNDataset(val_path, val_label,
        transforms.Compose([
            transforms.Resize((60, 120)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])),
    batch_size=40,
    shuffle=True,
    num_workers=2,
)

class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc1(feat)
        c4 = self.fc2(feat)
        c5 = self.fc1(feat)
        return c1, c2, c3, c4, c5
train_loss = []
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = []
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        c0, c1, c2, c3, c4 = model(input)
        print(c0.size())
        print(target[:, 0].size())
        loss = criterion(c0, target[:, 0].long()) + criterion(c1, target[:, 1].long()) + criterion(c2, target[:, 2].long()) + criterion(c3, target[:, 3].long()) + criterion(c4, target[:, 4].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(loss.item())
        train_loss.append(loss.item())
    return np.mean(train_loss)
def validate(val_loader, model, criterion):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            c0,c1,c2,c3,c4 = model(input)
            loss = criterion(c0, target[:,0]) + \
                    criterion(c1, target[:,1]) + \
                    criterion(c2, target[:,2]) + \
                    criterion(c3, target[:,3]) + \
                    criterion(c4, target[:,4])
            val_loss.append(loss.item())
    return np.mean(val_loss)

def predict(test_loader, model, tta = 10):
    model.eval()
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()
                c0, c1, c2, c3, c4 = model(input)
                output = np.concatenate([
                    c0.data.numpy(),
                    c1.data.numpy(),
                    c2.data.numpy(),
                    c3.data.numpy(),
                    c4.data.numpy()], axis=1)
                test_pred.append(output)
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    return test_pred_tta

model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_loss = 1000.0

if use_cuda:
    model = model.cuda()
for epoch in range(2):
    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion)

    val_label = [''.join(map(str,x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, 11].argmax(dim=1),
        val_predict_label[:, 11:22].argmax(dim=1),
        val_predict_label[:, 22:33].argmax(dim=1),
        val_predict_label[:, 33:44].argmax(dim=1),
        val_predict_label[:, 44:55].argmax(dim=1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x!=10])))
    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))


