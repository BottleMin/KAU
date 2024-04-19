from google.colab import drive
drive.mount('/content/drive')

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor

# display images
from torchvision import utils
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

# utils
import numpy as np
from torchsummary import summary
from tqdm.notebook import tqdm
import os, random, glob
import pandas as pd

def make_csv_v1(url):
    root_dir = '/content/drive/MyDrive/프로젝트/딥러닝/final/Linnaeus_5_128X128/'
    data_annotation = os.path.join(root_dir,url)
    data_path = sorted(os.listdir(data_annotation))
    class_dir_A = glob.glob(f'{os.path.join(data_annotation,data_path[2])}/*.jpg')
    class_dir_B = glob.glob(f'{os.path.join(data_annotation,data_path[3])}/*.jpg')
    class_dir = class_dir_A + class_dir_B 

    label = []
    for i in range(len(class_dir_A)):
        label.append(0)
    for i in range(len(class_dir_B)):
        label.append(1)

    making_csv = {'filepath': class_dir, 'label': label}

    df = pd.DataFrame(data=making_csv)
    df = df.set_index("filepath")

    return df.to_csv(f'/content/drive/MyDrive/프로젝트/딥러닝/final/{url}_annotation.csv')

class DogsFlowersDataset(torch.utils.data.Dataset): 
    def __init__(self, annotation_path, root_dir = '/content/drive/MyDrive/프로젝트/딥러닝/final/'):
        self.data_annotation = pd.read_csv(os.path.join(root_dir,annotation_path))
        self.data_path = self.data_annotation['filepath']
        self.labels = self.data_annotation['label']        
        np.random.seed(42)
        self.index_B = np.random.randint(0,len(self.data_path), size=len(self.data_path))
        if annotation_path == 'train_annotation.csv':
            self.num = 4000
            self.transforms = Compose([ 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(9, 9), sigma=(1.0, 2.0)),
                ToTensor()
            ])
            
        else:
            self.num = 800
            self.transforms = Compose([ 
                ToTensor()
            ])

    def __len__(self):
        return len(self.data_path)

    def label_choosing(self, label1, label2):
        labelA = [0.,0.,0.,0.]
        labelB = [0.,0.,0.,0.]
        if (label1 == 0 and label2 == 0):
            labelA[0] = 1.
            labelB[1] = 1.
            return labelA, labelB
        elif (label1 == 0 and label2 == 1) or ((label1 == 1 and label2 == 0)):
            labelA[0] = 1.
            labelB[2] = 1.
            return labelA, labelB
        elif (label1 == 1 and label2 == 1):
            labelA[2] = 1.
            labelB[3] = 1.
            return labelA, labelB

    def synthesis(self, X1,X2, label1, label2):
        labelA = []
        labelB = []
        indice = torch.randperm(18)
        X_img1_list = [X1[:,0:40,0:40], X1[:,0:40,44:84], X1[:,0:40,88:128],X1[:,44:84,0:40], 
                        X1[:,44:84,44:84], X1[:,44:84,88:128], X1[:,88:128,0:40], X1[:,88:128,44:84], X1[:,88:128,88:128]]
        X_img2_list = [X2[:,0:40,0:40], X2[:,0:40,44:84], X2[:,0:40,88:128],X2[:,44:84,0:40], 
                        X2[:,44:84,44:84], X2[:,44:84,88:128], X2[:,88:128,0:40], X2[:,88:128,44:84], X2[:,88:128,88:128]]  
        X_img = torch.stack(X_img1_list + X_img2_list, dim=0)[indice]

        img = torch.zeros((3,120,240))
        img[:,0:40,:] = torch.cat([X_img[0,:,:,:],X_img[1,:,:,:],X_img[2,:,:,:],X_img[3,:,:,:],X_img[4,:,:,:],X_img[5,:,:,:]], dim=2) 
        img[:,40:80,:] = torch.cat([X_img[6,:,:,:],X_img[7,:,:,:],X_img[8,:,:,:],X_img[9,:,:,:],X_img[10,:,:,:],X_img[11,:,:,:]], dim=2) 
        img[:,80:120,:] = torch.cat([X_img[10,:,:,:],X_img[11,:,:,:],X_img[12,:,:,:],X_img[13,:,:,:],X_img[14,:,:,:],X_img[15,:,:,:]], dim=2) 

        labelA, labelB= self.label_choosing(label1, label2)
        
        sub_label1 = [labelA]*9
        sub_label2 = [labelB]*9

        sub_image = torch.stack(X_img1_list + X_img2_list, dim=0)
        sub_label = torch.tensor(sub_label1 + sub_label2)
        sub_image, sub_label = sub_image[indice], sub_label[indice]      
        return img, sub_label.float()

    def __getitem__(self,index):
        # all_image = []
        # all_label = []
        file_dir1 = self.data_path[index]
        file_dir2 = self.data_path[self.index_B[index]]
        input_image_1 = Image.open(file_dir1)
        input_image_2 = Image.open(file_dir2)
        image1 = self.transforms(input_image_1)
        image2 = self.transforms(input_image_2)
        label1 = self.labels[index]
        label2 = self.labels[self.index_B[index]]

        sub_image, sub_label = self.synthesis(image1, image2, label1, label2)
        # all_image.append(sub_image)
        # all_label.append(sub_label) 
        # torch.stack(all_image,dim=0), torch.stack(all_label,dim=0)
        
        return sub_image, sub_label

trainset = DogsFlowersDataset(annotation_path = 'train_annotation.csv')
testset = DogsFlowersDataset(annotation_path = 'test_annotation.csv')

print(len(trainset))
num = 50
image, label = trainset[2399]
print(label)

f = [[1,2,3],[4,5,6],[7,8,9]]
class_name = ['Dog', 'Flower']

plt.imshow(image.permute(1,2,0).numpy())
plt.axis('off')

import torch
from torch import Tensor
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(LayerNorm, self).__init__()
        ln = nn.GroupNorm(1, in_channel, eps=1e-08)
        ...
    def forward(self,x):
        out = ln(x)
        return out
        
class BasicBlock(nn.Module):
    expansion_factor = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU()
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion_factor))
    
    def forward(self, x: Tensor) -> Tensor:
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += self.residual(out)
        x = self.relu2(x)
        return x


class BottleNeck(nn.Module):
    expansion_factor = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion_factor, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion_factor)
        
        self.relu3 = nn.ReLU()
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion_factor))
        
    def forward(self, x:Tensor) -> Tensor:
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += self.residual(out)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * block.expansion_factor, num_classes)

        self._init_layer()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion_factor
        return nn.Sequential(*layers)

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x


class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def resnet34(self):
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def resnet50(self):
        return ResNet(BottleNeck, [3, 4, 6, 3])

    def resnet101(self):
        return ResNet(BottleNeck, [3, 4, 23, 3])

    def resnet152(self):
        return ResNet(BottleNeck, [3, 8, 36, 3])

  def divide_img(X):
    train_loader_stack = [X[:,:,0:40,0:40], X[:,:,0:40,40:80], X[:,:,0:40,80:120],
                          X[:,:,0:40,120:160], X[:,:,0:40,160:200], X[:,:,0:40,200:240],
                          X[:,:,40:80,0:40], X[:,:,40:80,40:80], X[:,:,40:80,80:120], 
                          X[:,:,40:80,120:160], X[:,:,40:80,160:200], X[:,:,40:80,200:240], 
                          X[:,:,80:120,0:40], X[:,:,80:120,40:80], X[:,:,80:120,80:120],
                          X[:,:,80:120,120:160], X[:,:,80:120,160:200], X[:,:,80:120,200:240]]
    return torch.stack(train_loader_stack, dim=1)

def train(model, optimizer, train_loader, epoch):
    train_loss = 0
    train_accuracy = 0
    train_count = len(trainset)
    model.train()
    for i, (X,y) in enumerate(train_loader):
        reiteraton = 0
        X_stack = divide_img(X)
        for j in range(18):
            X_cell = X_stack[:,j,:,:,:]
            X_cell = X_cell.to(device)
            y_cell = y[:,j,:]
            y_cell = y_cell.to(device)

            optimizer.zero_grad()
            output = model(X_cell)
            loss = criterion(output, y_cell)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu() * X_cell.shape[0]
            prediction = torch.max(output,1)[1]
            true = torch.max(y_cell,1)[1]
            train_accuracy += int(torch.sum(prediction == true))

    train_accuracy = train_accuracy/train_count
    train_loss = train_loss/train_count
    return train_accuracy, train_loss

def test(model, optimizer, test_loader, epoch):
    test_loss = 0
    test_accuracy = 0
    test_count = len(testset)
    model.eval()
    for i, (X,y) in enumerate(test_loader):
        with torch.no_grad():
            X_stack = divide_img(X)
            for j in range(18):
                X_cell = X_stack[:,j,:,:,:]
                X_cell = X_cell.to(device)
                y_cell = y[:,j,:]
                y_cell = y_cell.to(device)
                output = model(X_cell)
                loss = criterion(output, y_cell)

                test_loss += loss.cpu() * X_cell.shape[0]
                prediction = torch.max(output,1)[1]
                true = torch.max(y_cell,1)[1]
                test_accuracy += int(torch.sum(prediction == true))

    test_accuracy = test_accuracy/test_count
    test_loss = test_loss/test_count
    return test_accuracy, test_loss

## parameters
epoch_num = 5
batchsize = 32
lr = 0.001 # learning rate

## dataloader
train_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batchsize,
                                          shuffle=True,
                                          num_workers=3,
                                          drop_last=True)

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=batchsize,
                                          shuffle=True,
                                          num_workers=3,
                                          drop_last=True)
## make model and using GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().resnet101()
print("device : ", device)
model_ResNet = model.to(device)

## optimizer setting
optimizer = torch.optim.Adam(model_ResNet.parameters(), ## Adam optimizer
                            lr=lr) 

## loss function
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()

train_resnet_loss, train_resnet_accuracy = [], []
test_resnet_loss, test_resnet_accuracy = [], []

for i in range(epoch_num):
    avg_train_loss, avg_train_accuracy = train(model_ResNet, optimizer, train_loader, i)
    train_resnet_loss.append(avg_train_loss)
    train_resnet_accuracy.append(avg_train_accuracy)

    avg_test_loss, avg_test_accuracy = test(model_ResNet, optimizer, test_loader, i)
    test_resnet_loss.append(avg_test_loss)
    test_resnet_accuracy.append(avg_test_accuracy)

    print(f'epoch {i+1}) train loss : {avg_train_loss:.4f} / train_accuracy : {avg_train_accuracy:.4f} \
    / test loss : {avg_test_loss:.4f} / test_accuracy : {avg_test_accuracy:.4f}')

def test_predict(model, test_loader):
    model.eval()
    predict_label = []
    X_plot = []
    y_label = []
    for i, (X,y) in enumerate(test_loader):
        with torch.no_grad():
            X_stack = divide_img(X)
            for j in range(batchsize):
                X_cell = X_stack[j,:,:,:,:]
                X_cell = X_cell.to(device)
                y_cell = y[j,:,:]
                y_cell = y_cell.to(device)
                predict = model(X_cell)
                X_plot.append(X_cell.cpu())
                y_label.append(y_cell.cpu())
                predict_label.append(predict.cpu())
    return X_plot, y_label, predict_label

class classisification:
    def __init__(self, img_class, y_label, predict_label, num):
        self.class_name = ['Dog', 'Flower']

        self.X_test = img_class[num]
        self.true_label = y_label[num].flatten().tolist()
        self.test_label = np.round(predict_label[num].flatten().tolist())

        self.dog_label = []
        self.dog_image = []
        self.flower_label = []
        self.flower_image = []

        for i in range(18):
            if self.class_name[int(self.test_label[i])] == 'Dog':
                self.dog_label.append(self.true_label[i]) # 참 값 (Dog 아닐수도)
                self.dog_image.append(self.X_test[i]) # 추정 값 (Dog)
            else:
                self.flower_label.append(self.true_label[i]) # 참 값 (Flower 아닐수도)
                self.flower_image.append(self.X_test[i]) # 추정 값 (Flower)
        print(f'dog class:{len(self.dog_image)}, flower class:{len(self.flower_image)}')
        self.dog_image = torch.stack(self.dog_image, dim=0)
        self.flower_image = torch.stack(self.flower_image, dim=0)

    def row_columns(self, img_class):
        rows = img_class.shape[0] // 3
        columns = rows + (img_class.shape[0] % 3)

        while (rows * columns) < img_class.shape[0]:
            columns += 1
        return rows, columns
    
    def plot_for_variable(self, object_name, label, rows, columns, name):
        plt.suptitle(f'class: {name}', fontsize = 15)
        for i in range(object_name.shape[0]):
            ax1 = plt.subplot(rows,columns,i+1)
            plt.imshow(object_name[i].permute(1,2,0).numpy())
            plt.title(self.class_name[int(label[i])])
            plt.axis('off')
        plt.tight_layout()
        
    def plot_dog(self):
        row_class0, columns_class0 = self.row_columns(self.dog_image)

        plt.figure(1)
        dog = self.plot_for_variable(self.dog_image, self.dog_label, row_class0, columns_class0, self.class_name[0])
    
    def plot_flower(self):
        row_class1, columns_class1 = self.row_columns(self.flower_image)

        plt.figure(2)
        flower = self.plot_for_variable(self.flower_image, self.flower_label, row_class1, columns_class1, self.class_name[1])
        

X_plot, y_label, predict_label = test_predict(model_ResNet, test_loader)

num = np.random.randint(0,len(y_label))
# num = 300
print(f'length: {len(y_label)}, random number: {num}')
print(np.round(predict_label[num].flatten().tolist(),2))
inference = classisification(X_plot, y_label, predict_label, num)

inference.plot_dog()
inference.plot_flower()
