'''

'''
import time

from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

# 解决 运行时错误：libiomp5md.dll
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BATCH_SIZE = 8


# 1. 数据集基本信息
def create_dataset():
    # 加载数据集:训练集数据和测试数据
    # ToTensor: 将image（一个PIL.Image对象）转换为一个Tensor
    train = CIFAR10(root='../../data', train=True, transform=ToTensor())
    valid = CIFAR10(root='../../data', train=False, transform=ToTensor())

    # 返回数据集结果
    return train, valid


# 模型构建
class ImageClassification(nn.Module):

    # 定义网络结构
    def __init__(self):
        super(ImageClassification, self).__init__()
        # 定义网络层：卷积层+池化层
        # 第一个卷积层, 输入图像为3通道,输出特征图为6通道,卷积核3*3
        self.conv1 = nn.Conv2d(3, 6, stride=1, kernel_size=3)
        # 第一个池化层, 核宽高2*2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层, 输入图像为6通道,输出特征图为16通道,卷积核3*3
        self.conv2 = nn.Conv2d(6, 16, stride=1, kernel_size=3)
        # 第二个池化层, 核宽高2*2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        # 第一个隐藏层 输入特征576个(一张图像为16*6*6), 输出特征120个
        self.linear1 = nn.Linear(576, 120)
        # 第二个隐藏层
        self.linear2 = nn.Linear(120, 84)
        # 输出层
        self.out = nn.Linear(84, 10)

    # 定义前向传播
    def forward(self, x):
        # 卷积+relu+池化
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        # 卷积+relu+池化
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # 将特征图做成以为向量的形式：相当于特征向量 全连接层只能接收二维数据集
        # 由于最后一个批次可能不够8，所以需要根据批次数量来改变形状
        # x[8, 16, 6, 6] --> [8, 576] -->8个样本,576个特征
        # x.size(0): 第1个值是样本数 行数
        # -1：第2个值由原始x剩余3个维度值相乘计算得到 列数(特征个数)
        x = x.reshape(x.size(0), -1)
        # 全连接层
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # 返回输出结果
        return self.out(x)


def train(model, train_dataset):
    # 构建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()  # 构建损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 构建优化方法

    # epoch = 100  # 训练轮数
    epoch = 10
    for epoch_idx in range(epoch):

        sum_num = 0  # 样本数量
        total_loss = 0.0  # 损失总和
        correct = 0  # 预测正确样本数
        start = time.time()  # 开始时间

        # 遍历数据进行网络训练
        for x, y in dataloader:
            model.train()
            output = model(x)
            loss = criterion(output, y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
            correct += (torch.argmax(output, dim=-1) == y).sum()  # 计算预测正确样本数
            # 计算每次训练模型的总损失值 loss是每批样本平均损失值
            total_loss += loss.item() * len(y)  # 统计损失和
            sum_num += len(y)

        print('epoch:%2s loss:%.5f acc:%.2f time:%.2fs' % (
        epoch_idx + 1, total_loss / sum_num, correct / sum_num, time.time() - start))

    # 模型保存
    torch.save(model.state_dict(), '../../model/image_classification.pth')


if __name__ == '__main__':
    # 数据集加载
    train_dataset, valid_dataset = create_dataset()

    # 数据集类别
    # print("数据集类别:", train_dataset.class_to_idx)

    # 数据集中的图像数据
    # print("训练集数据集:", train_dataset.data.shape)
    # print("测试集数据集:", valid_dataset.data.shape)

    # 图像展示
    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_dataset.data[1])
    # plt.title(train_dataset.targets[1])
    # plt.show()

    # 模型实例化
    model = ImageClassification()
    summary(model, input_size=(3, 32, 32), batch_size=1)

    # 数据集加载
    train_dataset, valid_dataset = create_dataset()
    # 模型实例化
    model = ImageClassification()
    # 模型训练
    train(model, train_dataset)
