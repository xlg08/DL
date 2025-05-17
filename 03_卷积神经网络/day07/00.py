"""
案例:
    基于之前的知识点, 搭建 卷积神经网络, 进行: 图像识别.

卷积神经网络:
    1. 输入层
    2. 卷积层 + 激励层 + 池化层 ...
    3. 输出层
"""

# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

# 1. 准备数据集.
def create_dataset():
    # 1. 从 torchvision中获取 CIFAR10数据集 -> 训练集(5W张图片)
    # 参1: 数据集存放路径, 参2: 是否是训练集, 参3: 数据转换, 参4: 是否下载数据集(如果指定路径下没有, 则会自动下载)
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)

    # 2. 从 torchvision中获取 CIFAR10数据集 -> 测试集(1W张图片)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

    # 3. 返回结果.
    return train_dataset, test_dataset

# 2. 搭建卷积神经网络模型.
class ImageModel(nn.Module):
    # 1. 初始化父类成员, 搭建神经网络.
    def __init__(self):
        # 1.1 初始化父类成员
        super().__init__()

        # 1.2 搭建第1个 卷积层 + 池化层
        # 参1: 输入通道数, 参2: 输出通道数, 参3: 卷积核大小, 参4: 步长, 参5: 填充
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 1.3 搭建第2个 卷积层 + 池化层
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 1.4 搭建全连接层.
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    # 2. 前向传播(正向传播)
    def forward(self, x):
        # 2.1 第1个 卷积层 + 激励层 + 池化层.
        x = self.pool1(torch.relu(self.conv1(x)))
        # 2.2 第2个 卷积层 + 激励层 + 池化层.
        x = self.pool2(torch.relu(self.conv2(x)))

        # 2.3 因为全连接层只能处理2维的数据, 需要对x进行拉平(降维)处理.
        # 参1: 批次的行数(最小是2, 具体由传入的决定), 参2: 每行的列数(-1表示自动计算, 这里是: 576)
        x = x.reshape(x.size(0), -1)
        # print(f'x.shape: {x.shape}, x.size: {x.size(), x.size(0)}')

        # 2.4 第1个全连接层处理.
        x = torch.relu(self.linear1(x))
        # 2.5 第2个全连接层处理.
        x = torch.relu(self.linear2(x))

        # 2.6 输出层处理.
        x = self.output(x)  # 多分类问题, 损失函数用 CrossEntropyLoss -> softmax() + 损失计算.

        # 2.7 返回结果.
        return x

# 3. 模型训练.
def train_model(train_dataset):

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失 = softmax() + 损失计算.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    for epoch in range(epochs):

        total_loss, total_correct, total_sample, start_time = 0, 0, 0, time.time()

        for x, y in train_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(torch.argmax(y_pred, dim=-1))     # -1表示行.
            # print(f'真实值: {y} \n')

            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_sample += len(y)


        end_time = time.time()
        print(f'轮次: {epoch + 1}, 训练集准确率: {total_correct / total_sample:.4f}, 训练集损失: {total_loss / total_sample:.4f}, 训练时间: {end_time - start_time:.2f} s')

    torch.save(model.state_dict(), './model/image_model.pth')

# 4. 模型测试.




# 5. 测试代码.
if __name__ == '__main__':
    # 1. 获取数据集.
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集总数: {train_dataset.data.shape}')
    # print(f'测试集总数: {test_dataset.data.shape}')
    # # 查看各分类对应的索引, {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print(f'各分类对应的索引: {train_dataset.class_to_idx}')
    #
    # # 绘制某张图片.
    # plt.imshow(train_dataset.data[666])
    # plt.title(train_dataset.targets[666])
    # plt.show()

    # 2. 创建 神经网络模型对象.
    # model = ImageModel()
    # summary(model, (3, 32, 32), batch_size=1)

    # 3. 模型 训练.
    train_model(train_dataset)