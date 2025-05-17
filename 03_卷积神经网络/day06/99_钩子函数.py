"""
案例:
    基于ANN(人工神经网络)的内容, 完成: 手机价格分类预测案例.

背景:
    基于手机的20个特征列, 来预测手机的 价格区间, 我们预测手机的具体价格, 而是属于哪个价格区间, 所以还是: 分类问题(4个区间)

ANN案例的 操作步骤:
    1. 准备数据集.
    2. 搭建神经网络 模型对象.
    3. 模型训练.
    4. 模型评估(测试)
    5. 模型优化.
"""

# 导包
import torch                                    # PyTorch框架, 用于 深度学习计算相关
from torch.utils.data import TensorDataset      # 数据集对象 -> 用于封装 数据加载器
from torch.utils.data import DataLoader         # 数据加载器 -> 可以分批次获取数据, Iteration
import torch.nn as nn                           # 模型对象 -> 用于搭建模型
import torch.optim as optim                     # 优化器 -> 用于优化模型参数
from sklearn.model_selection import train_test_split    # 数据集分割 -> 用于分割训练集和测试集
import matplotlib.pyplot as plt                 # 绘图
import numpy as np                              # 数组处理
import pandas as pd                             # 数据处理
import time                                     # 时间处理
from torchsummary import summary                # 模型结构可视化


# 1. 准备数据集.
def create_dataset():
    # 1. 读取csv文件, 获取df对象.
    data = pd.read_csv('../../data/手机价格预测.csv')
    # print(f'data: {data.head()}')
    # print(f'data.shape: {data.shape}')      # (2000, 21)

    # 2. 提取特征 和 标签.
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(f'x: {x.shape}, y: {y.shape}')    # (2000, 20), (2000,)
    # print(f'x: {x.head()}')
    # print(f'y: {y.head()}')

    # 3. 把x特征列, 转成: float类型.
    x = x.astype(np.float32)
    # print(f'x: {x.head()}')
    # print(f'x: {x.shape}, {x.shape[1]}')    # (2000, 20), 20(特征列数)
    # print(f'y: {np.unique(y)}')             # [0 1 2 3]

    # 4. 数据分割, 按照 8:2的比例, 分割训练集和测试集.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24, stratify=y)

    # 5. 把上述的 训练集 和 测试集分别封装成: 数据集对象DataSet对象.
    # 思路: 数据集 -> 张量 -> 数据集对象(DataSet)
    train_dataset = TensorDataset(torch.Tensor(x_train.values), torch.Tensor(y_train.values))
    test_dataset = TensorDataset(torch.Tensor(x_test.values), torch.Tensor(y_test.values))

    # 6. 返回结果: 训练集数据集对象, 测试集数据集对象, 特征数(充当 输入层的输入维度), 标签数(充当 输出层的输出维度).
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# 2. 搭建神经网络 模型对象.
class PhonePriceModel(nn.Module):
    # 1. 初始化方法: 初始化父类成员, 搭建神经网络.
    def __init__(self, input_dim, output_dim):
        # 1.1 初始化父类成员.
        super().__init__()

        # 1.2 搭建神经网络.
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)

    # 2. 前向(正向)传播, 输入层 -> 隐藏层 -> 输出层.
    def forward(self, x):
        # 2.1 隐藏层1: 加权求和 + 激活函数(relu)
        # x = self.linear1(x)     # 加权求和
        # x = torch.relu(x)       # 激活函数
        x = torch.relu(self.linear1(x))

        # 2.2 隐藏层2: 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))

        # 2.3 输出层: 加权求和 + 激活函数(softmax)
        # x = torch.softmax(self.output(x), dim=1)
        # 因为我们处理的是多分类的问题, 损失函数用的是: CrossEntropyLoss = softmax() + 损失计算, 所以这里不需要再使用softmax().
        # 细节: 训练时有损失函数, 可以不用softmax(),  但是测试时, 预测时, 需要使用softmax().
        x = self.output(x)

        # 2.4 返回结果.
        return x

def first(a,b,c):
    print(f"参数a: {a}")
    print(f"参数b: {b}")
    print(f"参数c: {c}")
    print(f"前向计算开始！")


# 3. 模型训练.
def train_model(train_dataset, input_dim, output_dim):
    # 1. 创建 数据加载器, 可以分批次获取数据.
    # 参1: 数据集对象, 参2: 批次大小, 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 2. 创建模型对象.
    model = PhonePriceModel(input_dim, output_dim)

    # 将first方法注册为前向后钩子函数，用于在模型前向传播计算完后时执行。
    model.register_forward_hook(first)

    # 3. 创建损失函数, 因为是多分类问题, 所以损失函数用的是: CrossEntropyLoss(多分类交叉熵损失) = softmax() + 损失计算.
    criterion = nn.CrossEntropyLoss()

    # 4. 创建优化器对象.
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))


    # 5. 定义变量, 记录: 训练的总轮数.
    epochs = 200

    # 6. 具体的训练过程.
    for epoch in range(epochs):

        # 6.1 具体的每轮训练动作, 记录: 本轮的总损失(各批次平均损失 和), 本轮的批次数, 本轮开始训练的时间.
        total_loss, batch_num, start = 0.0, 0, time.time()

        # 6.2 本轮, 各批次训练过程. 即: 从 数据加载器中, 取出数据, 进行训练.
        for x, y in train_loader:   # 数据是按批次获取的, 16条/批
            # 6.2.1 切换模式为: 训练模式.
            model.train()
            # 6.2.2 模型预测.
            y_pred = model(x)
            # 6.2.3 计算损失.
            loss = criterion(y_pred, y.long())     # loss = 本批次的 平均损失.
            # 6.2.4 梯度清零 + 反向传播 + 更新参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 6.2.5 记录: 本批次的损失 和 批次数.
            total_loss += loss.item()       # total_loss = 本轮第1批的平均损失 + 本轮第2批的平均损失 + ... + 本轮第n批的平均损失.
            batch_num += 1                  # batch_num = 1批 + 1批 + 1批 + ... + 1批

        # 6.3 走到这里, 本轮训练结束, 打印训练结果.
        print(f'epoch: {epoch + 1}, 本轮的平均损失: {total_loss / batch_num:.4f}, 耗时: {time.time() - start:.4f} s')

    # 7. 模型保存.
    # 参1: 模型对象(的参数), 参2: 保存模型的文件名.
    torch.save(model.state_dict(), './model/手机价格预测.pth')    # model文件夹必须存在.
    # print(model.state_dict())


# 4. 模型评估(测试)
def evaluate_model(test_dataset, input_dim, output_dim):
    # 1. 创建 数据加载器, 可以分批次获取数据.
    # 参1: 测试集数据集对象, 参2: 批次大小, 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 2. 创建模型对象.
    model = PhonePriceModel(input_dim, output_dim)

    # 3. 加载(训练好的)模型参数.
    model.load_state_dict(torch.load('./model/手机价格预测.pth'))

    # 4. 定义变量, 记录: 预测正确的样本数.
    correct = 0

    # 5. 具体的 每批预测 过程.
    for x, y in test_loader:
        # 5.1 切换模型模式.
        model.eval()

        # 5.2 模型预测(预测的不是直接分类的结果, 而是: 该样本属于每个分类的概率)
        output = model(x)
        # y_pred = torch.softmax(output, dim=1)    # dim=1表示逐行处理.
        # print(f'output: {output}')

        # 5.3 从预测的概率中, 获取: 最大概率对应的分类
        y_pred = torch.argmax(output, dim=1)    # dim=1表示逐行处理.
        # print(f'y_pred: {y_pred}')

        # 5.4 查看样本是否预测正确, 并记录: 正确的样本数.
        # print(f'正确样本: {y}')
        # print(f'预测样本: {y_pred}')
        # print(f'预测结果: {y_pred == y}')
        # print(f'预测结果: {(y_pred == y).sum()}')
        correct += (y_pred == y).sum()

    # 6. 打印: 准确率
    print(f'预测正确的样本数: {correct}, Accuracy(准确率): {correct / len(test_dataset):.4f}')

# 5. 测试
if __name__ == '__main__':
    # 1. 创建数据集.
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f'训练集数据集对象: {train_dataset}')
    # print(f'测试集数据集对象: {test_dataset}')
    # print(f'(输入层, 输入的)特征数: {input_dim}')    # 20
    # print(f'(输出层, 输出的)标签数: {output_dim}')   # 4

    # 2. 创建模型对象.
    # model = PhonePriceModel(input_dim, output_dim)
    # 参1: (神经网络)模型对象, 参2: (批次条数, 输入维度->输入的特征数)
    # summary(model, input_size=(16, input_dim))

    # 3. 模型训练.
    train_model(train_dataset, input_dim, output_dim)

    # 4. 模型测试.
    evaluate_model(test_dataset, input_dim, output_dim)
