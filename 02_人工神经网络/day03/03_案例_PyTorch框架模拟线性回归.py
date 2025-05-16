"""
案例:
    通过PyTorch框架, 模拟线性回归, 给后续课程做铺垫.

后续的 ANN(人工神经网络), CNN(卷积神经网络), RNN(循环神经网络), 都会有 神经元 的概念.
    神经元 = 加权求和 + 激活函数.

总结:
    如果你现在掌握了 PyTorch构建线性回归的 模型训练步骤, 说明后续要用的 神经元的 加权求和这部分 流程你就基本清楚了.
"""

# 导包
import torch
from torch.utils.data import TensorDataset      # 构造数据集对象
from torch.utils.data import DataLoader         # 数据加载器
from torch import nn                            # nn模块中有平方损失函数和假设函数
from torch import optim                         # optim模块中有优化器函数
from sklearn.datasets import make_regression    # 创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


# 1. 构建 线性回归 数据集.
def create_dataset():
    # 1. 创建数据集 采用 make_regression 函数创建线性回归数据集.
    x, y, coef = make_regression(
        n_samples=100,  # 样本数量
        n_features=1,   # 特征数量
        n_targets=1,    # 目标数量
        bias=14.5,      # 偏置
        noise=10,       # 噪声
        coef=True,      # 是否返回系数
        random_state=24 # 随机数种子
    )

    # 2. 把上述的x(特征), y(标签)封装成 张量.
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 3. 返回结果.
    # print(f'x: {x.shape}, y: {y.shape}, coef: {coef}')
    # print(f'x: {x}')
    # print(f'y: {y}')
    # print(f'coef: {coef}')

    return x, y, coef

# 2. 模型训练.
def train(x, y, coef):

    # 1. 基于x, y构建 数据集对象.
    dataset = TensorDataset(x, y)

    # 2. 基于上述的 数据集对象 -> 封装成 数据加载器对象, 可以: 分批次获取数据(目的: 防止内存溢出)
    # 参1: 数据集对象, 参2:  每批次的数据条数, 参3: 是否打乱数据.
    # 例如: 数据集100条, 16条/批, 共计: 7批
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. 构建(线性回归)模型.
    # 参1: 输入的特征数量, 参2: 输出的标签数量.
    model = nn.Linear(in_features=1, out_features=1)

    # 4. 构建损失函数.
    criterion = nn.MSELoss()

    # 5. 构建优化器, 这里是模拟: 随机梯度下降.
    # 参1: 待优化的模型参数, 参2: 学习率.
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 6. 模型训练.
    # 6.1 定义变量记录 训练的总轮数, 每轮的平均损失, 总损失, 训练的样本数.
    epochs, loss_list, total_loss, train_samples = 100, [], 0.0, 0
    # 6.2 具体的每轮训练的过程.
    for epoch in range(epochs):                 # 例如: 第1轮,                   第2轮....
        # 6.3 每轮都会针对于100条数据集, 按16条/批 进行批次训练. 这里是具体的每批次的训练过程.
        for train_x, train_y in dataloader:     # 例如: 第1批, 第2批...第7批      第1批, 第2批...第7批
            # 6.4 模型预测(前向传播, 正向传播).
            y_pred = model(train_x)
            print("y预测：", y_pred)
            # 6.5 计算损失.
            # 细节: 把 真实值(train_y)转成 n行1列的维度, 方便和 预测值进行计算.
            loss = criterion(y_pred, train_y.reshape(-1, 1))        # loss是 该批次的平均损失, 例如: 16条样本的平均损失...
            # 6.6 统计总损失.
            total_loss += loss.item()                               # 统计每批次的平均损失 求和    例如: total_loss = 7批的平均损失求和.
            # 6.7 计算训练的样本数.
            train_samples += 1                                      # 这里的1代表: 1批.          例如: train_samples = 7批
            # 6.8 反向传播, 计算梯度.
            # 6.8.1 梯度清零.
            optimizer.zero_grad()
            # 6.8.2 计算梯度 -> 反向传播
            loss.backward()
            # 6.8.3 更新参数.
            optimizer.step()    # 类似于做了: w = w - lr * w.grad
        # 6.9 统计(本轮的)平均损失, 并添加到列表中.
        loss_list.append(total_loss / train_samples)
        # 6.10 输出训练信息.
        print(f'轮数: {epoch + 1}, 平均损失: {total_loss / train_samples:.4f}, 训练样本数: {train_samples}')
    # 6.11 打印最终信息
    print(f'训练结束, 模型参数: {model.state_dict()}')

    # 7. 可视化.
    # 7.1 绘制损失变化曲线图.
    plt.plot(range(epochs), loss_list)  # x轴: 训练的轮数, y轴: 每轮的平均损失.
    plt.xlabel('训练的轮数')
    plt.ylabel('该轮的平均损失')
    plt.title('损失变化曲线图')
    plt.grid()
    plt.show()

    # 7.2 绘制拟合直线.
    # 7.2.1 绘制数据集(真实的样本, 标签)
    plt.scatter(x, y)
    # 7.2.2 生成1000个x值(等差数列), 预测对应的y值.
    x = torch.linspace(x.min(), x.max(), 1000)

    # 细节: 张量 * numpy -> 可以,   numpy * 张量 -> 不可以.
    # 解决办法:  ① 交换两个乘数的位置.   ② 把 numpy转成 张量, 然后相乘.

    # 7.2.3 计算预测值.
    y_pred = torch.tensor([v * model.weight + model.bias for v in x])
    # 7.2.4 计算真实值
    y_true = torch.tensor([v * coef + 14.5 for v in x])
    # y_true = torch.tensor([torch.tensor(coef) * v + 14.5 for v in x])


    # 7.2.5 绘制预测值和真实值 的 拟合回归线(直线)
    plt.plot(x, y_pred, label='预测值', color='red')
    plt.plot(x, y_true, label='真实值', color='green')
    # 7.2.6 绘制图例, 网格, 标题.
    plt.legend()
    plt.grid()
    plt.title('线性回归->预测值和真实值关系')
    plt.show()


# 3. 测试.
if __name__ == '__main__':
    # 3.1 获取 线性回归数据集.
    x, y, coef = create_dataset()

    # 3.2 线性回归 -> 模型训练, 预测, 绘制损失函数曲线图.
    train(x, y, coef)

