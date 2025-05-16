"""
案例:
    演示损失函数之 二分类的损失函数.

损失函数介绍:
    概述/作用:
        用来评估模型好坏的, 即: 模型的拟合情况.
    分类:
        回归损失函数:
            MAE:
                Mean Absolute Error, 平均绝对误差.
                误差的绝对值之和的 平均值, 不会放大异常值, 可能会越过极小值.
                一般不会单独用, 而是作为其它损失函数的正则化项来用, 增加稀疏性, 缓解过拟合.
            MSE:
                Mean Squared Error, 均方误差.
                误差的平方之和的 平均值, 放大异常值, 如果预测值和真实值差距较大, 容易梯度爆炸.
                可以单独用, 但是大多数应用场景是, 作为其它损失函数的正则化项来用.

            Smooth L1:


        分类损失函数:
            二分类损失函数:    BCELoss
            多分类损失函数:    CrossEntropyLoss = Softmax + 损失计算

            记忆:
                无论是二分类损失函数, 还是多分类损失函数, API底层默认都是算 平均损失(mean)
"""

# 需求: 演示下 多分类损失函数的应用.

# 导包
import torch
import torch.nn as nn


# 1. 定义函数, 演示: 多分类交叉熵损失函数.
def dm01():
    # 1. 定义真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)


    # 2. 定义预测值(概率)
    y_pred = torch.tensor([1.2, 1.7, 1.9], requires_grad=True, dtype=torch.float)

    # 3. 定义回归(MSE)损失函数.
    criterion = nn.MSELoss()


    # 4. 计算回归(MSE)损失函数.
    loss = criterion(y_pred, y_true)

    # 5. 打印损失.
    print(f'回归(MSE)损失函数: {loss}')


# 2. 测试
if __name__ == '__main__':
    dm01()