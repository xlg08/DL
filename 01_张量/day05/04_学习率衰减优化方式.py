"""
案例:
    演示 学习率衰减优化方式.

背景:
    之前学的 AdaGrad, RMSProp, Adam这些梯度下降优化方式, 都可以修改 学习率, 但是底层无法人为精准控制, 例如: *轮 更新一次, 制定轮更新等...
    针对于此, PyTorch中加入了 学习率衰减优化方式, 可以人为的控制学习率的变化.

方式:
    1. 等间隔学习率优化.
        间隔制定的轮数, 优化(更新)学习率.
        lr新 = lr旧 * gamma
    2. 指定间隔学习率优化.
        设置指定的轮数, 当达到指定的轮数后, 即可自动优化 学习率.
        lr新 = lr旧 * gamma
    3. 指数学习率优化.
        lr新 = lr旧 * gamma ^ epoch
"""

# 导包
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 定义函数, 演示: 如何实现 等间隔学习率优化.
def dm01():
    # 1. 定义变量, 记录: 训练的总轮数, 初始学习率, 每轮的迭代器个数.
    epochs, lr, iteration = 200, 0.1, 10

    # 2. 定义变量, 记录: 真实值.
    y_true = torch.tensor([0], dtype=torch.float32)

    # 3. 定义变量, 记录: w(权重), x(特征)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad= True)

    # 4. 创建优化器对象, 采用: 动量法.
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 5. 创建 学习率衰减优化对象.
    # 思路1: 等间隔学习率衰减优化对象, 设置 50轮/次, 更新学习率.
    # 参1: 优化器对象.   参2: 衰减的轮数间隔.  参3: 衰减系数, 即: lr新 = lr旧 * gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)   # [0.1, 0.05, 0.0125...]

    # 6. 定义列表, 分别记录: 训练的轮数, 该轮的学习率.
    epoch_list, lr_list = [], []

    # 7. 具体的 每轮 训练过程
    # 7.1 逐轮训练.
    for epoch in range(epochs):     # epoch: 0 ~ 200, 包左不包右
        # 7.2 添加轮数, 以及本轮的 学习率
        epoch_list.append(epoch)                    # [0, 1, 2, 3, 4, 5, 6......50, 51.......100, 101.....]
        lr_list.append(scheduler.get_last_lr())     # [0.1, 0.1.................0.05, 0.05...0.025........]

        # 7.3 每轮的训练过程, 即: 每轮有 10个 迭代器.
        for i in range(iteration):
            # 7.4 计算预测值.
            y_pred = w * x
            # 7.5 计算损失值.
            loss = (y_pred - y_true) ** 2
            # 7.6 反向传播.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 7.7 走这里,说明: 本轮(一轮)训练完成. 更新本轮的学习率.
        scheduler.step()

    # 8. 打印下: 每轮的 学习率
    print(f'epoch_list: {epoch_list}')  # [0, 1, 2, 3, 4, 5, 6......50, 51.......100, 101.....199]
    print(f'lr_list: {lr_list}')        # [0.1, 0.1.................0.05, 0.05...0.025........]

    # 9. 画图, 绘制: 训练的轮数 以及 每轮的 学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

# 2. 定义函数, 演示: 如何实现 指定间隔学习率优化.
def dm02():
    # 1. 定义变量, 记录: 训练的总轮数, 初始学习率, 每轮的迭代器个数.
    epochs, lr, iteration = 200, 0.1, 10

    # 2. 定义变量, 记录: 真实值.
    y_true = torch.tensor([0], dtype=torch.float32)

    # 3. 定义变量, 记录: w(权重), x(特征)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad= True)

    # 4. 创建优化器对象, 采用: 动量法.
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 5. 创建 学习率衰减优化对象.
    # 思路1: 等间隔学习率衰减优化对象, 设置 50轮/次, 更新学习率.
    # 参1: 优化器对象.   参2: 衰减的轮数间隔.  参3: 衰减系数, 即: lr新 = lr旧 * gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)   # [0.1, 0.05, 0.0125...]

    # 思路2: 指定间隔学习率衰减优化对象, 设置第50, 125, 160轮 更新学习率.
    # milestones = [24, 50, 60, 90, 125, 130, 160]
    milestones = [50, 125, 160]
    # 参1: 优化器对象.   参2: 衰减的(指定)轮数.  参3: 衰减系数, 即: lr新 = lr旧 * gamma
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # 6. 定义列表, 分别记录: 训练的轮数, 该轮的学习率.
    epoch_list, lr_list = [], []

    # 7. 具体的 每轮 训练过程
    # 7.1 逐轮训练.
    for epoch in range(epochs):     # epoch: 0 ~ 200, 包左不包右
        # 7.2 添加轮数, 以及本轮的 学习率
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        # 7.3 每轮的训练过程, 即: 每轮有 10个 迭代器.
        for i in range(iteration):
            # 7.4 计算预测值.
            y_pred = w * x
            # 7.5 计算损失值.
            loss = (y_pred - y_true) ** 2
            # 7.6 反向传播.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 7.7 走这里,说明: 本轮(一轮)训练完成. 更新本轮的学习率.
        scheduler.step()

    # 8. 打印下: 每轮的 学习率
    print(f'epoch_list: {epoch_list}')
    print(f'lr_list: {lr_list}')

    # 9. 画图, 绘制: 训练的轮数 以及 每轮的 学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()


# 3. 定义函数, 演示: 如何实现 指数学习率优化.
def dm03():
    # 1. 定义变量, 记录: 训练的总轮数, 初始学习率, 每轮的迭代器个数.
    epochs, lr, iteration = 200, 0.1, 10

    # 2. 定义变量, 记录: 真实值.
    y_true = torch.tensor([0], dtype=torch.float32)

    # 3. 定义变量, 记录: w(权重), x(特征)
    x = torch.tensor([1.0], dtype=torch.float32)
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad= True)

    # 4. 创建优化器对象, 采用: 动量法.
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    # 5. 创建 学习率衰减优化对象.
    # 思路1: 等间隔学习率衰减优化对象, 设置 50轮/次, 更新学习率.
    # 参1: 优化器对象.   参2: 衰减的轮数间隔.  参3: 衰减系数, 即: lr新 = lr旧 * gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)   # [0.1, 0.05, 0.0125...]

    # 思路2: 指定间隔学习率衰减优化对象, 设置第50, 125, 160轮 更新学习率.
    # milestones = [24, 50, 60, 90, 125, 130, 160]
    # milestones = [50, 125, 160]
    # 参1: 优化器对象.   参2: 衰减的(指定)轮数.  参3: 衰减系数, 即: lr新 = lr旧 * gamma
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    # 思路3: 指数衰减学习率衰减优化对象.  计算公式: lr新 = lr旧 * gamma ^ epoch
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 6. 定义列表, 分别记录: 训练的轮数, 该轮的学习率.
    epoch_list, lr_list = [], []

    # 7. 具体的 每轮 训练过程
    # 7.1 逐轮训练.
    for epoch in range(epochs):     # epoch: 0 ~ 200, 包左不包右
        # 7.2 添加轮数, 以及本轮的 学习率
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())

        # 7.3 每轮的训练过程, 即: 每轮有 10个 迭代器.
        for i in range(iteration):
            # 7.4 计算预测值.
            y_pred = w * x
            # 7.5 计算损失值.
            loss = (y_pred - y_true) ** 2
            # 7.6 反向传播.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 7.7 走这里,说明: 本轮(一轮)训练完成. 更新本轮的学习率.
        scheduler.step()

    # 8. 打印下: 每轮的 学习率
    print(f'epoch_list: {epoch_list}')
    print(f'lr_list: {lr_list}')

    # 9. 画图, 绘制: 训练的轮数 以及 每轮的 学习率
    plt.plot(epoch_list, lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()


# 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()