"""
案例:
    自动微分 结合反向传播, 更新梯度.  多轮训练.

需求:
    求 y = x**2 + 20 的极小值点 并打印y是最小值时 w的值(梯度)

思路/步骤:
    1. 定义点 x=10 requires_grad=True  dtype=torch.float32
    2. 定义函数 y = x**2 + 20
    3. 利用梯度下降法 循环迭代1000 求最优解
    3.1 正向计算(前向传播)
    3.2 梯度清零 x.grad.zero_()
    3.3 反向传播
    3.4 梯度更新 x.data = x.data - 0.01 * x.grad

细节:
    多轮更新参数是, 梯度值会累加, 我们每轮训练, 梯度要清零.
"""

# 导包
import torch

# 需求: 求 y = x**2 + 20 的极小值点 并打印y是最小值时 w的值(梯度)       y就是损失函数, 其值越小, 模型的拟合情况越好.
# 1. 定义点 x=10 requires_grad=True  dtype=torch.float32
# 参1: 初始权重.  参2: 是否需要求梯度.  参3: 浮点型
x = torch.tensor(10, requires_grad=True, dtype=torch.float32)

# 2. 定义函数 y = x**2 + 20
y = x ** 2 + 20  # y值就是: 损失值, 其值越小, 模型的拟合情况越好.
# print('开始 权重x初始值:%.6f (0.01 * x.grad):无 y:%.6f' % (x, y))
print(f'开始: 权重x初始值: {x}, 学习率*梯度 = 0.01 * x.grad: 无, 损失值y: {y}')

# 3. 利用梯度下降法 循环迭代100 求最优解
epochs = 100  # 表示训练的轮式
for epoch in range(epochs):  # epoch: 0 ~ 99
    # 3.1 正向计算(前向传播)
    y = x ** 2 + 20

    # 3.2 梯度清零 x.grad.zero_()
    if x.grad is not None:
        x.grad.zero_()

    # 3.3 反向传播
    y.sum().backward()  # y = x² + 20   ->  y' = 2x
    # print(f'第{epoch + 1}轮, x.grad: {x.grad}')

    # 3.4 梯度更新 x.data = x.data - 0.01 * x.grad
    x.data = x.data - 0.01 * x.grad  # 等价于: W1 = W0 - 学习率 * 梯度

    # 3.5 打印本轮训练后的结果.
    print(f'第{epoch + 1}轮: 权重x初始值: {x}, 学习率*梯度 = 0.01 * x.grad: {0.01 * x.grad}, 损失值y: {y}')
