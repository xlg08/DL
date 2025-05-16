"""
案例:
    演示自动微分的应用场景.

需求:
    把自动微分功能, 带入到 整体流程图中, 观察 前向传播 和 反向传播的完整动作.

前向传播(正向传播):
    x, w, b -> z, 预测值

反向传播:
    Loss(z,    y)   -> 梯度 -> W1 = W0 - lr * 梯度 -> 更新权重
       预测值 真实值
"""

# 导包
import torch

# 1. 准备特征数据
x = torch.ones(2, 5)
print(f'x: {x}')

# 2. 准备标签数据(真实值)
y = torch.zeros(2, 3)
print(f'y: {y}')

# 3. 准备 权重矩阵(weight), 采用 标准的正态分布进行初始化.
w = torch.randn(5, 3, requires_grad=True)
print(f'w: {w}')

# 4. 准备 偏置矩阵(bias), 采用 标准的正态分布进行初始化.
b = torch.randn(3, requires_grad=True)
print(f'b: {b}')

# 5. 结合上述的x,w,b, 进行前向传播, 得到预测值(z)
# 回顾多元线性公式: z = w的转置 @ x + b
# 这里是模拟, 无需写的这么麻烦, 我们将公式调整为: z = x @ w + b
z = x @ w + b
# z = x.matmul(w) + b     # 效果同上.

# 6. 定义损失函数, 这里采用 MSE(Mean Square Error, 均方误差)
loss = torch.nn.MSELoss()

# 7. 计算损失值.
loss = loss(z, y)

# 8. 开始反向传播, 计算梯度. 梯度会被保存到 w, b 的 .grad 属性中.
loss.backward()     # loss本身就是一个标量, 所以这里无需sum()
# loss.sum().backward()

# 9. 打印梯度.
print('-' * 24)
print(f'w.grad: {w.grad}')
print(f'b.grad: {b.grad}')








