"""
案例:
    演示PyTorch的自动微分模块.

细节:
    1. 所谓的自动微分模块, 就是 PyTorch提供的 求导的 方式, 用于结合 反向传播(BP算法, Back Propagation) 更新权重(梯度)的.
    2. PyTorch不支持 向量张量 和 向量张量的求导, 只支持 标量张量 和 向量张量的求导.
        例如: y.backward()        y必须是 标量.
    3. x.grad 方式可以获取到 x的 梯度值.
    4. 梯度值计算时, 每次会 累加 上一次的梯度值.
"""

# 需求: 通过自动微分模块, 求出 y = 2 * x^2 的 梯度值.

# 导包
import torch


def dm01():

    # 1. 定义变量x, 表示: 初始的梯度值. 即: W0
    # 参1: 初始梯度值.  参2: 是否需要求导(自动微分, 计算梯度).  参3: 数据类型, 必须是浮点型.
    # x = torch.tensor(10, requires_grad=True, dtype=torch.float32)
    x = torch.tensor([10, 20], requires_grad=True, dtype=torch.float32)

    # 2. 定义变量y, 表示: 目标函数(损失函数), 即: Loss
    y = 2 * x ** 2  # y = 2 * 10² = 200

    # 3. 打印一下y值, 和 y的曲线函数类型.
    print(f'y: {y}')  # 200

    print(f'y.grad_fn: {y.grad_fn}')  # <MulBackward0 object at 0x000002B1C3806890>

    # 4. 计算之前, 打印一下x的值(权重) 和 x的梯度值.
    print(f'x.data: {x.data}, x.grad: {x.grad}')  # 10, None

    # 5. 损失函数求导 = 自动微分 = 计算结果为: 梯度.
    # 手动算梯度: y = 2*x² -> 求导后, y' = 4x = 40
    # 自动微分模块, 求导.
    # y.backward()        # 务必保证这里的y是 -> 标量, 因为这里的y是标量, 所以可以这么写.
    # 真正写法.
    y.sum().backward()

    # 6. 打印一下x的值(权重) 和 x的梯度值.
    print(f'x.data: {x.data}, x.grad: {x.grad}')  # 10, 40
    # .data 可以获取一个与原始张量共享数据但不参与计算图的张量。
    print(f"x.data的类型：{type(x.data)}")      # <class 'torch.Tensor'>
    print(f"x.grad的类型：{type(x.grad)}")      # <class 'torch.Tensor'>


    # 7. 假设学习率是0.01, 结合 梯度更新公式: W1 = W0 - lr * grad
    x.data = x.data - 0.01 * x.grad

    print(f'x.data: {x.data}')  # 9.6

if __name__ == '__main__':

    dm01()