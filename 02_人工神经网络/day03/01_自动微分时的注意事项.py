"""
案例:
    演示自动微分时的注意事项.

自动微分介绍:
    它是PyTorch框架的内置功能, 用于计算 损失函数的 导数 -> 梯度.
    目的:
        更新梯度.
    梯度计算公式:
        W1 = W0 - lr * grad

    注意事项:
        一个张量一旦设置了 自动微分, 则该张量就不能直接转成numpy对象了.
        可以通过 detach()函数解决.
"""

# 导包
import torch
import numpy as np


# 1. 定义张量, 设置自动微分.
t1 = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float)
print(f't1: {t1}')

# 2. 尝试把t1张量 -> numpy数组
# arr = t1.numpy()        # 报错.
# print(f'arr: {arr}, {type(arr)}')

# 3. 解决方案, 通过 detach()函数 把t1 拷贝一份, 共用数据, 但是不共享地址.
t2 = t1.detach()
print(f't2: {t2}')

# 4. 把t2张量 -> numpy数组
arr = t2.numpy()
print(f'arr: {arr}, {type(arr)}')
print('-' * 24)

# 5. 查看张量是否设置了 自动微分.
print(f't1: {t1.requires_grad}')    # True
print(f't2: {t2.requires_grad}')    # False
print('-' * 24)

# 6. 查看t1 和 t2的地址.
print(f't1: {id(t1)}')  # 0x01
print(f't2: {id(t2)}')  # 0x02
print('-' * 24)

# 7. 链式编程.
t3 = t1.detach().numpy()     
print(f't3: {t3}')



