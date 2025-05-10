"""
案例:
    演示 张量 和 numpy的 ndarray 之间相互转换.
"""

# 导包
import torch
import numpy as np


# 1. 演示 张量 -> numpy
def dm01():
    # 1. 定义张量.
    t1 = torch.tensor([1, 2, 3])
    print(f't1: {t1}, {type(t1)}')

    # 2. 张量 -> numpy.
    # arr = t1.numpy()          # 共享内存
    arr = t1.numpy().copy()  # 不共享内存
    print(f'arr: {arr}, {type(arr)}')

    # 3. 测试是否共享内存, 思路: 修改arr, 看t1是否被修改
    arr[0] = 100
    print(f'arr: {arr}')  # [100, 2, 3]
    print(f't1: {t1}')  # [1, 2, 3]


# 2. 演示 numpy -> 张量
def dm02():
    # 1. 定义numpy的 ndarray数组.
    arr = np.array([1, 2, 3])
    print(f'arr: {arr}, {type(arr)}')

    # 2. numpy -> 张量.
    # 思路1: from_numpy(), 共享内存.
    t1 = torch.from_numpy(arr)
    # t1 = torch.from_numpy(arr.copy())   # 不共享

    # 思路2: tensor(), 不共享.
    t2 = torch.tensor(arr)

    # 3. 测试是否共享内存, 思路: 改变arr, 看t1, t2是否被修改
    arr[0] = 100
    print(f'arr: {arr}')  # [100, 2, 3]
    print(f't1: {t1}')  # [100, 2, 3]
    print(f't2: {t2}')  # [1, 2, 3]


# 3. 演示 标量(1个数据)  <-> 张量
def dm03():
    # 1. 标量 -> 张量
    t1 = torch.tensor(24)
    print(f't1: {t1}, {type(t1)}')

    # 2. 张量 -> 标量
    value = t1.item()  # 张量中只有1个值才可以用这个函数.
    print(f'value: {value}, {type(value)}')


def dm04():
    # 1. 定义一个张量.
    t1 = torch.tensor([1, 2, 3])
    t1 = torch.Tensor
    print(f't1: {t1}')

    for to in t1:
        print(to.item())


# 4. 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()