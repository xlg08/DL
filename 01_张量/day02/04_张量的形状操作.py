"""
案例:
    张量的形状操作, 可以基于需求 改变张量的形状.

涉及到的函数:
    reshape()   可以在保证张量数据不变的前提下, 改变数据的维度.
    squeeze()   删除(所有)形状为1的维度 -> 降维
    unsqueeze()  增加维度 -> 升维, 添加形状为1的维度(升维)

    transpose() 一次只能交换两个维度.
    permute()   一次可以交换多个维度.

    view()          修改张量的维度, 只能操作: 连续的张量(内存中存储顺序 和 张量的逻辑顺序一致)
    is_contiguous() 判断张量是否连续
    contiguous()    把 不连续的张量 -> 连续的.

掌握:
    reshape(), squeeze(), unsqueeze(), transpose(), permute()
"""

# 导包
import torch
import numpy as np


# 1. 定义函数, 演示 reshape()函数.
def dm01():
    # 1. 定义张量.  2行3列.
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}')
    print(f't1: {t1.shape}, 行:{t1.shape[0]}, 列:{t1.shape[1]}, {t1.shape[-1]}')
    print(f't1: {t1.size()}, 行:{t1.size()[0]}, 列:{t1.size()[1]}, {t1.size()[-1]}')  # 效果同上.
    print('-' * 24)

    # 2. 改变形状, 因为张量的总元素为 2 * 3 = 6, 所以改变形状后总元素不变.
    t2 = t1.reshape(3, 2)
    print(f't2: {t2}')
    print(f't2: {t2.shape}, 行:{t2.shape[0]}, 列:{t2.shape[1]}, {t2.shape[-1]}')
    print('-' * 24)

    t3 = t1.reshape(1, 6)
    print(f't3: {t3}')
    print(f't3: {t3.shape}, 行:{t3.shape[0]}, 列:{t3.shape[1]}, {t3.shape[-1]}')

    t4 = t1.reshape(6, 1)
    print(f't4: {t4}')
    print(f't4: {t4.shape}, 行:{t4.shape[0]}, 列:{t4.shape[1]}, {t4.shape[-1]}')

    # 3. 改变形状, 总元素个数变了.
    # t5 = t1.reshape(2, 6)   # 报错
    t5 = t1.reshape(2, 4)  # 报错
    print(f't5: {t5}')


# 2. 定义函数, 演示 squeeze(), unsqueeze()函数.
def dm02():
    # 1. 定义张量, 5个元素
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f't1: {t1}, t1.shape: {t1.shape}')

    # 2. 在0轴上, 增加形状为1的维度.
    t2 = t1.unsqueeze(dim=0)  # t2: [1, 5]
    print(f't2: {t2}, t2.shape: {t2.shape}')

    # 3. 在1轴上, 删除形状为1的维度.
    t3 = t1.unsqueeze(dim=1)  # t3: [5, 1]
    print(f't3: {t3}, t3.shape: {t3.shape}')

    # 4. 尝试在2维上增加形状为1的维度.   报错, 不能处理没有的维度.
    # t4 = t1.unsqueeze(dim=2)    # t3: [5, ?,  1]
    # print(f't4: {t4}, t4.shape: {t4.shape}')

    # 5. 降维. squeeze() -> 删除所有形状为1的维度.
    # t5 = t2.squeeze()
    t5 = t3.squeeze()
    print(f't5: {t5}, t5.shape: {t5.shape}')

    # 6. 重新定义多维, 且包含1的维度.
    t6 = torch.randint(1, 10, (2, 1, 3, 1, 5))
    print(f't6: {t6}, t6.shape: {t6.shape}')

    t7 = t6.squeeze()
    print(f't7: {t7}, t7.shape: {t7.shape}')  # [2, 3, 5]

    t8 = t6.squeeze(dim=1)
    print(f't8: {t8}, t8.shape: {t8.shape}')  # (2, 3, 1, 5)


# 3. 定义函数, 演示 transpose(), permute()函数.
def dm03():
    # 1. 创建3维张量.
    t1 = torch.randint(1, 10, (2, 3, 5))
    # t1 = torch.tensor(np.random.randint(1, 10, (2, 3, 5)))
    print(f't1: {t1}, t1.shape: {t1.shape}')  # (2, 3, 5)

    # 2. 需求1: 交换0轴 和 1轴.  (2, 3, 5) -> (3, 2, 5)
    t2 = t1.transpose(dim0=0, dim1=1)
    print(f't2: {t2}, t2.shape: {t2.shape}')

    # 3. 需求2: 从 (2, 3, 5) -> (5, 2, 3)
    t3 = t1.permute(dims=(2, 0, 1))
    print(f't3: {t3}, t3.shape: {t3.shape}')


# 4. 定义函数, 演示 view(), contiguous(), is_contiguous() -> 理解.
def dm04():
    # 1. 定义张量.      2行3列
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}, t1.shape: {t1.shape}')

    # 2. 用transpose()修改形状.
    # 数据: (1, 2, 3, 4, 5, 6) -> (1, 4, 2, 5, 3, 6)
    t2 = t1.transpose(dim0=0, dim1=1)
    print(f't2: {t2}, t2.shape: {t2.shape}')  # (3, 2)

    # 3. 判断 张量 是否连续, 即: 张量在底层的存储顺序 和 张量中的逻辑顺序是否一致.
    print(t1.is_contiguous())  # True
    print(t2.is_contiguous())  # False
    print('-' * 24)

    # 4. 修改t1张量的形状.
    t4 = t1.view(3, 2)  # 从(2, 3) -> (3, 2)
    print(t4.is_contiguous())  # T rue
    print(f't4: {t4}, t4.shape: {t4.shape}')

    # 5. 尝试修改t2形状, 错误.
    # t5 = t2.view(2, 3)          # (3, 2) -> (2, 3)
    # print(f't5: {t5}, t5.shape: {t5.shape}')

    # 6. 先通过 contiguous() 把t2 -> 连续的 -> view()改变形状.
    t6 = t2.contiguous().view(2, 3)
    print(t6.is_contiguous())  # True
    print(f't6: {t6}, t6.shape: {t6.shape}')


def dm08():
    np.random.seed(24)
    t1 = torch.tensor(np.random.randint(1, 10, (2, 3, 5)))
    print(t1.shape)

    t2 = t1.transpose(dim0=0, dim1=1)
    print(t2.shape)

    t3 = t1.transpose(dim0=0, dim1=2)
    print(t3.shape)
    t4 = t3.transpose(dim0=2, dim1=1)
    print(t4.shape)

    t5 = t1.permute(dims=(2, 0, 1))
    print(t5.shape)


# 5. 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()
