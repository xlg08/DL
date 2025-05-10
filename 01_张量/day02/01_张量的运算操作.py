"""
案例:
    演示张量的运算操作.

涉及到的操作如下:
    场景1: 张量的基本运算.
        涉及到的函数如下: 加减乘除取负号：
            add、sub、mul、div、neg
            add_、sub_、mul_、div_、neg_（其中带下划线的版本会修改原数据）

    场景2: 张量的点乘运算.
    场景3: 张量的矩阵运算.

掌握:
    t1 * t2  -> 点乘运算, 行列数一致.
    t1 @ t2  -> 矩阵相乘, 条件: A列=B行, 结果: A行B列
"""

# 导包
import torch


# 场景1: 定义函数, 实现 张量的基本运算.
def dm01_basic_operation():
    # 1. 定义张量.
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}')

    # 2. 张量 和 数值运算, 则: 该数值会和张量中的每个元素逐个运算.
    t2 = t1 + 10
    print(f't2: {t2}')
    print('-' * 24)

    # 3. 演示 add()函数, 不会修改源数据.
    t3 = t1.add(10)
    print(f't3: {t3}')
    print(f't1: {t1}')      # 源数据未发生变化.
    print('-' * 24)

    # 4. 演示 add_()函数, 会修改源数据.
    t4 = t1.add_(10)        # 类似于 Pandas部分的 inplace=True操作.
    print(f't4: {t4}')
    print(f't1: {t1}')      # 源数据发生变化.
    print('-' * 24)

    # 5. 演示 neg_()函数, 取反 ->  10: -10,   -10: 10
    t5 = torch.tensor([[1, -2, 3], [4, -5, -6]])
    t5.neg_()
    print(f't5: {t5}')

    # 对内部为布尔类型的值的张量进行取反运算会报错
    # t1 = torch.tensor(False)
    t1 = torch.tensor(True)
    # print(~t1)      # 对内部为内尔类型的值的张量进行取反操作
    print(t1.logical_not())  # 对内部为内尔类型的值的张量进行取反操作
    # print(torch.logical_not(t1))      # 对内部为内尔类型的值的张量进行取反操作


# 场景2: 定义函数, 实现 张量的点乘运算.
def dm02_dot_operation():
    # 1. 定义张量.   3行2列
    t1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    print(f't1: {t1}')

    # 2. 定义张量.   3行2列
    t2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
    print(f't2: {t2}')

    # 3. 点乘, 要求: 行列数一致,  结果: 对应元素进行乘法运算.
    # 写法1: t1.mul(t2)
    t3 = t1.mul(t2)
    print(f't3: {t3}')

    # 写法2: t1 * t2      推荐掌握
    t4 = t1 * t2
    print(f't4: {t4}')

    # 4.扩展 dot()函数, 也能实现点乘, 只能是一维的.
    # print(t1.dot(t2))       # 报错, 因为处理不了二维的.
    t5 = torch.tensor([1, 2, 3])
    t6 = torch.tensor([4, 5, 6])
    print(t5.dot(t6))   # 结果是矩阵乘法, 即: 先乘, 后加.


# 场景3: 定义函数, 实现 张量的矩阵运算.
def dm03_matrix_operation():
    # 1. 定义张量.   2行3列
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f't1: {t1}')

    # 2. 定义张量.   3行2列
    t2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
    print(f't2: {t2}')

    # 3. 矩阵相乘, 要求: A列=B行, 结果: A行B列
    # 写法1: t1 @ t2 
    t3 = t1 @ t2        # 推荐掌握.
    print(f't3: {t3}')

    # 写法2: t1.matmul(t2)
    t4 = t1.matmul(t2)      # matrix multiplication,  矩阵乘法
    print(f't4: {t4}')


def dm04():
    t1 = torch.tensor(1)

    t2 = torch.tensor([[1, -2, 3], [4, -5, 6]])
    # 加法操作
    print("加法操作：\n", torch.add(t2, 10))
    print("加法操作：\n", t2.add(10))

    # 减法操作
    print("减法操作：\n", torch.sub(t2, 10))
    print("减法操作：\n", t2.sub(10))

    # 乘法操作
    print("乘法操作：\n", torch.mul(t2, 10))
    print("乘法操作：\n", t2.mul(10))

    # 除法操作
    print("除法操作：\n", torch.div(t2, 10))
    print("除法操作：\n", t2.div(10))

    # 取反操作
    print("取反操作：\n", t1.neg_())
    print("取反操作：\n", torch.neg_(t1))

    print("♥" * 40)

    # 点乘 两个矩阵的行列数一致
    t3 = torch.tensor([1, 2, 3])
    t4 = torch.tensor([4, 5, 6])
    # print("点乘操作：\n", torch.mv(t3, t4))
    # print("点乘操作：\n", t3.mv(t4))
    print("点乘 * 操作：\n", t3 * t4)  # 对应元素相乘
    print("点乘 mul() 操作：\n", torch.mul(t3, t4))  # 对应元素相乘
    print("点乘 mul() 操作：\n", t3.mul(t4))  # 对应元素相乘
    # 点乘的拓展内容
    print("点乘dot()方法操作：\n", torch.dot(t3, t4))  # 也能实现点乘，但是只能是一维的,结果为矩阵乘法，即：对应元素先乘，后相加
    print("点乘dot()方法操作：\n", t3.dot(t4))

    print("♥" * 40)

    # 矩阵乘法操作，要求： A列 = B行        # 结果为：A行B列
    print("矩阵乘法mm()方法操作：\n", torch.mm(t2, t2.t()))
    print("矩阵乘法mm()方法操作：\n", t2.mm(t2.t()))
    print("矩阵乘法matmul()操作：\n", torch.matmul(t2, t2.t()))  # matrix multiplication, 矩阵乘法
    print("矩阵乘法matmul()操作：\n", t2.matmul(t2.t()))
    print("矩阵乘法 @ 操作：\n", t2 @ t2.t())

    t5 = torch.tensor([True, True, True])
    t6 = torch.tensor([[True], [True], [True]])

    print(t5 * t5)
    # print(t5 @ t5.t())        # 报错，


# 测试
if __name__ == '__main__':
    # dm01_basic_operation()
    # dm02_dot_operation()
    dm03_matrix_operation()