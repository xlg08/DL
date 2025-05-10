"""
案例: 演示张量拼接

涉及到的函数如下:
    cat()   把多个张量按照指定的维度拼接起来, 不改变维度数.
        细节: 除了拼接的维度外, 其它维度(值)必须相同.

    stack() 可以在新的维度上拼接张量, 张量的形状必须完全相同.
"""

# 导包
import torch  

# 定义函数, 演示 cat()函数.
def dm01():

    # 1. 随机获取到2个张量.
    t1 = torch.randint(1, 10, (2, 3))
    print(f't1: {t1}')

    # t2 = torch.randint(1, 10, (2, 6))
    t2 = torch.randint(1, 10, (2, 3))
    print(f't2: {t2}')

    # 2. 通过cat()方式, 拼接t1和t2, 拼接维度为0.
    t3 = torch.cat([t1, t2], dim=0)     # (2, 3) + (2, 3) -> (4, 3)
    print(f't3: {t3}, t3.shape: {t3.shape}')

    # 3. 通过cat()方式, 拼接t1和t2, 拼接维度为1.
    t4 = torch.cat([t1, t2], dim=1)     # (2, 3) + (2, 3) -> (2, 6)
    print(f't4: {t4}, t4.shape: {t4.shape}')

    # 4. 通过cat()方式, 拼接t1和t2, 拼接维度为2.
    # t5 = torch.cat([t1, t2], dim=2)     # (2, 3) + (2, 3) -> 报错.
    # print(f't5: {t5}, t5.shape: {t5.shape}')


# 定义函数, 演示 stack()函数: 可以在新维度上拼接张量, 张量的形状必须完全相同.
def dm02():
    # 1. 随机获取到2个张量.
    t1 = torch.randint(1, 10, (2, 3))
    print(f't1: {t1}')

    # t2 = torch.randint(1, 10, (2, 6))
    t2 = torch.randint(1, 10, (2, 3))
    print(f't2: {t2}')

    # 2. 通过stack()方式, 拼接t1和t2, 拼接维度为0.
    t3 = torch.stack([t1, t2], dim=0)     # (2, 3) + (2, 3) -> (2, 2, 3)
    print(f't3: {t3}, t3.shape: {t3.shape}')

    # 3. 通过stack()方式, 拼接t1和t2, 拼接维度为1.
    t4 = torch.stack([t1, t2], dim=1)     # (2, 3) + (2, 3) -> (2, 2, 3)
    print(f't4: {t4}, t4.shape: {t4.shape}')

    # 4. 通过stack()方式, 拼接t1和t2, 拼接维度为2.
    t5 = torch.stack([t1, t2], dim=2)     # (2, 3) + (2, 3) -> (2, 3, 2)
    print(f't5: {t5}, t5.shape: {t5.shape}')

    t7 = torch.stack([t1, t2, t2], dim=0)  # (2, 3) + (2, 3) + (2, 3) --> (3, 3, 2)
    print(f"t7：", t7.shape)

    # 报错：Dimension out of range (expected to be in range of [-3, 2], but got 3)
    # 尺寸超出范围（预计在 [-3， 2] 范围内，但得到 3）
    # t6 = torch.stack([t1, t2], dim=3)     # 报错
    # print(f"t6：", t6.shape)

# 测试
if __name__ == '__main__':
    # dm01()
    dm02()