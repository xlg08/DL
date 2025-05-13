"""
案例:
    Relu激活函数演示 --> 绘制 Relu函数图像 和 Relu导数图像.

回顾: ANN介绍
    概述:
        人工神经网络(Artificial neural network), 通过多个 神经元 搭建的神经网络, 模仿生物学的神经元, 模拟世间万事万物.

    组成:
        输入层, 隐藏层, 输出层,   每层都可以有N个神经元.
        其中 输入层神经元的个数 = 特征的数量.

    细节:
        1. 同层的多个神经元之间相互不连接.
        2. 本层的每个神经元都会和 上一层的所有神经元建立连接, 叫: 全连接层(Linear, FC)
        3. 搭建神经网络时, 只需要搭建 隐藏层 和 输出层即可.
           且: 输入维度是上一层的神经元的数量, 输出维度是本层的神经元的数量.
        4. 关于神经元个数的经验之谈, 浅层的神经元数量可以多一点, 深层的神经元数量可以少一些.
        5. 激活函数的作用是: 给神经元添加 非线性因素, 则神经元就可以处理 分类问题了.

    激活函数分类:
        Sigmoid激活函数:
            既考虑正样本, 也考虑负样本, 切回把结果映射到: [0, 1]之间.
            数据在[-6, 6]之间有效果, 在[-3, 3]之间效果显著, 求导后范围在 [0, 0.25]
            不适用于深层网络, 容易造成: 梯度消失.
            一般应用于 输出层, 且输出层是二分类的.

        Tanh激活函数:
            既考虑正样本, 也考虑负样本, 切把结果映射到: [-1, 1]之间.
            数据在[-3, 3]之间有效果, 在[-1, 1]之间效果显著, 求导后范围在 [0, 1]
            不太适合于深层网络, 容易造成: 梯度消失.
            一般应用于 浅层隐藏层.

        ReLu激活函数:
            (默认)只考虑正样本, 公式为: f(x) = max(0, x), 所以可能会导致 神经元死亡(可以缓解过拟合情况)
            导数范围: <= 0 导数为0,  >0 导数为 1
            应用最多, 且大多应用于 隐藏层.  有变形版 -> Leaky Relu, PRelu...

"""

# 导包
import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


# 遇到的问题: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# 解决方案:  去 你的Anaconda软件安装路径下的 anaconda3\Lib\site-packages\torch\lib 删除 libiomp5md.dll 这个文件.
# 例如: 我的路径是 C:\Software\DevelopSofware\anaconda3\Lib\site-packages\torch\lib


# 需求1: 绘制Relu函数图像

# 1. 创建1个 1行2列的画布.
fig, axes = plt.subplots(1, 2)
# 2. 准备x轴的值 -> [-20, 20]的等差数列, 1000个元素.
x = torch.linspace(-20, 20, 1000)
# 3. 计算Relu函数的值
y = torch.relu(x)
# 4. 绘制折线图.
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('Relu 函数图像')
# plt.show()

# 需求2: 绘制Relu函数的导数图像.
# 1. 准备x轴的值 -> [-20, 20]的等差数列, 1000个元素.
x = torch.linspace(-20, 20, 1000, requires_grad=True)
# 2. 计算Relu函数的导数
torch.relu(x).sum().backward()   # 表示 标量张量 才能求导.
# 3. 绘制折线图
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('Relu 函数导数图像')
plt.show()