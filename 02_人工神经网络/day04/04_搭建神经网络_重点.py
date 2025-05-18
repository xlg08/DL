"""
案例:
    演示如何搭建神经网络.

搭建神经网络的思路, 步骤:
    1. 定义1个类, 继承 nn.Module 类.
    2. 在 __init__()魔法方法中, 完成:
        step1: 初始化父类的成员
        step2: 搭建神经网络, 即:
            隐藏层(输入的神经元个数, 输出的神经元个数),  输出层(输入的神经元个数, 输出的神经元个数)
    3. 在 forward()函数中, 完整:
        前向传播(正向传播)过程. 该函数在模型预测时, 会自动调用, 无需手动操作.

    神经网络搭建和参数计算
    在pytorch中定义深度神经网络其实就是层堆叠的过程，继承自nn.Module，
    实现两个方法：
        1.__init__方法中定义网络中的层结构，主要是全连接层，并进行初始化
        2.forward方法，在实例化模型的时候，底层会自动调用该函数。
            该函数中为 初始化定义的 layer(层) 传入数据，进行前向传播等。

"""

# 需求: 搭建神经网络, 2个隐藏层, 1个输出层.
#   其中:  隐藏层1 -> Xavier正态  + Sigmoid,
#         隐藏层2 -> kaiming正态 + Relu,
#          输出层 -> Softmax()

# 导包
import torch                      # 深度学习计算框架, PyTorch框架
import torch.nn as nn             # 神经网络模块, neural network(神经网络)
from torchsummary import summary  # 计算模型参数,查看模型结构, pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/


# 1. 定义一个类, 继承 nn.Module 类 -> 神经网络模型类.
class ModelDemo(nn.Module):
    # 1.1 在 __init__()魔法方法中, 完成: 初始化父类成员 + 搭建神经网络.
    def __init__(self):
        # 初始化父类成员
        super().__init__()
        # 搭建神经网络
        # step1: 搭建 隐藏层1, 隐藏层2, 输出层
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)

        # step2: 初始化(各层的权重, 偏置)参数
        # 隐藏层1: Xavier正态 + Sigmoid
        nn.init.xavier_normal_(self.linear1.weight)     # 权重矩阵
        nn.init.zeros_(self.linear1.bias)               # 偏置向量

        # 隐藏层2: kaiming正态 + Relu
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


    # 1.2 在 forward()函数中,
    #   完整: 前向传播(正向传播)过程. 该函数在模型预测时, 会自动调用, 无需手动操作.
    def forward(self, x):

        # 隐藏层1计算过程: 加权求和 + 激活函数
        # 分解版
        # x = self.linear1(x)
        # x = torch.sigmoid(x)
        # 合并版
        x = torch.sigmoid(self.linear1(x))

        # 隐藏层2计算过程: 加权求和 + 激活函数
        x = torch.relu(self.linear2(x))

        # 输出层计算过程: 加权求和 + 激活函数
        # dim=-1 的意思是:
        #   最后一维的各个分类(多分类) 概率值相加求和结果为: 1   类似于按: 行统计
        x = torch.softmax(self.output(x), dim=-1)
        return x


# 2. 测试模型对象, 看是否OK.
if __name__ == '__main__':

    # 1. 创建 神经网络模型对象.
    my_model = ModelDemo()
    print(f'my_model: {my_model}')

    # 2. 准备数据集.  5行3列
    my_data = torch.randn(5, 3)
    print(f'my_data: {my_data}')

    # 3. 测试模型对象, 看是否OK.
    output = my_model(my_data)              # 底层会自动调用 forward()函数
    print(f'output: {output}')
    print('-' * 24)

    # 4. 查看模型结构.
    # 参1: (要被查看的)模型对象, 参2: 模型的输入维度(每行样本的特征数量)
    summary(my_model, (3,), 5)
    print('-' * 24)

    # 5. 查看模型参数.
    for name, param in my_model.named_parameters():
        print(f'name: {name}, param: {param}')

