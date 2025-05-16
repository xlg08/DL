'''

    # 快捷键 ctrl + r： 更换部分内容


'''

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch.nn as nn


# 准备数据集
def create_dataset():
    data = pd.read_csv("../../data/手机价格预测.csv")

    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # print(x)
    # print(y)

    x = x.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

    train_dataset = TensorDataset(torch.Tensor(x_train.values), torch.Tensor(y_train.values))
    test_dataset = TensorDataset(torch.Tensor(x_test.values), torch.Tensor(y_test.values))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y_train))


class model(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forword(self):
        pass


# 模型训练

# 模型测试


if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    # print(f"train_dataset：", train_dataset)
    # print(f"test_dataset：", test_dataset)
    print(f"input_dim：", input_dim)
    print(f"output_dim：", output_dim)
