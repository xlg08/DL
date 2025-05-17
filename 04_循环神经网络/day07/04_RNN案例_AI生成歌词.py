"""
案例:
    基于杰伦的歌词训练模型, 然后给定 提示词 和 生成的长度, 由训练好的模型来 生成歌词.

步骤:
    1. 构建词表.
    2. 创建数据集.
    3. 搭建网络模型.
    4. 模型训练.
    5. 模型测试.
"""

# 导包
import torch            # PyTorch深度学习框架, 主要做 张量处理的.
import jieba            # 分词包.
from torch.utils.data import DataLoader         # 数据加载器
import torch.nn as nn           # 构建网络模型
import torch.optim as optim  # 优化器
import time  # 时间包


# 1. 构建词表.
#   词表：将数据进行分词，然后给每一个词分配一个唯一的编号，用于送入词嵌入层获取每个词的词向量。
def build_vocab():

    # 1. 定义变量, 分别记录: 去重后所有的词, 每行文本分词结果.
    #   定义的unique_words变量为：存放对歌曲句子进行分词以后再进行去重之后的词
    #   定义的all_words变量为：存放以每句歌曲为单位存放(一个列表)，对每句歌曲分词之后的词
    unique_words, all_words = [], []

    # 2. 遍历数据集, 获取到每行文本.
    for line in open('../../data/jaychou_lyrics.txt', 'r', encoding='utf-8'):

        # 2.1 通过jieba分词器, 对每行文本进行分词.
        words = jieba.lcut(line)
        # 2.2 把每行的分词结果, 添加到: all_words中.
        # 格式: [['想要', '有', '直升机', '\n'], ['想要', '和', '你', '飞到', '宇宙', '去', '\n'].....]
        all_words.append(words)
        # 2.3 遍历每行文本的每个词, 如果不在unique_words中, 则添加到unique_words中.
        for word in words:
            if word not in unique_words:        # 进行去重
                unique_words.append(word)

    # 3. 统计语料中词的个数.（语料中存放的是去重之后的词）
    word_count = len(unique_words)      # 统计去重之后的词的个数，歌曲分词后有多少不重复的词
    # print(f'语料中不同词的个数: {word_count}')       # 5703

    # 4. 构建词表, 词表是字典类型: key是词, value是词的索引.
    #   词表就是：以 语料中的词为键，词的索引 为值的一个字典
    # 格式: {'想要': 0, '有': 1, '直升机': 2, '\n': 3, ...... '好人好事': 5700, '冠军': 5701, '要大卖': 5702}
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    # print(f'word_to_index: {word_to_index}')

    # 5. 把歌词中的文本, 用词索引替换.
    corpus_idx = []         # 存储歌词文本的词索引.    词索引本质上就是词表中的值

    # 6. 遍历每一行的分词结果.
    for words in all_words:  # words -> 该行的分词结果.
        # 6.1 定义变量, 记录: (当前这句话)的词索引列表.
        tmp = []
        # 6.2 获取每一行的词, 并获取相应的索引.
        for word in words:      # word -> 该行的每一个词.
            # 6.3 把该词对应的索引, 添加到tmp中.
            tmp.append(word_to_index[word])         # 根据词表找到每一个词的索引
        # 6.4 每行词之间, 添加空格隔开.
        tmp.append(word_to_index[' '])      # 2 一句话之后加一个空格，用于区分一句话，也因为在原歌曲中本身就是空格隔开的话
        # 6.5 把tmp添加到all_index中.
        corpus_idx.extend(tmp)

    # 7. 返回结果: (1) 所有去重后词的列表; (2) 词表; (3) (去重后)词的个数; (4) 词索引列表: 所有文本都用 词索引替换后的结果. .
    return unique_words, word_to_index, word_count, corpus_idx


# 2. 创建数据集.
class LyricsDataset(torch.utils.data.Dataset):
    # 1. 初始化词索引, 词个数等...
    def __init__(self, corpus_idx, num_chars):
        # 1.1 文档数据中的词索引.
        self.corpus_idx = corpus_idx
        # 1.2 每个句子中的词的个数
        self.num_chars = num_chars
        # 1.3 文档数据中词的数量, 不去重.
        self.word_count = len(corpus_idx)
        # 1.4. 计算句子的数量.
        self.number = self.word_count // self.num_chars

    # 2. 当使用 len(obj)时, 会自动调用 __len__()这个魔法方法.
    def __len__(self):
        return self.number  # 返回句子数量.

    # 3. 当使用 obj[index]时, 会自动调用 __getitem__()这个魔法方法.
    def __getitem__(self, idx):
        # 3.1 确保索引的 start值, 是在合法范围内的.
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        # 3.2 计算当前样本的结束索引.
        end = start + self.num_chars
        # 3.3 获取 start ~ end范围的元素, 作为: 输入的x
        x = self.corpus_idx[start:end]
        # 3.4 获取到 start + 1 ~ end + 1范围的元素, 作为: 输出的y
        y = self.corpus_idx[start + 1:end + 1]
        # 3.5 返回输入x和输出y, 记得封装成张量.
        return torch.tensor(x), torch.tensor(y)


# 3. 搭建网络模型.
class TextGenerator(nn.Module):
    # 1. 初始化方法.
    def __init__(self, unique_word_count):  # 去重后词的数量 -> 5703
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 构建词嵌入层.
        # 参1: 输入的词的数量(5703), 参2: 词向量的维度(128), 会得到: 5703行 * 128列
        self.ebd = nn.Embedding(unique_word_count, 128)

        # 1.3 构建RNN层.
        # 参1: 词向量维度(128), 参2: 隐藏层的维度(256), 参3: 网络层数(1)
        self.rnn = nn.RNN(128, 256, 1)

        # 1.4 构建输出层(全连接层)
        self.out = nn.Linear(256, unique_word_count)

    # 2. 前向传播(正向传播)
    def forward(self, inputs, hidden):
        # 2.1 初始化 词嵌入层处理.
        embed = self.ebd(inputs)
        # 2.2 rnn层 处理
        # 参1: (句子数量, 句子长度, 词向量维度)  -> (句子长度, 句子数量, 隐藏层的维度)
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)
        # 2.3 全连接层处理.
        # 因为: 全连接层处理的都是 二维的数据, 所以, 需要将 output 的维度进行转换.
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 2.4 返回处理结果.
        return output, hidden

    # 3. 隐藏层的初始化方法, 用于初始化: h0
    def init_hidden(self, bs):  # batch_size: 批次大小
        # 参1: 网络层数, 参2: 批次大小, 参3: 隐藏层的维度.
        return torch.zeros(1, bs, 256)


# 4. 模型训练.
def train_model():

    # 1. 构建词表.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 2. 获取数据集
    lyrics = LyricsDataset(corpus_idx, num_chars=32)
    # 3. 创建模型对象
    model = TextGenerator(unique_word_count)
    # 4. 创建数据加载器.
    lyrics_dataloader = DataLoader(lyrics, batch_size=5, shuffle=True)
    # 5. 定义多分类交叉熵损失.
    criterion = nn.CrossEntropyLoss()
    # 6. 定义优化器.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 7. 具体的训练动作.
    epochs = 10
    for epoch_idx in range(epochs):

        # 7.1 定义变量, 记录: 本轮开始训练的时间, 迭代次数, 总损失.
        start, iter_num, total_loss = time.time(), 0, 0.0

        # 7.2 遍历数据集, 获取到具体的数据, 底层会自动调用: LyricsDataset# __getitem__函数
        for x, y in lyrics_dataloader:
            # 7.3 获取隐藏层.
            hidden = model.init_hidden(bs=5)

            # 7.4 模型计算.
            output, hidden = model(x, hidden)

            # 7.5 计算损失.
            # y的形状: (batch, seq_len, 词向量维度) -> 转成一维的向量

            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            # print(f"y的形状：{y.shape}")        # torch.Size([160])

            loss = criterion(output, y)

            # 7.6 梯度清零 + 反向传播 + 参数优化.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 7.7 计算训练的累计损失, 累计迭代次数
            total_loss += loss.item()
            iter_num += 1

            # break

        # break

        # 7.8 打印训练信息.
        print(f'epoch: {epoch_idx + 1}, 耗时: {time.time() - start:.2f}, 训练损失: {total_loss / iter_num:.4f}')

    # 8. 保存模型.
    torch.save(model.state_dict(), '../../model/text_model.pth')


# 5. 模型测试.
def evaluate_model(start_word, sentence_length):

    # 1. 构建词表.
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()

    # 2. 创建模型对象
    model = TextGenerator(unique_word_count)  # 去重后单词总数: 5703

    # 3. 加载模型参数.
    model.load_state_dict(torch.load('../../model/text_model.pth'))

    # 4. 获取隐藏层.
    hidden = model.init_hidden(1)

    # 5. 把起始词 -> 索引.
    word_idx = word_to_index[start_word]

    # 6. 定义列表, 存放: 产生(预测)的词的索引.
    generate_sentence = [word_idx]

    # 7. 遍历句子长度, 获取到每一个词.
    for _ in range(sentence_length):

        # 8. 模型预测.
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 9. 获取预测结果.
        word_idx = torch.argmax(output)
        # 10. 将预测结果添加到列表中.
        generate_sentence.append(word_idx)

    # 11. 基于上述的歌词索引 -> 转为歌词, 并打印.
    for idx in generate_sentence:
        print(unique_words[idx], end='')


# 6. 测试.
if __name__ == '__main__':
    # # 1. 构建词表.
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print(f'去重后 词的数量: {word_count}')
    # print(f'去重后 所有的词: {unique_words}')
    # print(f'词表 -> 每个词及其对应的索引: {word_to_index}')
    # print(f'词索引列表 -> 当前文档中每个词对应的索引: {corpus_idx}')

    # 2. 构建数据集.
    dataset = LyricsDataset(corpus_idx, num_chars=5)  # 1句话5个词.
    # print(f'句子的数量: {len(dataset)}')

    # x, y = dataset[0]   # 第1句话的5个词, x: 输入, y: 输出
    # print(f'输入值: {x}')
    # print(f'输出值: {y}')

    # 3. 构建 循环神经网络模型.
    model = TextGenerator(word_count)

    # 4. 模型训练.
    train_model()

    # 5. 模型测试.
    # evaluate_model('', 50)
