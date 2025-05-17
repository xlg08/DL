"""
案例:
    演示词嵌入层的相关代码.

循环神经网络:
    全称是: Recurrent neural network, 包含 词嵌入层 + 循环层 + 全连接层.
    其中:
        词嵌入层: 把词转成 词向量矩阵, 即: word_to_vector -> word2vec -> word_vec
"""

# 导包
import torch
import torch.nn as nn
import jieba  # pip install jieba 需要安装一下.

# 1. 创建文本(语料, 1句话)
# text = '我喜欢北京天安门'
text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'

# 2. 使用jieba进行分词.
words = jieba.lcut(text)
print(f'words: {words}')

# 3. 创建词嵌入层.
# 参1: 词表大小(即: 词表中单词的数量),  参2: 词向量维度.
embed = nn.Embedding(num_embeddings=len(words), embedding_dim=20)
print(f'embed: {embed}')  # 19个词, 每个词用4维的向量表示(1行, 4列)

# 4. 获取每个词的下标索引.5
for i, word in enumerate(words):
    # i: word这个单词对应的 下标(索引)
    # word: words词表中, 某个具体的词.
    # print(i, word)

    # 5. 将词索引 转成  词向量.
    word_vec = embed(torch.tensor(i))
    print(f'word_vec: {word_vec}')
