import pickle
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchtext.data.functional import to_map_style_dataset
from model import Sentiment
import torch.optim as optim
from train import train
from sklearn.model_selection import train_test_split


TEXT_MAX_LEN = 15  # 评论最大长度


# 从csv中读取数据
def data_read(path):
    # train_data = pd.read_csv(path + 'train.csv')  # [45806, 4]
    # test_data = pd.read_csv(path + 'test.csv')  # [5090, 3]
    # sample_submission = pd.read_csv(path + 'sample_submission.csv')

    data = pd.read_csv(path + 'train.csv')  # [45806, 4]
    # 取原始数据集第 1 列数据作为 x
    x = data.iloc[:, 1]
    # 取原始数据集第 2 列数据作为 y
    y = data.iloc[:, 3]
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1234)
    # 将 x 和 y 的训练集数据沿着列的方向拼接在一起
    train_data = pd.concat([x_train, y_train], axis=1)  # [34354, 2]
    # 将 x 和 y 的测试集数据沿着列的方向拼接在一起
    test_data = pd.concat([x_test, y_test], axis=1)  # [11452, 2]
    return train_data, test_data


def text_clean(text):
    # 去除特定符号
    cleaned_text = re.sub('[,.!?]+', ' ', text)
    # 替换行内多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # 删除行首尾的空格
    cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text, flags=re.MULTILINE)
    return cleaned_text


# 从数据中提取text列表
def text_df2list(data):
    data_text = data['text']
    text_list = data_text.tolist()
    return text_list


def label_df2list(data):
    data_label = data['label']
    label_list = data_label.tolist()
    return label_list


# 将text转为索引列表
def text_list2index(textlist):
    text_index_list = []
    for sentence in textlist:
        index_list = list()
        for word in sentence.split():
            word_index = word2index_dict[word]
            index_list.append(word_index)
        text_index_list.append(index_list)
    return text_index_list


def label_list2index(label_list):
    label_index_list = []
    for label in label_list:
        label_index_list.append(label2index_dict[label])
    return label_index_list


def text_cut(indexlist, max_len=TEXT_MAX_LEN):
    text_dataset = np.zeros((len(indexlist), max_len))
    for i, sentence in enumerate(indexlist):
        if len(sentence) < max_len:
            # 若评论长度<最长设定，用'0'填补
            text_dataset[i, :len(sentence)] = sentence
        else:
            # 若评论长度>最长设定，截断
            text_dataset[i, :] = sentence[:max_len]
    return text_dataset


# 创建词汇表
def vocab_build(text_list):
    unique_words = set()
    for text in text_list:
        words = text.split()
        unique_words.update(words)  # 加入集合
    return unique_words


def labels_build(label_list):
    return set(label_list)


if __name__ == '__main__':
    # 从文件中读取数据
    train_data, test_data = data_read('data/')
    #print(train_data[:1])
    # text部分数据清洗
    train_data['text'] = train_data['text'].apply(text_clean)
    test_data['text'] = test_data['text'].apply(text_clean)
    #train_data_reset_index = train_data.reset_index(drop=True)  # 重置索引
    #test_data_reset_index = test_data.reset_index(drop=True)
    # 将所有text放入一个list
    train_text_list = text_df2list(train_data)
    test_text_list = text_df2list(test_data)
    print("train: " + str(len(train_text_list)) + "  |test: " + str(len(test_text_list)))

    # 创建标签表（test没有这一项）
    train_label_list = label_df2list(train_data)  # 放入一个list
    test_label_list = label_df2list(test_data)
    all_label_list = train_label_list + test_label_list
    labels = labels_build(all_label_list)  # 创建集合
    labels = list(labels)  # set转list
    #print(label)
    print("Length of labels: " + str(len(labels)))
    # 创建标签字典
    index2label_dict = dict(enumerate(labels, 1))  # {index: label}                                  ----会用到
    label2index_dict = {l: int(i) for i, l in index2label_dict.items()}  # {label: index}            ----会用到
    #print(index2label_dict)
    #print(label2index_dict)
    # 保存标签字典
    with open('./model/index2label.pkl', 'wb') as f:
        pickle.dump(index2label_dict, f)

    # 将train_label_list中的标签映射为labels的索引
    train_label_index = label_list2index(train_label_list)
    train_label_index = np.array(train_label_index)                                                # ----会用到
    test_label_index = label_list2index(test_label_list)
    test_label_index = np.array(test_label_index)                                                  # ----会用到
    #print(train_label_index[0])
    #print(len(train_label_index))

    # 创建词汇表
    all_text_list = train_text_list + test_text_list
    vocab = vocab_build(all_text_list)  # 将train和test合并制作词汇表
    vocab = list(vocab)  # set转list
    #print(vocab)
    print("Length of vocab: " + str(len(vocab)))
    # 创建字典
    index2word_dict = dict(enumerate(vocab, 1))  # {index: word}                                     ----会用到
    word2index_dict = {w: int(i) for i, w in index2word_dict.items()}  # {word: index}               ----会用到
    #print(index2word_dict)
    #print(word2index_dict)
    # 保存单词字典
    with open('./model/word2index.pkl', 'wb') as f:
        pickle.dump(word2index_dict, f)

    # 将text中的单词映射为vocab的索引
    train_text_index = text_list2index(train_text_list)
    test_text_index = text_list2index(test_text_list)
    #print(train_text_list[0])
    #print(train_text_index[0])
    # 将所有评论长度固定为同一大小，单词量不足用'0'填补，超出直接截断
    train_text_dataset = text_cut(train_text_index, max_len=TEXT_MAX_LEN)  # [train_len, TEXT_MAX_LEN]    ----会用到
    test_text_dataset = text_cut(test_text_index, max_len=TEXT_MAX_LEN)  # [test_len, TEXT_MAX_LEN]       ----会用到
    #print(train_text_reset.shape)
    #print(test_text_reset.shape)


    # 创建训练集和测试集中各自的评论张量和标签张量
    # torch.Size([34354, 500])
    train_text_tensor = torch.from_numpy(train_text_dataset)
    # torch.Size([34354])
    train_label_tensor = torch.from_numpy(train_label_index)
    # torch.Size([11452, 500])
    test_text_tensor = torch.from_numpy(test_text_dataset)
    # torch.Size([11452])
    test_label_tensor = torch.from_numpy(test_label_index)

    # 对数据进行封装 (评论，标签)
    train_dataset = TensorDataset(train_text_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_text_tensor, test_label_tensor)

    batch_size = 64
    # Dataloader在每一轮迭代结束后，重新生成索引并将其传入到 to_map_style_dataset 中，就能返回一个个样本
    # shuffle=True 表示打乱样本顺序
    # collate_fn 可以对 Dataloader 生成的 mini-batch 进行后处理
    # pin_memory=True 表示使用 GPU
    # drop_last=True 表示若最后数据量不足 64 个，则将其全部舍弃
    train_loader = DataLoader(to_map_style_dataset(train_dataset), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(to_map_style_dataset(test_dataset), batch_size=batch_size, shuffle=True, drop_last=True)

    # 打印第一个批次来查看结构
    # for batch in train_loader:
    #     print(batch)
    #     break


    # 创建模型
    # 词汇表的大小加上 1
    input_size = len(vocab) + 1
    # 总共有6个类别
    output_size = 46
    embedding_size = 300
    hidden_size = 128
    # LSTM 层数
    num_layers = 2
    # 循环次数
    num_epoch = 20
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    model = Sentiment(input_size, embedding_size, hidden_size, output_size, num_layers, dropout=0.5)
    model.to(device)


    # 定义交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)


    # 训练
    train(model, device, train_loader, test_loader, criterion, optimizer, num_epoch)
