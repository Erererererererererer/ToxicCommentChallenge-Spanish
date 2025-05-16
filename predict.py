import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import csv

import beginning
from model import Sentiment
from train import predict


def data_read(path):
    test_data = pd.read_csv(path + 'test.csv')  # [5090, 3]
    return test_data


# 将text转为索引列表
def text_list2index(textlist):
    text_index_list = []
    for sentence in textlist:
        index_list = list()
        for word in sentence.split():
            if word not in word2index_dict.keys():
                index_list.append(0)
            else:
                word_index = word2index_dict[word]
                index_list.append(word_index)
        text_index_list.append(index_list)
    return text_index_list


def id_df2list(data):
    data_id = data['id']
    id_list = data_id.tolist()
    return id_list


if __name__ == '__main__':
    # 从文件中读取数据
    test_data = data_read('data/')
    #print(train_data[:1])
    # text部分数据清洗
    test_data['text'] = test_data['text'].apply(beginning.text_clean)
    # 将所有text放入一个list
    test_text_list = beginning.text_df2list(test_data)
    print("predict: " + str(len(test_text_list)))

    # 加载标签字典
    with open('./model/index2label.pkl', 'rb') as f:
        index2label_dict = pickle.load(f)
    # 加载单词字典
    with open('./model/word2index.pkl', 'rb') as f:
        word2index_dict = pickle.load(f)

    # 将text中的单词映射为vocab的索引
    test_text_index = text_list2index(test_text_list)
    # print(test_text_list[0])
    # print(test_text_index[0])
    # 将所有评论长度固定为同一大小，单词量不足用'0'填补，超出直接截断
    test_text_dataset = beginning.text_cut(test_text_index, max_len=beginning.TEXT_MAX_LEN)  # [test_len, TEXT_MAX_LEN]       ----会用到
    # print(test_text_reset.shape)

    # 创建测试集中的评论张量
    # torch.Size([5090, 500])
    test_text_tensor = torch.from_numpy(test_text_dataset)

    # 对数据进行封装 (评论)
    test_dataset = TensorDataset(test_text_tensor)

    batch_size = 64
    # Dataloader在每一轮迭代结束后，重新生成索引并将其传入到 to_map_style_dataset 中，就能返回一个个样本
    # shuffle=True 表示打乱样本顺序
    # collate_fn 可以对 Dataloader 生成的 mini-batch 进行后处理
    # pin_memory=True 表示使用 GPU
    # drop_last=True 表示若最后数据量不足 64 个，则将其全部舍弃
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 打印第一个批次来查看结构
    # for batch in test_loader:
    #     print(batch)
    #     break

    # 创建模型
    # 词汇表的大小加上 1
    input_size = len(word2index_dict) + 1
    # 总共有6个类别
    output_size = 46
    embedding_size = 300
    hidden_size = 128
    # LSTM 层数
    num_layers = 2
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))

    model = Sentiment(input_size, embedding_size, hidden_size, output_size, num_layers, dropout=0.5)
    model.load_state_dict(torch.load('./model/sample_model-19.pt'))
    model.to(device)

    target_list = predict(model, test_loader, device)
    #print(target)


    # 生成预测文档
    id_list = id_df2list(test_data)
    # print(len(id_list))
    # print(len(target_list))
    # print(id_list)
    # print(target_list)
    predict_header = ['id', 'label']
    with open('predict.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(predict_header)
        for i in range(len(id_list)):
            record = [id_list[i], index2label_dict[target_list[i]]]
            writer.writerow(record)
