import torch


class Sentiment(torch.nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(Sentiment, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 将输入的文本进行词嵌入表示的操作
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        # LSTM 的输入维度就是 embedding_size，即 300。batch_first=True 表示将 batch_size 设置成第一个维度。bidirectional=True 表示设置 LSTM 模型为双向的
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        # 全连接层，其中 output_size 就是类别数量，即 6
        self.linear = torch.nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        '''
        x original shape: (seqLen, batch_size, input_size)
        x transform shape (batch_first=True) : (batch_size, seqLen, input_size)
        batch_size：一组数据有多少个，即 64
        seqLen：每个影评列表的大小，即 500
        input_size：每个评论中每个数字的输入特征的维度
        '''
        batch_size = x.size(0)
        # 将输入的影评转换为长整型，形状为 (batch_size, seqLen, input_size)
        x = x.long()
        # 1. 初始化隐藏层中的隐藏状态 h0 (用于传递序列中前一个时间点的信息到下一个时间点)，同时将其转移到与输入影评相同的设备上 (即GPU)
        # 2. h0 的形状为 (num_layers * 2, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        # 1. 初始化隐藏层中的单元状态 c0 (用于在网络中长期传递信息)，同时将其转移到与输入影评相同的设备上 (即GPU)
        # 2. c0 的形状为 (num_layers * 2, batch_size, hidden_size)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        # 输出 x 的形状为 (batch_size, seqLen, embedding_size)
        x = self.embedding(x)
        # 1. 输出 output 的形状为 (batch_size, seqLen, hidden_size)
        # 2. 输出 hn 的形状为 (num_layers * 2, batch_size, hidden_size)
        # 3. 输出 cn 的形状为 (num_layers * 2, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        # 1. 选择最后一个时间步的输出
        # 2. 输入 output 的形状变为 (batch_size, hidden_size)
        # 3. 输出 output 的形状变为 (batch_size, output_size)
        output = output[:, -1]
        # 输出 output 的形状为 (batch_size, output_size)，表示每个序列属于目标类别的概率
        output = self.linear(output)
        return output
