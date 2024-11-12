import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class RC2DNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec    # 词向量，用预训练的词嵌入
        self.class_num = class_num  # 类别数

        # 超参数和其他配置
        self.max_len = config.max_len   # 输入文本的最大长度
        self.word_dim = config.word_dim # 词向量维度
        self.pos_dim = config.pos_dim   # 位置嵌入维度
        self.pos_dis = config.pos_dis   # 位置偏移范围（用于位置嵌入）

        self.dropout_value = config.dropout # dropout概率
        self.filter_num = config.filter_num # 卷积核的数量
        self.window = config.window # 卷积窗口的大小
        self.hidden_size = config.hidden_size   # 隐藏层大小

        self.dim = self.word_dim + 2 * self.pos_dim # 输入嵌入的总维度

        # 网络结构和操作
        self.word_embedding = nn.Embedding.from_pretrained(embeddings=self.word_vec, freeze=False, )
        # 词嵌入层，使用预训练的词向量，且在训练过程中允许更新这些词向量
        self.pos1_embedding = nn.Embedding(num_embeddings=2 * self.pos_dis + 3, embedding_dim=self.pos_dim)
        # 位置嵌入层1，表示词语相对于句子起始位置的偏移量
        self.pos2_embedding = nn.Embedding(num_embeddings=2 * self.pos_dis + 3, embedding_dim=self.pos_dim)
        # 位置嵌入层2，表示词语相对于句子结束位置的偏移量

        self.conv = nn.Conv2d(
            in_channels=1,  # 输入通道数，1表示单通道（每个词的表示是一个向量）
            out_channels=self.filter_num,   # 输出通道数，即卷积核的数量
            kernel_size=(self.window, self.dim),    # 卷积核的大小，窗口大小为self.window，输入维度为self.dim
            stride=(1, 1),  # 步幅
            bias=False, # 不使用偏置
            padding=(1, 0),  # 填充（为了保证输出的尺寸合适）
            padding_mode='zeros'    # 填充模式，使用零填充
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))  # 最大池化层，池化窗口大小为(max_len, 1)，目的是进行句子级的池化
        self.tanh = nn.Tanh()   # Tanh激活函数
        self.dropout = nn.Dropout(self.dropout_value)   # Dropout层，用于防止过拟合
        self.linear = nn.Linear(in_features=self.filter_num, out_features=self.hidden_size, bias=False) # 全连接层，用于映射到隐藏层
        self.dense = nn.Linear(in_features=self.hidden_size + 6 * self.word_dim, out_features=self.class_num, bias=False)   # 最后的分类层，输出类别数

        # initialize weight
        init.xavier_normal_(self.pos1_embedding.weight) # 使用Xavier初始化位置嵌入1的权重
        init.xavier_normal_(self.pos2_embedding.weight) # 使用Xavier初始化位置嵌入2的权重
        init.xavier_normal_(self.conv.weight)   # 使用Xavier初始化卷积层的权重
        # init.constant_(self.conv.bias, 0.)    # 如果使用偏置，这行代码会将其初始化为0
        init.xavier_normal_(self.linear.weight) # 使用Xavier初始化全连接层的权重
        # init.constant_(self.linear.bias, 0.)   如果使用偏置，这行代码会将其初始化为0
        init.xavier_normal_(self.dense.weight)  # 使用Xavier初始化输出层的权重
        # init.constant_(self.dense.bias, 0.)   # 如果使用偏置，这行代码会将其初始化为0

    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B*L*word_dim：将token转换为词向量
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim：将pos1（词相对于起始位置的偏移）转换为位置嵌入
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim：将pos2（词相对于结束位置的偏移）转换为位置嵌入
        emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        # 将词嵌入和两个位置嵌入在最后一个维度拼接在一起，得到每个词的最终表示
        return emb  # 返回拼接后的嵌入，形状为B*L*D，其中D=word_dim+2*pos_dim

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D：增加一个维度以适应卷积操作
        conv = self.conv(emb)  # B*C*L*1：进行卷积操作，得到特征图

        # 通过mask去除PAD的影响
        conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L：将卷积结果reshape为B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L：增加一个维度以适应卷积
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L：将mask扩展为与卷积结果相同的形状
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))   # B*C*L：使用-∞替换掉PAD部分
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1：恢复维度，准备进行池化
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1：进行最大池化操作，压缩到句子级别
        pool = pool.view(-1, self.filter_num)  # B*C：将池化后的结果flatten为B*C
        return pool

    def forward(self, data):
        token = data[0][:, 0, :].view(-1, self.max_len) # B*L：获取token序列
        pos1 = data[0][:, 1, :].view(-1, self.max_len)  # B*L：获取pos1序列
        pos2 = data[0][:, 2, :].view(-1, self.max_len)  # B*L：获取pos2序列
        mask = data[0][:, 3, :].view(-1, self.max_len)  # B*L：获取mask，标记padding部分
        lexical = data[1].view(-1, 6)   # B*6：获取词汇信息
        lexical_emb = self.word_embedding(lexical)  # B*6*word_dim：对词汇信息进行词向量映射
        lexical_emb = lexical_emb.view(-1, self.word_dim * 6)   # B*(6*word_dim)：flatten词汇信息
        emb = self.encoder_layer(token, pos1, pos2) # B*L*D：获取输入序列的嵌入表示
        emb = self.dropout(emb) # 应用dropout
        conv = self.conv_layer(emb, mask)   # B*C*L*1：进行卷积操作
        pool = self.single_maxpool_layer(conv)  # B*C：进行最大池化操作，获取句子的最重要特征
        sentence_feature = self.linear(pool)    # B*hidden_size：将卷积池化后的特征映射到隐藏层
        sentence_feature = self.tanh(sentence_feature)  # 应用Tanh激活函数
        sentence_feature = self.dropout(sentence_feature)   # 应用dropout
        features = torch.cat((lexical_emb, sentence_feature), 1)    # B*(6*word_dim + hidden_size)：拼接词汇信息和句子特征
        logits = self.dense(features)    # B*class_num：通过全连接层计算分类得分
        return logits    # 返回分类得分

