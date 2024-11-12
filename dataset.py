import os
import torch
import numpy as np
import json
import re
# import nltk
# nltk.download('punkt_tab') # when you first run this code, you need to download the punkt package
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from config import Config

class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """
    def __init__(self, config):
        self.path_word = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

    # 从预训练的词嵌入中提取词汇表中的词向量，并添加特殊字符（未知词和填充词）
    def trim_from_pre_embedding(self, vocab):
        word2id = dict()    # word to wordID mapping
        word_vec = {}    # word to word vector mapping
        trim_word_vec = list()  # list to store trimmed word vectors
        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue    # skip lines that do not match the expected dimension
                word_vec[line[0]] = np.asarray(line[1:], dtype=np.float32)  # store word vectors
        for word in vocab:
            word2id[word] = len(word2id)    # assign a unique ID to each word
            if word in word_vec:
                trim_word_vec.append(word_vec[word])     # use pre-trained vector if available
            else:
                trim_word_vec.append(np.random.uniform(-1, 1, self.word_dim))    # use random vector if not available
        # 添加特殊字符
        if ("*UNKNOWN*" not in word2id):
            word2id['*UNKNOWN*'] = len(word2id) # add unknown token
            unk_emb = np.random.uniform(-1, 1, self.word_dim)
            trim_word_vec.append(unk_emb)
        if ("PAD" not in word2id):
            word2id['PAD'] = len(word2id)   # add padding token
            pad_emb = np.zeros(self.word_dim)
            trim_word_vec.append(unk_emb)
        trim_word_vec = np.array(trim_word_vec)
        trim_word_vec = trim_word_vec.astype(np.float32).reshape(-1, self.word_dim)
        return word2id, torch.from_numpy(trim_word_vec)

    # 加载预训练的词嵌入，并添加未知词和填充词。
    def load_embedding(self):
        word2id = dict()  # word to wordID mapping
        word_vec = list()  # wordID to word embedding or list to store word vectors
        word2id['PAD'] = len(word2id)  # add padding token

        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue    # skip lines that do not match the expected dimension
                word2id[line[0]] = len(word2id) # assign a unique ID to each word
                word_vec.append(np.asarray(line[1:], dtype=np.float32)) # store word vectors
        if ("*UNKNOWN*" not in word2id):
            word2id['*UNKNOWN*'] = len(word2id) # add unknown token
            unk_emb = np.random.uniform(-1, 1, self.word_dim)
            word_vec.append(unk_emb)
        pad_emb = np.zeros([1, self.word_dim], dtype=np.float32)  # initialize padding vector as zero
        word_vec = np.concatenate((pad_emb, word_vec), axis=0)  # concatenate padding vector with word vectors
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec


class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir # 数据目录

    # 从文件中加载关系数据，并构建关系与ID的映射
    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')  # 关系文件路径
        rel2id = {} # 关系到ID的映射
        id2rel = {} # ID到关系的映射
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()   # 读取关系和ID
                id_d = int(id_s)    # 将ID转换为整数
                rel2id[relation] = id_d # 将关系映射到ID
                id2rel[id_d] = relation # 将ID映射到关系
        return rel2id, id2rel, len(rel2id)  # 返回关系映射、ID映射和关系数量

    def get_relation(self):
        return self.__load_relation()   # 调用__load_relation方法并返回结果

# 用于处理与关系抽取任务相关的数据。核心功能包括加载数据、处理句子、计算实体位置的相对关系，并生成适用于模型输入的格式。
class SemEvalDateset(Dataset):
    # 初始化方法，接收文件名、关系映射、词汇映射和配置信息
    def __init__(self, filename, rel2id, word2id, config):
        self.filename = filename     # 文件名
        self.rel2id = rel2id    # 关系到ID的映射
        self.word2id = word2id  # 词汇到ID的映射
        self.max_len = config.max_len   # 最大句子长度
        self.pos_dis = config.pos_dis   # 位置距离阈值
        self.data_dir = config.data_dir # 数据目录路径
        self.dataset, self.label = self.__load_data()   # 加载数据和标签

    # 根据位置计算相对位置索引
    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0    # 小于负阈值返回0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1  # 在阈值内，返回位置偏移
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2     # 大于正阈值，返回最大值

    # 计算某个位置相对于实体的位置
    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x - entity_pos[0])  # 在实体前面
        elif x > entity_pos[1]:
            return self.__get_pos_index(x - entity_pos[1])  # 在实体后面
        else:
            return self.__get_pos_index(0)  # 在实体内部

    # 将句子中的词符号化，同时处理实体的相对位置
    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
        """
        mask = [1] * len(sentence)  # 初始化掩码，所有位置设为1
        # 根据实体的位置调整掩码
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1] + 1):
                mask[i] = 2 # 实体 1 区域
            for i in range(e2_pos[1] + 1, len(sentence)):
                mask[i] = 3 # 实体 2 区域
        else:
            for i in range(e2_pos[0], e1_pos[1] + 1):
                mask[i] = 2 # 实体 2 区域
            for i in range(e1_pos[1] + 1, len(sentence)):
                mask[i] = 3 # 实体 1 区域

        words = []  # 存储词ID
        pos1 = []   # 存储实体1的相对位置
        pos2 = []   # 存储实体2的相对位置
        length = min(self.max_len, len(sentence))   # 取最大句长与实际句子长度中的最小值
        mask = mask[:length]    # 裁剪掩码到句子长度

        # 填充词ID和相对位置
        for i in range(length):
            words.append(self.word2id.get(sentence[i], self.word2id['*UNKNOWN*']))  # 词ID
            # 上面这行确保将句子中的每个单词转换为其对应的单词 ID，如果在词汇表中找不到单词，它将使用未知标记的 ID。
            pos1.append(self.__get_relative_pos(i, e1_pos)) # 实体1的相对位置
            pos2.append(self.__get_relative_pos(i, e2_pos)) # 实体2的相对位置

        # 如果句子长度不足最大长度，填充PAD
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # PAD掩码为0
                words.append(self.word2id['PAD'])   # PAD词ID
                pos1.append(self.__get_relative_pos(i, e1_pos)) # 实体1的相对位置
                pos2.append(self.__get_relative_pos(i, e2_pos)) # 实体2的相对位置
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)    # 将数据转为NumPy数组
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))   # 重塑为符合模型输入的形状
        return unit

    # 计算实体的上下文特征
    def _lexical_feature(self, e1_idx, e2_idx, sent):
        def _entity_context(e_idx, sentence):
            '''返回实体前后词的上下文，格式为 [w(e-1), w(e), w(e+1)]'''
            context = []
            context.append(sentence[e_idx])  # 当前词
            if e_idx >= 1:
                context.append(sentence[e_idx - 1]) # 前一个词
            else:
                context.append(sentence[e_idx]) # 边界情况

            if e_idx < len(sentence) - 1:
                context.append(sentence[e_idx + 1]) # 后一个词
            else:
                context.append(sentence[e_idx]) # 边界情况
            return context

        # print(e1_idx,sent)
        # 获取两个实体的上下文
        context1 = _entity_context(e1_idx[0], sent)
        context2 = _entity_context(e2_idx[0], sent)
        # ignore WordNet hypernyms in paper
        lexical = context1 + context2   # 拼接两个实体的上下文
        lexical_ids = [self.word2id.get(word, self.word2id['*UNKNOWN*']) for word in lexical]   # 转换为词ID
        lexical_ids = np.asarray(lexical_ids, dtype=np.int64)   # 转换为NumPy数组
        return np.reshape(lexical_ids, newshape=(1, 6)) # 重塑为模型输入形状

    # 加载数据文件，解析句子、标签和实体位置
    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename) # 构造数据文件路径
        data = [] # 存储数据
        labels = [] # 存储标签
        with open(path_data_file, 'r', encoding='utf-8') as fr: # 打开数据文件
            for line in fr:
                line = json.loads(line.strip()) # 解析每一行JSON
                label = line['relation']    # 获取关系标签
                sentence = line['sentence'] # 获取句子
                e1_pos = (line['subj_start'], line['subj_end']) # 获取实体1的位置
                e2_pos = (line['obj_start'], line['obj_end'])   # 获取实体2的位置
                label_idx = self.rel2id[label]   # 将标签转换为ID
                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)   # 处理句子
                lexical = self._lexical_feature(e1_pos, e2_pos, sentence)   # 获取实体的上下文特征
                temp = (one_sentence, lexical)  # 存储句子和上下文特征
                data.append(temp)   # 添加到数据列表
                # data.append(one_sentence)
                labels.append(label_idx)    # 添加到标签列表
        return data, labels # 返回处理后的数据和标签

    # 根据索引获取数据项
    def __getitem__(self, index):
        data = self.dataset[index]  # 获取数据
        label = self.label[index]   # 获取标签
        return data, label  # 返回数据和标签

    # 获取数据集的大小
    def __len__(self):
        return len(self.label)


# 这是 SemEvalDataLoader 类的代码，它用于加载 SemEvalDataset 的数据并处理数据加载的过程，主要是批量化、数据预处理和转换为模型输入所需的格式。
class SemEvalDataLoader(object):
    # 初始化方法，接收关系映射、词汇映射和配置信息
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.config = config

    # 定义如何批量加载数据
    def collate_fn(self, batch):
        data, label = zip(*batch)  # 解包批次中的数据和标签
        data = list(data)   # 转换为列表
        label = list(label) # 转换为列表
        # 合并所有句子特征数据（即将batch中所有数据的第一个部分连接成一个大矩阵）;句子特征（x[0]）
        sentence_feat = torch.from_numpy(np.concatenate([x[0] for x in data], axis=0))
        # 合并所有词汇特征数据（即将batch中所有数据的第二个部分连接成一个大矩阵）;词汇特征（x[1]）
        lexical_feat = torch.from_numpy(np.concatenate([x[1] for x in data], axis=0))
        # 转换标签为Tensor
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return (sentence_feat, lexical_feat), label # 返回句子特征、词汇特征和标签

    # 定义如何从文件中加载数据
    def __get_data(self, filename, shuffle=False):
        # 使用 SemEvalDataset 类加载数据集
        dataset = SemEvalDateset(filename, self.rel2id, self.word2id, self.config)
        # 使用 DataLoader 将数据集转换为可以批量加载的格式
        loader = DataLoader(
            dataset=dataset,    # 数据集
            batch_size=self.config.batch_size,  # 批次大小
            shuffle=shuffle,    # 是否打乱数据
            num_workers=2,      # 使用的工作线程数
            collate_fn=self.collate_fn    # 定义如何组合一个batch的数据
        )
        return loader   # 返回数据加载器

    # 获取训练数据加载器
    def get_train(self):
        return self.__get_data('train.json', shuffle=True)

    # 获取验证数据加载器
    def get_dev(self):
        return self.__get_data('test.json', shuffle=False)

    # 获取测试数据加载器
    def get_test(self):
        return self.__get_data('test.json', shuffle=False)


class processor(object):
    # 初始化方法
    def __init__(self):
        pass

    # 从句子中提取实体并标记它们的位置
    def search_entity(self, sentence):
        # 使用正则表达式找到 <e1> 和 </e1> 标签之间的文本
        e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
        # 使用正则表达式找到 <e2> 和 </e2> 标签之间的文本
        e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
        # 在原句中将 <e1> 和 <e2> 标签周围增加空格，以便稍后进行分词
        sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
        sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
        # 使用 word_tokenize 进行分词处理
        sentence = word_tokenize(sentence)
        # 将分词后的列表转换回字符串，以便进行下一步处理
        sentence = ' '.join(sentence)
        # 恢复 <e1>, <e2>, </e1>, 和 </e2> 标签的正常格式
        sentence = sentence.replace('< e1 >', '<e1>')
        sentence = sentence.replace('< e2 >', '<e2>')
        sentence = sentence.replace('< /e1 >', '</e1>')
        sentence = sentence.replace('< /e2 >', '</e2>')
        # 这行代码将句子按空格分割成单词列表。可能是在确保所有标签都被恢复成正常格式之后进行最终的分词操作。
        sentence = sentence.split()

        # 语句检查句子中是否包含了 <e1>、<e2>、</e1> 和 </e2> 标签。
        # 这是一个检查步骤，用来确保标签没有在处理过程中丢失。如果某个标签没有出现在句子中，程序会抛出一个 AssertionError 错误。
        assert '<e1>' in sentence
        assert '<e2>' in sentence
        assert '</e1>' in sentence
        assert '</e2>' in sentence

        # 初始化实体位置变量
        subj_start = subj_end = obj_start = obj_end = 0
        pure_sentence = []  # 用于存储不包含标签的纯文本句子
        for i, word in enumerate(sentence):
            if '<e1>' == word:
                subj_start = len(pure_sentence) # 设置第一个实体的开始位置
                continue
            if '</e1>' == word:
                subj_end = len(pure_sentence) - 1   # 设置第一个实体的结束位置
                continue
            if '<e2>' == word:
                obj_start = len(pure_sentence)  # 设置第二个实体的开始位置
                continue
            if '</e2>' == word:
                obj_end = len(pure_sentence) - 1    # 设置第二个实体的结束位置
                continue
            pure_sentence.append(word)  # 将当前词添加到纯文本句子中
        # 返回提取的实体、位置索引和纯文本句子
        return e1, e2, subj_start, subj_end, obj_start, obj_end, pure_sentence

    # 转换文件格式，将原始数据转换成包含实体位置信息的 JSON 格式
    def convert(self, path_src, path_des):
        # 读取源文件数据
        with open(path_src, 'r', encoding='utf-8') as fr:
            data = fr.readlines()
        # 打开目标文件写入转换后的数据
        with open(path_des, 'w', encoding='utf-8') as fw:
            # 遍历每个句子（每4行表示一个完整的数据条目）
            for i in range(0, len(data), 4):
                id_s, sentence = data[i].strip().split('\t')     # 获取句子ID和句子内容
                sentence = sentence[1:-1]   # 移除句子外侧的双引号
                e1, e2, subj_start, subj_end, obj_start, obj_end, sentence = self.search_entity(sentence)   # 使用 search_entity 方法提取实体和纯句子
                # 创建字典，包含所有需要的信息
                meta = dict(
                    id=id_s,
                    relation=data[i + 1].strip(),   # 关系类型
                    head=e1,    # 第一个实体
                    tail=e2,    # 第二个实体
                    subj_start=subj_start,  # 第一个实体的开始位置
                    subj_end=subj_end,  # 第一个实体的结束位置
                    obj_start=obj_start,    # 第二个实体的开始位置
                    obj_end=obj_end,    # 第二个实体的结束位置
                    sentence=sentence,  # 纯文本句子（去除了标签）
                    comment=data[i + 2].strip()[8:] # 注释内容，去除前8个字符
                )
                json.dump(meta, fw, ensure_ascii=False)   # 将字典写入目标文件，格式为 JSON
                fw.write('\n')  # 每条记录后换行


class VocabGenerator(object):
    # 初始化方法，接受训练和测试数据的文件路径
    def __init__(self, train_path, test_path):
        self.train_path = train_path    # 训练数据文件路径
        self.test_path = test_path   # 测试数据文件路径

    # 获取词汇表的方法
    def get_vocab(self):
        vocab = {}  # 创建一个空字典用于存储词汇
        # 打开训练数据文件并遍历每一行
        with open(self.train_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                # 将 JSON 格式的行数据解析为字典
                line = json.loads(line.strip())
                sentence = line['sentence'] # 获取句子，句子是一个单词列表
                # 遍历句子中的每个单词，将单词添加到词汇表字典中
                for word in sentence:
                    vocab[word] = 1 # 使用字典键保存唯一单词
        # 打开测试数据文件并遍历每一行
        with open(self.test_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                # 将 JSON 格式的行数据解析为字典
                line = json.loads(line.strip())
                sentence = line['sentence'] # 获取句子，句子是一个单词列表
                # 遍历句子中的每个单词，将单词添加到词汇表字典中
                for word in sentence:
                    vocab[word] = 1 # 使用字典键保存唯一单词
        # 返回词汇表中的所有唯一单词
        return vocab.keys()


if __name__ == '__main__':
    # path_train = 'dataset/train_file.txt'
    # path_test = 'dataset/test_file.txt'
    # processor1 = processor()
    # processor1.convert(path_train, 'dataset/train.json')
    # processor1.convert(path_test, 'dataset/test.json')
    # vocab = VocabGenerator('dataset/train.json', 'dataset/test.json').get_vocab()
    # print(len(vocab)) # 打印词表大小 value = 25692

    config = Config()   # 初始化配置对象
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()    # 加载单词到ID的映射字典和词向量
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()   # 加载关系ID映射、ID到关系的映射、和类别数量
    loader = SemEvalDataLoader(rel2id, word2id, config) # 创建数据加载器对象，将关系ID字典、单词ID字典和配置对象传入
    test_loader = loader.get_train()    # 使用数据加载器获取训练数据，返回的是训练数据的 DataLoader 对象
    # 初始化用于存储位置编码的最小值和最大值的变量
    min_v, max_v = float('inf'), -float('inf')  # 将初始值设置为极限，方便后续更新
    # 遍历训练数据，获取批次索引、数据和标签
    for step, (data, label) in enumerate(test_loader):
        # 可以在这里打印数据类型和形状以供调试
        print(type(data), len(data))  # 查看 data 是什么类型和长度
        sentence_feat, lexical_feat = data  # 解包 data; data是个元组，包含句子特征和词汇特征 （tuple）
        print(sentence_feat.shape, lexical_feat.shape)  # 查看 sentence_feat 和 lexical_feat 的形状： 前者是句子特征，后者是词汇特征
        # break

        # 提取数据中第二维度 (即位置编码 pos1) 并重新调整其形状为 [batch_size, max_len]
        pos1 = sentence_feat[:, 1, :].view(-1, config.max_len)
        print(pos1.shape)
        print(pos1)
        print(lexical_feat)
        print(label)
        # 提取数据中第三维度 (即位置编码 pos2) 并调整形状为 [batch_size, max_len]
        pos2 = sentence_feat[:, 2, :].view(-1, config.max_len)

        # 提取数据中的掩码 mask 信息，并调整形状为 [batch_size, max_len]
        mask = sentence_feat[:, 3, :].view(-1, config.max_len)

        break
        # 获取 pos1 中的最小值并更新 min_v
        min_v = min(min_v, torch.min(pos1).item())
        # 获取 pos1 中的最大值并更新 max_v
        max_v = max(max_v, torch.max(pos1).item())
        # 获取 pos2 中的最小值并更新 min_v
        min_v = min(min_v, torch.min(pos2).item())
        # 获取 pos2 中的最大值并更新 max_v
        max_v = max(max_v, torch.max(pos2).item())
    # 打印数据中 pos1 和 pos2 的最小和最大值
    # print(min_v, max_v)

