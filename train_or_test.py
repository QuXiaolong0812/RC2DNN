import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from dataset import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader, VocabGenerator
from model import RC2DNN
from evaluate import Eval


def print_result(predict_label, id2rel, start_idx=8001):
    # 打开一个文件 'script/predicted_result.txt' 以写入模式 ('w') 打开，如果文件不存在将会创建它
    # 使用utf-8编码，确保能够处理中文字符等
    with open('script/predicted_result.txt', 'w', encoding='utf-8') as fw:
        # 遍历所有预测标签
        for i in range(0, predict_label.shape[0]):
            # 写入每一行数据，格式是 '索引  标签'
            # start_idx + i 用来生成连续的索引值
            # id2rel[int(predict_label[i])] 将预测标签的整数值映射为实际的标签名称
            fw.write('{}\t{}\n'.format(start_idx + i, id2rel[int(predict_label[i])]))


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader
    # 下面的两行代码被注释掉了，但它们用于控制优化器中参数的权重衰减策略
    # weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    # no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    # parameters = [{'params': weight_decay_list},
    #               {'params': no_decay_list, 'weight_decay': 0.}]

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.L2_decay)
    # 使用Adam优化器来优化模型的参数，学习率和L2正则化衰减参数从config中读取

    print(model)    # 打印模型结构
    print('traning model parameters:')  # 打印模型的所有参数及其形状
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)     # 创建一个Eval对象，用于模型评估
    min_f1 = -float('inf')   # 初始化最小F1分数为负无穷，用于模型保存时选择最好的模型
    for epoch in range(1, config.epoch + 1):     # 迭代多个训练周期（epoch）
        for step, (data, label) in enumerate(train_loader): # 遍历训练数据集
            model.train()   # 将模型设置为训练模式，启用dropout等训练特性
            # 将数据和标签移动到配置的设备（如GPU或CPU）
            sent_feat = data[0].to(config.device)   # 句子特征
            lex_feat = data[1].to(config.device)    # 词汇特征
            data = (sent_feat, lex_feat)    # 将它们打包为一个元组
            label = label.to(config.device) # 标签也移动到配置的设备

            optimizer.zero_grad()   # 清除梯度缓存
            logits = model(data)    # 前向传播，得到模型的预测输出
            loss = criterion(logits, label) # 计算损失
            loss.backward() # 反向传播计算梯度
            optimizer.step()    # 使用优化器更新模型参数

        # 在训练集和验证集上进行评估
        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)   # 评估训练集上的损失
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader)  # 评估验证集上的损失和F1分数

        # 打印训练损失、验证损失和验证集上的F1分数
        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1), end=' ')
        # 如果当前epoch的F1分数比最小的F1分数还高，就保存模型
        if f1 > min_f1:
            min_f1 = f1
            torch.save(model.state_dict(), os.path.join(config.model_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print() # 如果F1分数没有提升，不保存模型


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('start test ...')  # 打印测试开始的提示信息
    _, _, test_loader = loader  # 从loader中获取测试数据加载器
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'model.pkl')))
    # 加载保存的最优模型权重，模型文件保存在config.model_dir路径下，文件名为'model.pkl'
    eval_tool = Eval(config)    # 创建评估工具对象，用于评估模型的性能
    f1, test_loss, predict_label = eval_tool.evaluate(model, criterion, test_loader)
    # 使用评估工具对测试集进行评估，返回F1分数、测试损失和预测标签
    # 打印测试集的损失和F1分数
    print('test_loss: %.3f | micro f1 on test:  %.4f' % (test_loss, f1))
    return predict_label    # 返回预测标签


if __name__ == '__main__':  # 如果这是主程序入口（确保只有在直接运行时执行）
    config = Config()   # 创建一个配置对象，配置模型、训练等超参数
    print('--------------------------------------')
    print('some config:')   # 打印配置信息
    config.print_config()   # 打印配置内容

    print('--------------------------------------')
    print('start to load data ...') # 开始加载数据
    vocab = VocabGenerator('dataset/train.json', 'dataset/test.json').get_vocab()
    # 创建VocabGenerator对象，读取训练集和测试集文件，生成词汇表
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    # 加载词向量，返回词到ID的映射（word2id）和词向量矩阵（word_vec）
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    # 加载关系数据，返回关系到ID的映射（rel2id），ID到关系的映射（id2rel）以及类别数量（class_num）
    loader = SemEvalDataLoader(rel2id, word2id, config) # 创建数据加载器对象，处理数据

    train_loader, dev_loader = None, None   # 初始化训练集和验证集加载器
    if config.mode == 1:  # 如果是训练模式
        train_loader = loader.get_train()   # 获取训练集的加载器
        dev_loader = loader.get_dev()   # 获取验证集的加载器
    test_loader = loader.get_test() # 获取测试集的加载器
    loader = [train_loader, dev_loader, test_loader]    # 将训练、验证和测试加载器放入一个列表
    print('finish!')    # 打印“加载数据完成”

    print('--------------------------------------')
    model = RC2DNN(word_vec=word_vec, class_num=class_num, config=config)
    # 创建模型对象RC2DNN，传入词向量、类别数和配置对象
    model = model.to(config.device)
    # 将模型移动到指定的设备（如GPU或CPU）
    criterion = nn.CrossEntropyLoss()
    # 定义损失函数，使用交叉熵损失函数，适用于分类问题

    if config.mode == 1:   # 如果是训练模式
        train(model, criterion, loader, config) # 调用训练函数进行训练
    predict_label = test(model, criterion, loader, config)  # 调用测试函数进行测试并预测标签
    print_result(predict_label, id2rel) # 打印预测结果，将预测标签转回原始的关系名称
