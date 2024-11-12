import numpy as np
import torch


def semeval_scorer(predict_label, true_label, class_num=10):
    # F1 分数是 精确度（precision）和 召回率（recall）的调和平均数，宏平均（Macro F1）是对所有类别的 F1 分数的平均
    import math
    # 确保预测标签和真实标签的数量是相同的。
    # true_label.shape[0] 和 predict_label.shape[0] 都表示标签的数量，若不一致会抛出异常。
    assert true_label.shape[0] == predict_label.shape[0]
    confusion_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float32)
    # 用于存储与类别 i 相关的特殊情况（当预测标签和真实标签不完全一致，但属于同一类时）
    xDIRx = np.zeros(shape=[class_num], dtype=np.float32)
    for i in range(true_label.shape[0]):
        true_idx = math.ceil(true_label[i]/2)   # 后九中关系是对称的。例如，关系 1 和 2 是对称的，关系 3 和 4 是对称的，以此类推。
        predict_idx = math.ceil(predict_label[i]/2)
        if true_label[i] == predict_label[i]:
            confusion_matrix[predict_idx][true_idx] += 1
        else:
            if true_idx == predict_idx:
                xDIRx[predict_idx] += 1
            else:
                confusion_matrix[predict_idx][true_idx] += 1

    col_sum = np.sum(confusion_matrix, axis=0).reshape(-1)  # 计算每一列的总和，表示每个类别被预测的次数。
    row_sum = np.sum(confusion_matrix, axis=1).reshape(-1)  # 计算每一行的总和，表示每个类别的真实标签出现的次数。
    f1 = np.zeros(shape=[class_num], dtype=np.float32)

    for i in range(0, class_num):  # ignore the 'Other'
        try:
            p = float(confusion_matrix[i][i]) / float(col_sum[i] + xDIRx[i])
            r = float(confusion_matrix[i][i]) / float(row_sum[i] + xDIRx[i])
            f1[i] = (2 * p * r / (p + r))
        except:
            pass
    actual_class = 0
    total_f1 = 0.0
    for i in range(1, class_num):
        if f1[i] > 0.0:  # classes that not in the predict label are not considered
            actual_class += 1
            total_f1 += f1[i]
    try:
        macro_f1 = total_f1 / actual_class  # 计算宏平均 F1 分数，即所有类别 F1 分数的平均值
    except:
        macro_f1 = 0.0
    return macro_f1


class Eval(object):
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, criterion, data_loader):
        predict_label = []  # 存储所有预测的标签
        true_label = [] # 存储所有真实标签
        total_loss = 0.0 # 用于累加所有批次的损失
        with torch.no_grad(): # 关闭梯度计算。这通常用于推理阶段，以节省内存和计算资源。
            model.eval() # 将模型设置为评估模式。某些层（如 Dropout 和 BatchNorm）在训练和推理阶段行为不同，调用 eval() 会禁用这些层的训练行为。
            for _, (data, label) in enumerate(data_loader):
                sent_feat = data[0].to(self.device)
                lex_feat = data[1].to(self.device)
                data = (sent_feat, lex_feat)
                label = label.to(self.device)

                scores = model(data)
                loss = criterion(scores, label)
                # 计算当前批次的损失并累计到 total_loss 中。scores.shape[0] 是当前批次的样本数量，用于按样本数量加权损失。
                total_loss += loss.item() * scores.shape[0]

                # 对于每个样本，从 scores 中选择得分最高的类别。scores[:, 1:] 去掉了第一个类别，因为在某些任务中，第一个类别代表 "其他"，而我们只关心后续的类别。
                scores, pred = torch.max(scores[:, 1:], dim=1)
                pred = pred + 1 # 因为模型预测的是从类别 1 开始的索引（假设类别从 1 开始），所以需要加 1。
                # 将得分从 GPU 转移到 CPU，并转换为 NumPy 数组，detach() 表示不计算梯度，numpy() 将其转换为 NumPy 格式。
                scores = scores.cpu().detach().numpy().reshape((-1, 1))
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))

                # During prediction time, a relation is classified as Other
                # only if all actual classes have negative scores.
                # Otherwise, it is classified with the class which has the largest score.
                for i in range(pred.shape[0]):
                    if scores[i][0] < 0:
                        pred[i][0] = 0
                # 将每个批次的预测标签 pred 和真实标签 label 分别追加到 predict_label 和 true_label 列表中
                predict_label.append(pred)
                true_label.append(label)
        # 使用 np.concatenate 将所有批次的预测标签和真实标签合并成一个大的 NumPy 数组。
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        # 计算整个数据集的平均损失。
        eval_loss = total_loss / predict_label.shape[0]
        # 调用一个外部函数 semeval_scorer 来计算 F1 分数。该函数应返回 F1 分数的值
        f1 = semeval_scorer(predict_label, true_label)
        return f1, eval_loss, predict_label

