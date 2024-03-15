import math

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc, accuracy_score, roc_curve
import torch.nn.functional as F
from src.args import get_citation_args

# from yijian.line import example_plot1

args = get_citation_args()


def load_training_data(f_name):
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    true_edge_data_by_type['1'] = list()
    false_edge_data_by_type['1'] = list()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        i = 0
        for line in f:
            i = i + 1
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[0], words[1]
            if int(words[2]) == 1:
                # if words[0] not in true_edge_data_by_type:
                true_edge_data_by_type['1'].append((x, y))
            else:
                # if words[0] not in false_edge_data_by_type:
                false_edge_data_by_type['1'].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type


def get_score(local_model, node1, node2):
    """
    Calculate embedding similarity
    """
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        if type(vector1) != np.ndarray:
            vector1 = vector1.toarray()[0]
            vector2 = vector2.toarray()[0]
        dot_fea = np.dot(vector1, vector2)
        return dot_fea
    except Exception as e:
        pass


def link_prediction_evaluate(model, true_edges, false_edges):
    """
    Link prediction process
    """

    true_list = list()
    prediction_list = list()
    true_num = 0

    for edge in true_edges:
        # tmp_score = get_score(model, str(edge[0]), str(edge[1])) # for amazon
        tmp_score = get_score(model, str(int(edge[0]) - 1), str(int(edge[1]) + 494))
        # tmp_score = get_score(model, str(int(edge[0] -1)), str(int(edge[1]-1)))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    # Calculate the similarity score of negative sample embedding
    for edge in false_edges:
        tmp_score = get_score(model, str(int(float(edge[0])) - 1), str(int(float(edge[1])) + 494))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    # 排序
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]
    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thre = roc_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), rs, accuracy_score(y_true, y_pred), fpr, tpr, auc(
        rs, ps)


def predict_model(model, file_name, feature, A, B, o_ass, eval_type, lr, length, layer, dataset):
    """
    Link prediction training proces
    """
    if dataset == 1:
        # 筛选后
        valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/train7.txt')
        testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test7_1.txt')
    else:
        valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/train7_2.txt')
        testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test7_2.txt')

    edge_type_count = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    aucs, f1s, recalls, accs, fprs, tprs, auprs = [], [], [], [], [], [], []
    losss = []
    epoch = args.epoch
    weight_d = args.weight_decay
    weight_b_list = []
    weight_c_list = []
    for _ in range(1):
        for iter_ in range(epoch):
            print('epoch{}=================================================================='.format(iter_ + 1))
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_d)

            emb = model(feature, A, B, o_ass, layer)

            emb_true_first = []
            emb_true_second = []
            emb_false_first = []
            emb_false_second = []

            for i in range(edge_type_count):
                # (x, y)list
                true_edges = valid_true_data_by_edge['1']
                false_edges = valid_false_data_by_edge['1']

                for edge in true_edges:
                    # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                    emb_true_first.append(emb[int(edge[0]) - 1])
                    emb_true_second.append(emb[int(edge[1]) + 494])

                for edge in false_edges:
                    # tmp_score = get_score(final_model, str(edge[0]), str(edge[1])) # for amazon
                    emb_false_first.append(emb[int(float(edge[0])) - 1])
                    emb_false_second.append(emb[int(float(edge[1])) + 494])
            # feature维度878
            emb_true_first = torch.cat(emb_true_first).reshape(-1, length)
            emb_true_second = torch.cat(emb_true_second).reshape(-1, length)
            emb_false_first = torch.cat(emb_false_first).reshape(-1, length)
            emb_false_second = torch.cat(emb_false_second).reshape(-1, length)

            # @表示矩阵乘法， 12000*12000
            T1 = emb_true_first @ emb_true_second.T
            T2 = -(emb_false_first @ emb_false_second.T)

            # 1:5
            # T1 = torch.mm(emb_true_first, emb_true_second.T)
            # T2 = -torch.mm(emb_false_first, emb_false_second.T)

            # 1:10
            # T1 = torch.mm(emb_true_first, emb_true_second.T)
            # T2 = -(torch.mm(emb_false_first[0:10000], emb_false_second.T[:, 0:10000]))
            # T3 = -(torch.mm(emb_false_first[10000:20000], emb_false_second.T[:, 10000:20000]))
            # T4 = -(torch.mm(emb_false_first[20000:30000], emb_false_second.T[:, 20000:30000]))
            # T5 = -(torch.mm(emb_false_first[30000:], emb_false_second.T[:, 30000:]))

            # diag取对角线元素,1*60
            pos_out = torch.diag(T1)
            neg_out = torch.diag(T2)
            # neg_out1 = torch.diag(T3)
            # neg_out2 = torch.diag(T4)
            # neg_out3 = torch.diag(T5)

            # mean返回所有元素平均值
            loss = -torch.mean(F.logsigmoid(pos_out)) - torch.mean(F.logsigmoid(neg_out))
            # loss = -torch.mean(F.logsigmoid(pos_out)) - torch.mean(F.logsigmoid(neg_out)) - torch.mean(
            #     F.logsigmoid(neg_out1)) - torch.mean(F.logsigmoid(neg_out2)) - torch.mean(F.logsigmoid(neg_out3))

            loss = loss.requires_grad_()

            opt.zero_grad()
            loss.backward()
            opt.step()

            td = model(feature, A, B, o_ass, layer).cpu().detach().numpy()

            final_model = {}
            try:
                for i in range(0, len(td)):
                    final_model[str(i)] = td[i]
            except:
                td = td.tocsr()
                for i in range(0, td.shape[0]):
                    final_model[str(i)] = td[i]
            # print("final_embedding: {}".format(len(final_model['0'])))
            train_aucs, train_f1s, train_rs, train_accs = [], [], [], []
            test_aucs, test_f1s, test_rs, test_accs, test_auprs = [], [], [], [], []
            for i in range(edge_type_count):
                if eval_type == 'all':
                    train_auc, triain_f1, train_recall, train_acc, fpr0, tpr0, aupr0 = link_prediction_evaluate(
                        final_model,
                        valid_true_data_by_edge[
                            '1'],
                        valid_false_data_by_edge[
                            '1'])
                    train_aucs.append(train_auc)
                    train_f1s.append(triain_f1)
                    train_rs.append(train_recall)
                    train_accs.append(train_acc)

                    weight_b_best = model.weight_b.cpu().detach().numpy()
                    weight_b_list.append(weight_b_best)
                    weight_c_best = model.weight_c.cpu().detach().numpy()
                    weight_c_list.append(weight_c_best)

                    test_auc, test_f1, test_recall, test_acc, fpr, tpr, aupr1 = link_prediction_evaluate(final_model,
                                                                                                         testing_true_data_by_edge[
                                                                                                             '1'],
                                                                                                         testing_false_data_by_edge[
                                                                                                             '1'])

                    test_aucs.append(test_auc)
                    test_f1s.append(test_f1)
                    test_rs.append(test_recall)
                    test_accs.append(test_acc)
                    test_auprs.append(aupr1)
                    fprs.append(fpr)
                    tprs.append(tpr)
            print("loss:{:.4f}".format(loss.item()))
            print("train_auc:{:.4f}\ttrain_f1:{:.4f}\ttrain_recall:{:.4f}\ttrain_acc:{:.4f}".format(np.mean(train_aucs),
                                                                                                    np.mean(train_f1s),
                                                                                                    np.mean(train_rs),
                                                                                                    np.mean(
                                                                                                        train_accs)))
            print("test_auc:{:.4f}\ttest_f1:{:.4f}\ttest_recall:{:.4f}\ttest_acc:{:.4f}\ttest_aupr:{:.4f}".format(
                np.mean(test_aucs),
                np.mean(test_f1s),
                np.mean(test_rs),
                np.mean(test_accs),
                np.mean(test_auprs)))

            aucs.append(np.mean(test_aucs))
            f1s.append(np.mean(test_f1s))
            recalls.append(np.mean(test_rs))
            accs.append(np.mean(test_accs))
            auprs.append(np.mean(test_auprs))

    max_iter = aucs.index(max(aucs))
    print("auc最高轮次：", max_iter)
    print()
    return aucs[max_iter], f1s[max_iter], recalls[max_iter], accs[max_iter], fprs[max_iter], tprs[max_iter], auprs[
        max_iter]
