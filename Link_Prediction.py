import matplotlib.font_manager
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import time
import argparse

from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.link_prediction_evaluate import predict_model
import matplotlib.pyplot as plt

args = get_citation_args()
parser = argparse.ArgumentParser()
myfont = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\CaeciliaLTStd-Roman.otf")

args.dataset = 'result'
net_path = r"data/result.mat"
savepath = r'data/hcg_embedding'
eval_name = r'data'
file_name = r'data/train'
eval_type = 'all'

mat = loadmat(net_path)
train = mat['A']
train1 = mat['B']

try:
    feature = mat['full_feature']  # false
except:
    try:
        feature = mat['feature']  # 返回26128×4635数组
    except:
        try:
            feature = mat['features']
        except:
            feature = mat['node_feature']

feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

A = train
B = train1

node_matching = False
o_ass = mat['raw_association']
aus, f1s, recalls, accs, fprs, tprs, auprs = [], [], [], [], [], [], []

for i in [1]:
    print(i)
    t1 = time.time()
    model = get_model(args.model, 878, A, B, o_ass, args.hidden, 256, args.dropout, False, stdv=1 / 72, layer=2)
    auc, f1, recall, acc, fpr, tpr, aupr = predict_model(model, file_name, feature, A, B, o_ass, eval_type, 0.0005,
                                                         256, 2, 1)
    t2 = time.time()
    print('running time:{}'.format(t2-t1))
    aus.append(auc)
    f1s.append(f1)
    recalls.append(recall)
    accs.append(acc)
    fprs.append(fpr)
    tprs.append(tpr)
    auprs.append(aupr)
    print('Test auc: {:.10f}, F1: {:.10f}, aupr: {:.10f}, acc:{:.10f}, recall:{:.10f}'.format(auc, f1, aupr, acc,
                                                                                              recall))
print(aus)
print(f1s)
print(accs)
print(auprs)
print(recalls)

