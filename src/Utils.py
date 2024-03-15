import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from src.Model import MHGCN


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_our_data(dataset_str, cuda=True):
    """
    Load our Networks Datasets
    Avoid awful code
    """
    # result
    data = loadmat('data/' + dataset_str + '.mat')

    node_features = data['feature']
    features = csr_matrix(node_features)  # 转化为稀疏矩阵，那么打印出来的只有非零的下标以及它的值

    try:
        # 特征转换为int类型
        features = features.astype(np.int16)
    except:
        pass
    # 类型转换 numpy ->  torch
    features = torch.FloatTensor(np.array(features.todense())).float()

    if cuda:
        features = features.cuda()

    return features


def get_model(model_opt, nfeat, A, B, o_ass, nhid, out, dropout=0, cuda=True, stdv=1 / 72, layer=2):
    """
     Model selection
    """

    if model_opt == "MGCNSS":
        model = MHGCN(nfeat=nfeat,  # 878
                      nhid=nhid,  # 384
                      out=out,  # 256
                      dropout=dropout,
                      stdv=stdv,
                      layer=layer)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
