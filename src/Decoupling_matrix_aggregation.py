import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix


def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N = A[0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)

    a = coototensor(csc_matrix(A[0]).tocoo())
    b = coototensor(csc_matrix(A[1]).tocoo())
    c = coototensor(csc_matrix(A[2]).tocoo())
    A_t = torch.stack([a, b, c], dim=2).to_dense()

    A_t = A_t.to(device)

    temp = torch.matmul(A_t, adj_weight)

    temp = torch.squeeze(temp, 2)

    return temp
