import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch import tensor
from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge

from src.args import get_citation_args

args = get_citation_args()


class GraphConvolution(Module):

    # in_features 878， out 56
    def __init__(self, in_features, out_features, stdv, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  #
        # print(self.weight)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  #
        else:
            self.register_parameter('bias', None)
        # 随机初始化参数
        self.reset_parameters(stdv)

    def reset_parameters(self, stdv):

        print("stdv:", stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, adj):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(input.shape)
        try:
            input1 = input1.float()
        except:
            pass

        support = torch.mm(input1.to(device), self.weight)

        output = torch.mm(adj.to(torch.double), support.to(torch.double))
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """
    A Two-layer GCN.
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class MHGCN(nn.Module):

    #  nfeat:878, out 256
    def __init__(self, nfeat, nhid, out, dropout, stdv, layer):
        super(MHGCN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        print("layer:", layer)
        self.gc1 = GraphConvolution(nfeat, out, stdv)
        self.gc2 = GraphConvolution(out, out, stdv)
        self.gc3 = GraphConvolution(out, out, stdv)
        # self.gc3 = GraphConvolution(out, out)
        self.gc4 = GraphConvolution(out, out, stdv)
        self.gc5 = GraphConvolution(out, out, stdv)
        self.dropout = dropout

        self.weight_meta_path = Parameter(torch.FloatTensor(878, 878), requires_grad=True)  #

        self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        self.weight_c = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # 卷积后的融合系数，先跑两层卷积
        self.weight_f = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f1 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f2 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f3 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.weight_f4 = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        # uniform_函数给self.weight_b 随机赋值，使其满足均匀分布U（a，b）
        torch.nn.init.uniform_(self.weight_b, a=0.08, b=0.12)
        torch.nn.init.uniform_(self.weight_c, a=0.08, b=0.12)

        torch.nn.init.uniform_(self.weight_f, a=0.90, b=1.05)
        torch.nn.init.uniform_(self.weight_f1, a=0.10, b=0.30)
        torch.nn.init.uniform_(self.weight_f2, a=0.40, b=0.50)
        torch.nn.init.uniform_(self.weight_f3, a=0.40, b=0.50)
        torch.nn.init.uniform_(self.weight_f4, a=0.40, b=0.50)
        self.w_out = nn.ModuleList([nn.Linear(2 * out, out), nn.Linear(out, 128), nn.Linear(128, 64)])
        self.w_interaction = nn.Linear(64, 2)

    # A三个26128×26128矩阵，feature稀疏矩阵
    def forward(self, feature, A, B, o_ass, layer, use_relu=True):
        # 子网络融合
        final_A = adj_matrix_weight_merge(A, self.weight_b)
        # 383, 383
        final_B = adj_matrix_weight_merge(B, self.weight_c)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        index = torch.where(final_A < 0.015)
        final_A[index[0], index[1]] = 0

        index_B = torch.where(final_B < 0.15)
        final_B[index_B[0], index_B[1]] = 0

        all0 = np.zeros((383, 383))
        all01 = np.zeros((495, 495))
        # 按列合并，即列不变
        adjacency_matrix1 = torch.cat((torch.as_tensor(o_ass).to(device), torch.as_tensor(final_B).to(device)), 0)
        adjacency_matrix2 = torch.cat(
            (torch.as_tensor(final_A).to(device), torch.fliplr(torch.flipud(torch.as_tensor(o_ass.T).to(device)))), 0)
        # A: 878*878
        final_matrix = torch.cat((adjacency_matrix1, adjacency_matrix2), 1)

        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # Output of single-layer GCN
        U1 = self.gc1(feature, final_matrix)
        if layer >= 2:
            U2 = self.gc2(U1, final_matrix)
        if layer >= 3:
            U3 = self.gc3(U2, final_matrix)
        if layer >= 4:
            U4 = self.gc4(U3, final_matrix)
        if layer >= 5:
            U5 = self.gc5(U4, final_matrix)

        # 合并
        if layer == 1:
            H = U1 * self.weight_f
        if layer == 2:
            H = U1 * self.weight_f + U2 * self.weight_f1
        if layer == 3:
            H = U1 * self.weight_f + U2 * self.weight_f1 + U3 * self.weight_f2
        if layer == 4:
            H = U1 * self.weight_f + U2 * self.weight_f1 + U3 * self.weight_f2 + U4 * self.weight_f3
        if layer == 5:
            H = U1 * self.weight_f + U2 * self.weight_f1 + U3 * self.weight_f2 + U4 * self.weight_f3 + U5 * self.weight_f4


        return H


