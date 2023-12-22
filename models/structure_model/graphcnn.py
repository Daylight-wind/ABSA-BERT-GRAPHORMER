#!/usr/bin/env python
# coding:utf-8
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class HierarchyGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix,
                 out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root=None,
                 hierarchical_label_dict=None,
                 label_trees=None):
        """
        Graph Convolutional Network variant for hierarchy structure
        original GCN paper:
                Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
                    arXiv preprint arXiv:1609.02907.
        :param num_nodes: int, N
        :param in_matrix: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_matrix: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param layers: int, the number of layers <- config.structure_encoder.num_layer
        :param time_step: int, the number of time steps <- config.structure_encoder.time_step
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        """
        super(HierarchyGCN, self).__init__()
        self.model = nn.ModuleList()  # ModuleList 可以存储多个 model，传统的方法，一个model 就要写一个 forward ，但是如果将它们存到一个 ModuleList 的话，就可以使用一个 forward
        self.model.append(
            HierarchyGCNModule(num_nodes,
                               in_matrix, out_matrix,
                               in_dim,
                               dropout,
                               device))    # append方法往里面添加模型

    def forward(self, label):
        return self.model[0](label)


class HierarchyGCNModule(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_adj, out_adj,
                 in_dim, dropout, device, in_arc=True, out_arc=True,
                 self_loop=True):
        """
        module of Hierarchy-GCN
        :param num_nodes: int, N
        :param in_adj: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_adj: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        :param in_arc: Boolean, True
        :param out_arc: Boolean, True
        :param self_loop: Boolean, True
        """

        super(HierarchyGCNModule, self).__init__()
        self.self_loop = self_loop
        self.out_arc = out_arc
        self.in_arc = in_arc
        self.device = device
        assert in_arc or out_arc
        #  bottom-up child sum
        in_prob = in_adj  # 把子节点到父节点的【0，1】矩阵赋值给她
        self.adj_matrix = Parameter(torch.Tensor(in_prob))  # Parameter方法是对需要梯度计算进行参数更新的数据集成的一类类方法；nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        self.edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))
        self.gate_weight = Parameter(torch.Tensor(in_dim, 1))
        self.bias_gate = Parameter(torch.Tensor(num_nodes, 1))
        self.activation = nn.ReLU()
        self.origin_adj = torch.Tensor(np.where(in_adj <= 0, in_adj, 1.0)).to(device)
        # top-down: parent to child
        self.out_adj_matrix = Parameter(torch.Tensor(out_adj))
        self.out_edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))
        self.out_gate_weight = Parameter(torch.Tensor(in_dim, 1))
        self.out_bias_gate = Parameter(torch.Tensor(num_nodes, 1))
        self.loop_gate = Parameter(torch.Tensor(in_dim, 1))
        self.dropout = nn.Dropout(p=dropout)   # p指的是概率，目前是0.05，Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算
        self.reset_parameters()

    def reset_parameters(self):   # 用于重置参数
        """
        initialize parameters
        """
        for param in [self.gate_weight, self.loop_gate, self.out_gate_weight]:
            nn.init.xavier_uniform_(param)      # 均匀分布初始化，循环三次，因为上面是列表
        for param in [self.edge_bias, self.out_edge_bias, self.bias_gate]:
            nn.init.zeros_(param)  # 全部置为0

    def forward(self, inputs):
        """
        :param inputs: torch.FloatTensor, (batch_size, N, in_dim)
        :return: message_ -> torch.FloatTensor (batch_size, N, in_dim)
        """
        h_ = inputs  # batch, N, in_dim
        message_ = torch.zeros_like(h_).to(self.device)  # batch, N, in_dim

        h_in_ = torch.matmul(self.origin_adj * self.adj_matrix, h_)   # 矩阵乘法 # batch, N, in_dim
        in_ = h_in_ + self.edge_bias
        in_ = in_
        # N,1,dim
        in_gate_ = torch.matmul(h_, self.gate_weight)
        # N, 1
        in_gate_ = in_gate_ + self.bias_gate
        in_ = in_ * F.sigmoid(in_gate_)
        in_ = self.dropout(in_)
        message_ += in_  # batch, N, in_dim

        h_output_ = torch.matmul(self.origin_adj.transpose(0, 1) * self.out_adj_matrix, h_)   # 转置
        out_ = h_output_ + self.out_edge_bias
        out_gate_ = torch.matmul(h_, self.out_gate_weight)
        out_gate_ = out_gate_ + self.out_bias_gate
        out_ = out_ * F.sigmoid(out_gate_)
        out_ = self.dropout(out_)
        message_ += out_

        loop_gate = torch.matmul(h_, self.loop_gate)
        loop_ = h_ * F.sigmoid(loop_gate)
        loop_ = self.dropout(loop_)
        message_ += loop_

        return self.activation(message_)
