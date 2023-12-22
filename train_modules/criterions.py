#!/usr/bin/env python
# coding: utf-8

import torch

from helper.utils import get_hierarchy_relations


class ClassificationLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 recursive_constraint=True,):  #todo:这里的损失函数用什么？himatch里面这里的损失函数用的是loss_type="bce"
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(ClassificationLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()  # 这个loss将一个Sigmoid层和BCELoss结合在一个类里。这个版本比使用一个普通的Sigmoid后接一个BCELoss更数值稳定
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map)   # 递归关系 而不返回label_tree  self.recursive_relation是父子之间的关系
        self.recursive_penalty = recursive_penalty
        self.recursive_constraint = recursive_constraint

    def _recursive_regularization(self, params, device):
        """
        recursive regularization: constraint on the parameters of classifier among parent and children
        :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        :param device: torch.device -> config.train.device_setting.device
        :return: loss -> torch.FloatTensor, ()
        """
        rec_reg = 0.0
        for i in range(len(params)):
            if i not in self.recursive_relation.keys():
                continue
            child_list = self.recursive_relation[i]
            if not child_list:
                continue
            child_list = torch.tensor(child_list).to(device)
            child_params = torch.index_select(params, 0, child_list)  # torch.index_select:通过选择索引然后去得到想要的tensor,针对比较长的tensor torch.index_select(tensor,维度，选择的index）
            parent_params = torch.index_select(params, 0, torch.tensor(i).to(device))
            parent_params = parent_params.repeat(child_params.shape[0], 1)
            _diff = parent_params - child_params
            diff = _diff.view(_diff.shape[0], -1)
            rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
        return rec_reg

    def forward(self, logits, targets, recursive_params):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        if self.recursive_constraint:
            loss = self.loss_fn(logits, targets) + \
                   self.recursive_penalty * self._recursive_regularization(recursive_params,
                                                                           device)
        else:
            loss = self.loss_fn(logits, targets)
        return loss
