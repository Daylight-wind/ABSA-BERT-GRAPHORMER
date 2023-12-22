#!/usr/bin/env python
# coding:utf-8


import torch
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
import numpy as np
import helper.logger as logger

# pytorch nn.init 中实现的初始化函数
INIT_FUNC = {
    'uniform': uniform_,   # 均匀分布
    'kaiming_uniform': kaiming_uniform_,  # 均匀分布生成值
    'xavier_uniform': xavier_uniform_,  # 均匀分布
    'xavier_normal': xavier_normal_,   # 正态分布
    'kaiming_normal': kaiming_normal_      # 正态分布
}


class EmbeddingLayer(torch.nn.Module):    #继承父类
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',  # 卷积核默认初始化方式
                 negative_slope=0, mode_fan='fan_in',  # 负斜率
                 activation_type='linear',
                 ):
        """
        embedding layer
        :param vocab_map: vocab.v2i[filed] -> Dict{Str: Int}
        :param embedding_dim: Int, config.embedding.token.dimension
        :param vocab_name: Str, 'token' or 'label'
        :param config: helper.configure, Configure Object
        :param padding_index: Int, index of padding word
        :param pretrained_dir: Str,  file path for the pretrained embedding file
        :param model_mode: Str, 'TRAIN' or 'EVAL', for initialization
        :param initial_type: Str, initialization type
        :param negative_slope: initialization config
        :param mode_fan: initialization config
        :param activation_type: None
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout']) #其作用是，在 training 模式下，基于伯努利分布抽样，以概率 p 对张量 input 的值随机置0
        self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index)

        # initialize lookup table  ：lookup tabel的作用是：用一张表存储所有训练集中的词语，输入一个句子，查表直接读出句子向量送到后续层做cbow或者skipgram训练。比如我爱，这两个词在表中的位置为2,5。那么训练的时候就是把他们位置的embedding取出来堆在一起
        assert initial_type in INIT_FUNC
        if initial_type.startswith('kaiming'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=negative_slope,
                                                        mode=mode_fan,
                                                        nonlinearity=activation_type)
        elif initial_type.startswith('xavier'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        gain=torch.nn.init.calculate_gain(activation_type))
        else:
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=-0.25,
                                                        b=0.25)  # 生成lookup_tabel   torch.empty用于生成没有初始化的tensor

        if model_mode == 'TRAIN' and config['embedding'][vocab_name]['type'] == 'pretrain' \
                and pretrained_dir is not None and pretrained_dir != '':
            self.load_pretrained(embedding_dim, vocab_map, vocab_name, pretrained_dir)

        if padding_index is not None:
            self.lookup_table[padding_index] = 0.0   # 让第5000个索引的值为0  在查看矩阵的时候，第5000个索引的值确实是0；网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
        self.embedding.weight.data.copy_(self.lookup_table)      # embedding的权重copy等于lookup_table的值
        self.embedding.weight.requires_grad = True
        del self.lookup_table   # 删除这个字段;因为lookup——label是临时变量，lookup_table由两部分组成:不存在词表里的属于初始化的值，存在词表里面的就会将同一个位置一开始初始化的值替换

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, pretrained_dir):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        logger.info('Loading {}-dimension {} embedding from pretrained file: {}'.format(
            embedding_dim, vocab_name, pretrained_dir))
        with open(pretrained_dir, 'r', encoding='utf8') as f_in:
            num_pretrained_vocab = 0
            for line in f_in:
                row = line.rstrip('\n').split(' ')
                if len(row) == 2:
                    assert int(row[1]) == embedding_dim, 'Pretrained dimension %d dismatch the setting %d' \
                                                         % (int(row[1]), embedding_dim)
                    continue
                if row[0] in vocab_map:
                    current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
                    self.lookup_table[vocab_map[row[0]]] = current_embedding
                    num_pretrained_vocab += 1
        logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def forward(self, vocab_id_list):
        """
        :param vocab_id_list: torch.Tensor, (batch_size, max_length)
        :return: embedding -> torch.FloatTensor, (batch_size, max_length, embedding_dim)
        """
        embedding = self.embedding(vocab_id_list)
        return self.dropout(embedding)
