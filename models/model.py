#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
import torch
import torch.nn as nn
from transformers import BertModel

from models.structure_model.graph import GraphEncoder

from models.structure_model.structure_encoder import StructureEncoder
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.multi_label_attention import HiAGMLA
from models.text_feature_propagation import HiAGMTP
from models.origin import Classifier
DATAFLOW_TYPE = {
    'HiMatch-bert': 'serial',
    'HiAGM-LA': 'parallel',
    'Origin': 'origin'

}


class HiAGM(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN'):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_type: Str, ('HiAGM-TP' for the serial variant of text propagation,
                                 'HiAGM-LA' for the parallel variant of multi-label soft attention,
                                 'Origin' without hierarchy-aware module)
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(HiAGM, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device
        self.transformation_dropout = nn.Dropout(p=config.model.linear_transformation.dropout)

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.model_type = config.model.type
        self.transformation = nn.Linear(config.model.linear_transformation.text_dimension,
                                        len(self.label_map) * config.model.linear_transformation.node_dimension)
        # 此处加入了bert模型作为文本编码器
        if "bert" in self.model_type:
            self.bert = BertModel.from_pretrained("bert-base-chinese")
            self.bert_dropout = nn.Dropout(0.1)


        self.dataflow_type = DATAFLOW_TYPE[model_type]   # 数据流
        # classifier
        self.linear = nn.Linear(len(self.label_map) * config.embedding.label.dimension,
                                len(self.label_map))

        # dropout
        self.dropout = nn.Dropout(p=config.model.classifier.dropout)



        # # 标签编码器
        # self.structure_encoder = StructureEncoder(config=config,
        #                                           label_map=vocab.v2i['label'],
        #                                           device=self.device,
        #                                           graph_model_type=config.structure_encoder.type)

        # data_path等于文件的目录
        # ToDo：标签编码器
        tau = 1
        threshold = 0.02
        layer = 1
        graph = 1
        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=None, threshold=threshold, tau=tau)

        # self.hiagm = HiAGMTP(config=config,
        #                          device=self.device,
        #                          graph_model=self.structure_encoder,
        #                          label_map=self.label_map)


    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.hiagm.parameters()})
        return params

    def forward(self, inputs):   # 这里原来的实参是batch
        """
        forward pass of the overall architecture
        :param inputs: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return: 
        """

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)


        batch= inputs  #token：32*300，label：32*72

        outputs = self.bert(batch['input_ids'].to(self.config.train.device_setting.device),
                            batch['segment_ids'].to(self.config.train.device_setting.device),
                            batch['input_mask'].to(self.config.train.device_setting.device))
        pooled_output = outputs[1]  #cls(batch*768)
        token_output = self.bert_dropout(pooled_output) # (batch,768)

        attention_mask=torch.ones(((self.config.train.batch_size, 768)), device=self.config.train.device_setting.device)

        text_feature=token_output #cls作为文本特征，batch*768
       #  text_feature = torch.cat(token_output, 1)
        text_feature = text_feature.view(text_feature.shape[0], -1)
        text_feature = self.transformation_dropout(self.transformation(text_feature))
        text_feature = text_feature.view(text_feature.shape[0],
                                         len(self.label_map),
                                         self.config.model.linear_transformation.node_dimension)

        text_feature = self.graph_encoder(text_feature,attention_mask
                                          )# 这是把文本映射为标签节点向量，text_feature：32*72*768；attention_mask：32*768然后作为输入，经过图transformers的输出（8，130，768）

        logits = self.dropout(self.linear(text_feature.view(text_feature.shape[0], -1)))
        # logits= self.hiagm(
        #     token_output) # 这段代码要改，

        return logits    #（32，72） （batch，num_labels）



    def get_embedding(self, inputs):   # 这一行代码可能没啥用
        batch, mode = inputs[0], inputs[1]
        if "bert" in self.model_type:
            outputs = self.bert(batch['input_ids'].to(self.config.train.device_setting.device),
                                batch['segment_ids'].to(self.config.train.device_setting.device),
                                batch['input_mask'].to(self.config.train.device_setting.device))
            pooled_output = outputs[1]
            pooled_output = self.bert_dropout(pooled_output)
        else:
            embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
            seq_len = batch['token_len']
            token_output = self.text_encoder(embedding, seq_len)
            pooled_output = token_output.view(token_output.shape[0], -1)
        return pooled_output

        # #todo：原来的forword内容
        # embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))

        #
        # # get the length of sequences for dynamic rnn, (batch_size, 1)
        # seq_len = batch['token_len']
        #
        # token_output = self.text_encoder(embedding, seq_len)
        #
        # logits = self.hiagm(token_output)
        #
        # return logits
