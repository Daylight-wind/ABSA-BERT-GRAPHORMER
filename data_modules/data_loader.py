#!/usr/bin/env python
# coding:utf-8

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data_modules.collator import Collator
from data_modules.dataset import ClassificationDataset


def data_loaders(config, vocab, data={'train': None, 'val': None, 'test': None}, bert_tokenizer=None):
    """
    get data loaders for training and evaluation
    :param config: helper.configure, Configure Object
    :param vocab: data_modules.vocab, Vocab Object
    :param data: on-memory data, Dict{'train': List[str] or None, ...}
    :return: -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    on_memory = data['train'] is not None
    collate_fn = Collator(config, vocab)
    train_dataset = ClassificationDataset(config, vocab, stage='TRAIN', on_memory=on_memory, corpus_lines=data['train'],bert_tokenizer=bert_tokenizer)   #stage='TRAIN'指的是调用训练数据
    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              shuffle=True,
                              num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=True)

    val_dataset = ClassificationDataset(config, vocab, stage='VAL', on_memory=on_memory, corpus_lines=data['val'],bert_tokenizer=bert_tokenizer)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.eval.batch_size,
                            shuffle=True,
                            num_workers=config.train.device_setting.num_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)   # 函数具体用法见markdown,每一次的batch——size为512，一共有10个进程

    test_dataset = ClassificationDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=data['test'],bert_tokenizer=bert_tokenizer)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.eval.batch_size,
                             shuffle=True,
                             num_workers=config.train.device_setting.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
