#!/usr/bin/env python
# coding:utf-8

import os
import sys
import time
import wandb
import torch
from transformers import BertTokenizer, get_linear_schedule_with_warmup

import helper.logger as logger
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from helper.adamw import AdamW
from helper.configure import Configure
from helper.utils import load_checkpoint, save_checkpoint
from models.model import HiAGM
from train_modules.criterions import ClassificationLoss
from train_modules.trainer import Trainer


def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary   导入语料库产生词表
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=config.vocabulary.max_token_vocab)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # get data 获取数据
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab,
                                                         bert_tokenizer=tokenizer)  # todo:在match中的这一步有一个获取label_desc的过程，我们需要在这个位置加入desc

    # build up model 建立模型
    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
    hiagm.to(config.train.device_setting.device)
    # define training objective & optimizer
    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                   corpus_vocab.v2i['label'],
                                   recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_constraint=config.train.loss.recursive_regularization.flag)  # 这里可以增加了一个loss形参：bce  criterion中的recursive_relation存储了父子之间的关系

    if "bert" in config.model.type:
        t_total = int(len(train_loader) * (config.train.end_epoch - config.train.start_epoch))

        param_optimizer = list(hiagm.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(t_total * 0.1)
        optimize = AdamW(optimizer_grouped_parameters, lr=config.train.optimizer.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimize, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
    else:
        optimize = set_optimizer(config, hiagm)
        scheduler = None

    # get epoch trainer
    trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=optimize,
                      tokenizer=tokenizer,
                      scheduler=scheduler,
                      vocab=corpus_vocab,
                      config=config,
                      train_loader=train_loader,
                      dev_loader=dev_loader,
                      test_loader=test_loader
                      )  # criterion 分类损失  todo:这里需要按照himatch加入形参label_desc_loader吗？trainer存储了父子之间的关系

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    model_checkpoint = config.train.checkpoint.dir  # 检查点
    model_name = config.model.type
    wait = 0
    if not os.path.isdir(model_checkpoint):
        os.mkdir(model_checkpoint)
    else:
        # loading previous checkpoint
        dir_list = os.listdir(model_checkpoint)  # 获取该文件夹目录下包含的文件
        dir_list.sort(key=lambda fn: os.path.getatime(os.path.join(model_checkpoint, fn)))  # 排序，且按照返回文件最后的访问时间作为规则
        latest_model_file = ''

        for model_file in dir_list[::-1]:
            if model_file.startswith('best'):
                continue
            else:
                latest_model_file = model_file
                break
        if os.path.isfile(os.path.join(model_checkpoint, latest_model_file)):
            logger.info('Loading Previous Checkpoint...')
            logger.info('Loading from {}'.format(os.path.join(model_checkpoint, latest_model_file)))
            best_performance, config = load_checkpoint(model_file=os.path.join(model_checkpoint, latest_model_file),
                                                       model=hiagm,
                                                       config=config,
                                                       optimizer=optimize)
            logger.info('Previous Best Performance---- Micro-F1: {}%, Macro-F1: {}%'.format(
                best_performance[0], best_performance[1]))

    # train
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        start_time = time.time()
        trainer.train(train_loader,
                      epoch)
        trainer.eval(train_loader, epoch, 'TRAIN')  # eval（）计算指定表达式的值。也就是说它要执行的Python代码只能是单个运算表达式
        performance = trainer.eval(dev_loader, epoch, 'DEV')

        # 在这里记录当前epoch的性能指标到wandb

        wandb.log({"epoch": epoch, "Micro_F1": performance['micro_f1'], "Macro_F1": performance['macro_f1']})

        # saving best model and check model
        if not (performance['micro_f1'] >= best_performance[0] or performance['macro_f1'] >= best_performance[1]):
            wait += 1
            if wait % config.train.optimizer.lr_patience == 0:
                logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
                trainer.update_lr()
            if wait == config.train.optimizer.early_stopping:
                logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                               .format(wait))
                break

        if performance['micro_f1'] > best_performance[0]:
            wait = 0
            logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro_f1']))
            best_performance[0] = performance['micro_f1']
            best_epoch[0] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()  # 优化器
            }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
        if performance['macro_f1'] > best_performance[1]:
            wait = 0
            logger.info('Improve Macro-F1 {}% --> {}%'.format(best_performance[1], performance['macro_f1']))
            best_performance[1] = performance['macro_f1']
            best_epoch[1] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

        if epoch % 10 == 1:  # 每10次保存一个检查点
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))

        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[0], 'TEST')

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[1], 'TEST')

    return


if __name__ == "__main__":
    # 设置TOKENIZERS_PARALLELISM环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # configs = Configure(config_json_file=sys.argv[1])
    configs = Configure(config_json_file='/Users/a123456/Code/fei01bert_graphmer/config/patent-bert.json')
    # ToDo：修改文件目录
    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2019)  # 设置随机种子
    torch.cuda.manual_seed(2019)  # 为GPU设置随机种子
    logger.Logger(configs)
    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)
    # 初始化wandb
    wandb.init(project='ABSA-graphormer', name='BERT-GRAPHORMER', config=configs)


    train(configs)
# 修改测试