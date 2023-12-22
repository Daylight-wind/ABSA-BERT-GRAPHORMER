from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import json
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
source = []
labels = []
label_ids = []
label_dict = {}
hiera = defaultdict(set)
# ToDo：修改文件路径
with open("/Users/a123456/Code/fei01bert_graphmer/HGCN_data/meituan_total.json", 'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        source.append(tokenizer.encode(line['token'].strip(), truncation=True))
        labels.append(line['label']) # 此处不对标签进行encoding
# 其他代码保持不变
for label_pair in labels:
    for i in range(0, len(label_pair), 2):  # 每次跳过两个元素来获取父子关系
        parent, child = label_pair[i], label_pair[i + 1]

        # 如果父标签不存在于label_dict中，则添加
        if parent not in label_dict:
            label_dict[parent] = len(label_dict)

        # 如果子标签不存在于label_dict中，则添加
        if child not in label_dict:
            label_dict[child] = len(label_dict)

        # 构建标签的索引和层级关系
        parent_id, child_id = label_dict[parent], label_dict[child]
        label_ids.append([parent_id, child_id])
        hiera[parent_id].add(child_id)

# 其他代码保持不变
value_dict = {i: tokenizer.encode(v, add_special_tokens=False) for v, i in label_dict.items()} # v是对应的标签， i是标签在label_dict中的索引，直接encodeing会造成把一个标签编码成多个值
# ...之前的代码保持不变

# 验证编码和解码过程
# 选取一些标签进行测试
test_labels = list(label_dict.keys())[:5]  # 选取前5个标签进行测试
for label in test_labels:
    # 对标签进行编码
    encoded_label = tokenizer.encode(label, add_special_tokens=False)
    # 对编码后的标签进行解码
    decoded_label = tokenizer.decode(encoded_label)

    # 打印结果进行比较
    print(f"原始标签: '{label}'")
    print(f"编码后: {encoded_label}")
    print(f"解码后: '{decoded_label}'")
    print('-' * 30)

# ...之后的代码保持不变

torch.save(value_dict, 'bert_value_dict.pt') # todo：这里直接encoding和addtoken，效果一样吗
torch.save(hiera, 'slot.pt')