# 数据说明

为了应用层级多标签模型，因此对原始meituan数据集进行处理。将csv文件转换成JSON格式，并修改相应字段。

字段说明如下：
原始数据展示：![image-20231215213345909](/Users/a123456/Library/Application Support/typora-user-images/image-20231215213345909.png)

对字段进行简写，修改对比如下表所示

| csv字段                 | JSON字段 |
| ----------------------- | -------- |
| token                   | token    |
| Location#Transportation | LT       |
| Location#Downtown       | LD       |
| Location#Easy_to_find   | LE       |
| Service#Queue           | SQ       |
| Service#Hospitality     | SH       |
| Service#Parking         | SP       |
| Service#Timely          | ST       |
| Price#Level             | PL       |
| Price#Cost_effective    | PC       |
| Price#Discount          | PD       |
| Ambience#Decoration     | AD       |
| Ambience#Noise          | AN       |
| Ambience#Space          | AS       |
| Ambience#Sanitary       | ==AH==   |
| Food#Portion            | FP       |
| Food#Taste              | FT       |
| Food#Appearance         | FA       |
| Food#Recommend          | FR       |

特殊说明：由于Ambience#Space与Ambience#Sanitary的缩写冲突，因此将Ambience#Sanitary改成Ambience#Hygiene,缩写为AH

对于每个非"-2"的标签值，我们需要在"label"数组中同时包含列名和列名与其对应的值。

处理代码如下图所示

```python
import pandas as pd
import json

# 加载CSV文件
file_path = '/Users/a123456/Code/fei01bert_graphmer/HGCN_data/train.csv'  # 替换为您的文件路径
df = pd.read_csv(file_path)

# 进一步修改的函数，用于将每行数据转换为所需的JSON格式
def further_modified_row_to_json(row):
    labels = []
    for col in row.index[1:]:
        if row[col] != -2:
            labels.append(col)
            labels.append(f"{col}{int(row[col])}")

    json_obj = {
        "token": row['token'],
        "label": labels,
        "doc_topic": [],
        "doc_keyword": []
    }
    return json_obj

# 应用进一步修改的函数到数据框的每一行
further_modified_json_data = df.apply(further_modified_row_to_json, axis=1).tolist()

# 保存转换后的数据为JSON文件，但不在外部列表中
output_file_path = '/Users/a123456/Code/fei01bert_graphmer/HGCN_data/train.json'  # 指定输出文件的路径
with open(output_file_path, 'w', encoding='utf-8') as f:
    for entry in further_modified_json_data:
        f.write(json.dumps(entry, ensure_ascii=False, indent=4))
        f.write(",\n")  # 添加逗号和换行符，与示例格式匹配

# 输出信息表示操作完成
print(f"Data successfully saved to {output_file_path}")
```

