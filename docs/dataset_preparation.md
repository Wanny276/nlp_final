# 数据集下载与抽样说明

本项目采用 **中文人工课程评价 + Coursera 英文课程评论抽样数据** 构建中英双语课程评价数据集。英文数据用于模型训练和主要性能评估，中文数据用于中文 NLP 流程演示和跨语言输入验证。

## 一、Coursera 数据来源

英文原始数据来自 Hugging Face 数据集：

```text
https://huggingface.co/datasets/MungunshagaiT/coursera-reviews
```

本地原始文件名保持为网站上的原始文件名：

```text
data/raw/coursera_reviews_label_3.csv
```

当前本地文件校验结果：

```text
rows = 107018
columns = text,label
label 0 = 4720
label 1 = 5071
label 2 = 97227
```

`data/raw/` 已在 `.gitignore` 中忽略，不要把完整原始数据提交到 GitHub。

## 二、原始数据格式

原始数据包含两列：

```csv
text,label
```

字段含义：

| 字段 | 含义 |
|---|---|
| `text` | Coursera 课程评论文本 |
| `label` | 原始数字情感标签 |

标签映射：

| 原始 label | 项目 label | 含义 |
|---:|---|---|
| 0 | negative | 负面评论 |
| 1 | neutral | 中性/混合评论 |
| 2 | positive | 正面评论 |

## 三、Coursera 抽样数据

抽样脚本：

```text
scripts/prepare_coursera_dataset.py
```

推荐构建命令：

```bash
python scripts\prepare_coursera_dataset.py --input data\raw\coursera_reviews_label_3.csv --output data\processed\coursera_reviews_sampled.csv --per-label 3000 --min-chars 20 --max-chars 1200
```

处理规则：

```text
空值处理
压缩多余空白
去除少于 20 个字符的过短文本
去除超过 1200 个字符的过长文本
按 text 去重
映射 0/1/2 到 negative/neutral/positive
按三类情感均衡抽样
```

当前输出文件：

```text
data/processed/coursera_reviews_sampled.csv
```

当前抽样结果：

```text
negative    3000
neutral     3000
positive    3000
```

总计 9000 条英文课程评论。

处理后字段：

```csv
id,text,course,teacher,label,source_label
```

## 四、中文人工课程评价数据

中文数据文件：

```text
data/processed/chinese_manual_reviews.csv
```

字段格式：

```csv
id,text,course,teacher,label,topics,language,source
```

当前规模：

```text
positive    40
neutral     40
negative    40
```

总计 120 条中文课程评价。

中文数据覆盖的主题包括：

```text
教学内容
授课方式
作业任务
考试安排
实验实践
学习收获
```

中文课程评价样本由项目组根据高校课程评价场景人工构造和标注，用于系统演示、中文 NLP 流程验证和双语输入验证。由于公开的大规模中文课程评价三分类数据较难获得，当前模型主要性能评估以英文 Coursera 评论为主。

## 五、中英双语训练集

构建脚本：

```text
scripts/build_bilingual_dataset.py
```

构建命令：

```bash
python scripts\build_bilingual_dataset.py --per-label 3000 --min-chars 20 --max-chars 1200
```

输出文件：

```text
data/processed/bilingual_reviews_train.csv
```

当前规模：

```text
英文 en：9000 条
中文 zh：120 条
总计：9120 条
```

语言和标签交叉分布：

```text
en negative    3000
en neutral     3000
en positive    3000
zh negative      40
zh neutral       40
zh positive      40
```

说明：训练集整体以英文 Coursera 数据为主，中文样本保留为系统双语能力演示和中文流程验证材料。

## 六、模型训练验证

使用双语训练集训练：

```bash
python -m src.train_model --data data\processed\bilingual_reviews_train.csv --model-dir models
```

当前一次训练结果：

```text
best_model = Logistic Regression
accuracy = 0.6877
macro_f1 = 0.6867
train_size = 6840
test_size = 2280
```

模型对比结果：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Dummy Most Frequent | 0.3333 | 0.1667 |
| Logistic Regression | 0.6877 | 0.6867 |
| Naive Bayes | 0.6728 | 0.6738 |
| Linear SVM | 0.6820 | 0.6804 |

该结果基于扩充后的 Coursera 英文均衡抽样数据和现有中文人工样本。报告中建议优先展示 `macro_f1`，因为它比 accuracy 更能反映三分类任务中各类别的整体表现。

## 七、后续建议

1. 保留 `data/processed/coursera_reviews_sampled.csv` 作为英文训练和演示数据；
2. 不提交 `data/raw/` 中的完整原始数据；
3. 使用 `data/processed/chinese_manual_reviews.csv` 作为中文人工标注数据；
4. 使用 `data/processed/bilingual_reviews_train.csv` 作为当前双语训练数据；
5. 后续若继续提升中文效果，应优先补充课程领域中文评价，而不是混入酒店、电商、微博等通用中文情感数据；
6. 报告中说明 Coursera 数据作为英文在线课程评论来源，中文数据由项目组人工构造和标注。
