# 数据集下载与抽样说明

本项目当前采用 **中文人工课程评价 + Coursera 英文课程评论抽样数据** 的方式构建中英双语课程评价数据集。

## 一、Coursera 数据来源

本次下载的数据文件来自 Hugging Face 上的 Coursera Reviews 数据集：

```text
https://huggingface.co/datasets/MungunshagaiT/coursera-reviews
```

下载到本地的位置：

```text
data/raw/coursera_reviews_label_3.csv
```

原始数据文件较大，已通过 `.gitignore` 忽略：

```text
data/raw/
```

因此不要把完整原始数据上传到 GitHub。

## 二、原始数据格式

原始数据包含两列：

```csv
text,label
```

其中：

```text
text  = 课程评论文本
label = 原始数字标签
```

本项目使用以下映射规则：

| 原始 label | 项目 label | 含义 |
|---:|---|---|
| 0 | negative | 负面评论 |
| 1 | neutral | 中性/混合评论 |
| 2 | positive | 正面评论 |

## 三、抽样脚本

抽样脚本：

```text
scripts/prepare_coursera_dataset.py
```

运行命令：

```bash
python scripts\prepare_coursera_dataset.py --input data\raw\coursera_reviews_label_3.csv --output data\processed\coursera_reviews_sampled.csv --per-label 100
```

当前输出文件：

```text
data/processed/coursera_reviews_sampled.csv
```

当前抽样结果：

```text
negative    100
neutral     100
positive    100
```

总计 300 条英文课程评论。

## 四、处理后数据格式

处理后的数据格式为：

```csv
id,text,course,teacher,label,source_label
```

字段说明：

| 字段 | 含义 |
|---|---|
| `id` | 抽样后的编号 |
| `text` | Coursera 英文课程评论 |
| `course` | 统一填充为 `Coursera Online Course` |
| `teacher` | 统一填充为 `Coursera Instructor` |
| `label` | 转换后的情感标签 |
| `source_label` | 原始数字标签 |

## 五、模型训练验证

可以直接用处理后的 Coursera 抽样数据训练模型：

```bash
python -m src.train_model --data data\processed\coursera_reviews_sampled.csv --model-dir models
```

当前一次训练结果：

```text
accuracy = 0.6533
macro_f1 = 0.6577
```

后续随着数据清洗和中文数据扩充，模型效果还可以继续提升。

## 六、中文人工课程评价数据

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

数据来源说明：

```text
中文课程评价样本由项目组根据高校课程评价场景人工构造和标注，用于系统演示、中文 NLP 流程验证和双语模型训练。
```

## 七、英文数据清洗

基于 300 条 Coursera 均衡抽样数据，进一步生成清洗版英文数据：

```text
data/processed/coursera_reviews_cleaned.csv
```

清洗规则：

```text
去除空文本
去除重复文本
去除少于 20 个字符的过短文本
去除超过 1200 个字符的过长文本
每个情感类别保留 80 条
```

当前规模：

```text
positive    80
neutral     80
negative    80
```

总计 240 条英文课程评论。

## 八、中英双语训练集

合并中文人工数据和英文 Coursera 清洗数据，生成双语训练集：

```text
data/processed/bilingual_reviews_train.csv
```

当前规模：

```text
中文 zh：120 条
英文 en：240 条
总计：360 条
```

情感标签分布：

```text
positive    120
neutral     120
negative    120
```

语言和标签交叉分布：

```text
en negative    80
en neutral     80
en positive    80
zh negative    40
zh neutral     40
zh positive    40
```

构建命令：

```bash
python scripts\build_bilingual_dataset.py --per-label 80 --min-chars 20 --max-chars 1200
```

## 九、双语模型训练验证

使用双语训练集训练：

```bash
python -m src.train_model --data data\processed\bilingual_reviews_train.csv --model-dir models
```

当前一次训练结果：

```text
accuracy = 0.5333
macro_f1 = 0.5350
```

说明：

```text
该结果基于当前小规模中英混合数据和传统 TF-IDF + Logistic Regression 模型。由于中英文文本特征空间差异较大，且中文数据规模仍较小，后续可以通过扩充中文样本、增加英文样本、优化预处理和使用更强模型继续提升效果。
```

## 十、后续建议

1. 保留 `data/processed/coursera_reviews_sampled.csv` 作为英文训练和演示数据；
2. 不提交 `data/raw/` 中的完整原始数据；
3. 使用 `data/processed/coursera_reviews_cleaned.csv` 作为英文清洗数据；
4. 使用 `data/processed/chinese_manual_reviews.csv` 作为中文人工标注数据；
5. 使用 `data/processed/bilingual_reviews_train.csv` 作为当前双语训练数据；
6. 后续继续扩充中文课程评价数据；
7. 报告中说明 Coursera 数据作为英文在线课程评论来源，中文数据由项目组人工构造和标注。
