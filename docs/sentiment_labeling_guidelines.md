# 情感标签判定与测试准则

本项目采用三分类情感标签：

| 标签 | 判定标准 | 示例 |
|---|---|---|
| `positive` | 评价主体主要表达认可、满意或表扬，没有明显问题 | 老师讲课很清楚，课程内容很有帮助 |
| `neutral` | 有好有坏、转折评价、建议型评价，或正负面信号相对均衡 | 老师讲得很清楚，但是作业有点多 |
| `negative` | 评价主体主要表达抱怨、不满或明显问题，负面信号占主导 | 作业太多，实验环境经常报错 |

## 混合评价处理

如果文本同时包含正面和负面信息，并出现“但是 / but / however”等转折信号，本项目将其优先视为混合评价。

由于系统只使用 `positive / neutral / negative` 三分类，混合评价统一映射为 `neutral`。只有当负面信号明显多于正面信号时，才判为 `negative`。

示例：

```text
The instructor explains concepts clearly but the assignments are too many and the setup is confusing.
```

判定：

```text
neutral
```

原因：

```text
正面：explains clearly
负面：assignments are too many, setup is confusing
结构：but 表示转折，属于有好有坏的混合评价
```

## 与课程知识点的对应

- L2 Text Preprocessing：清洗文本、分词、去停用词、保留领域关键词；
- L3 n-gram Language Model：使用 unigram / bigram 作为 TF-IDF 特征；
- L4 Text Classification：将课程评价划分为 positive / neutral / negative；
- L5 Logistic Regression：使用 TF-IDF + Logistic Regression 训练情感分类模型；
- L6-L8：作为后续拓展，可进一步使用 Word2Vec、词向量或序列模型提升效果。

## 测试准则

普通单元测试应验证稳定规则和核心流程，不依赖本地训练好的 `.pkl` 模型文件。

原因：

```text
模型文件受训练数据、随机划分、scikit-learn 版本影响，不适合作为基础单元测试的隐式依赖。
```

因此，规则类测试使用 `unittest.mock.patch` 禁用模型加载，只测试固定规则；模型训练和模型指标单独通过训练命令验证。

