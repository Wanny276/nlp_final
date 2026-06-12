# 演示材料准备

本文档用于整理 CourseInsight 的课堂演示输入、截图清单和报告/PPT 可直接使用的表述。

## 一、固定演示输入

建议演示时使用以下输入，避免临场输入导致结果不稳定。

| 场景 | 输入文本 | 展示重点 |
|---|---|---|
| 中文正面 | 老师讲课逻辑清晰，案例很实用，课堂互动也很多。 | positive、授课方式、教学内容 |
| 中文负面 | 实验环境配置太复杂，经常报错，作业截止时间也很紧。 | negative、实验实践、作业任务 |
| 中文混合 | 课程内容很有用，但是作业量偏大，希望能增加示例讲解。 | neutral、混合评价判定 |
| 英文正面 | The instructor explains concepts clearly and the examples are practical and helpful. | positive、英文分词和关键词 |
| 英文负面 | The assignments are too many, the deadlines are stressful, and the setup is confusing. | negative、作业任务、实验实践 |
| 英文混合 | The course is difficult but I learned useful practical skills from the projects. | neutral、难度与收获并存 |
| 中英混合 | 老师讲解很 clear，但是 assignment 太多，deadline 有点紧。 | mixed 语言识别、中英混合处理 |

## 二、截图清单

至少准备 5 张截图：

- 单条中文评价分析结果；
- 单条英文评价分析结果；
- 批量 CSV 分析的情感分布和主题分布；
- 测试用例展示页面的通过结果；
- 模型与技术说明页面的模型对比指标。

可选补充截图：

- 大模型 API 或本地兜底生成的总结建议；
- 批量分析明细表；
- 下载分析结果 CSV 按钮。

## 三、报告/PPT 可用表述

数据集说明：

```text
英文课程评论来自 Hugging Face 的 MungunshagaiT/coursera-reviews 数据集。项目按 positive、neutral、negative 三类均衡抽样，每类 3000 条，共 9000 条英文 Coursera 评论。中文课程评价样本为项目组根据高校课程评价场景人工构造和标注，共 120 条，用于中文 NLP 流程和中英双语输入验证。最终双语训练集共 9120 条。
```

模型说明：

```text
系统使用 TF-IDF unigram/bigram 特征和传统机器学习模型进行三分类情感识别，并比较 Dummy、Logistic Regression、Naive Bayes 和 Linear SVM。当前最优模型为 Logistic Regression，accuracy 为 0.6877，macro_f1 为 0.6867。
```

系统说明：

```text
传统 NLP 模块负责语言识别、文本预处理、情感分类、主题识别、关键词提取和相似评论检索；大模型模块基于结构化分析结果生成课程反馈总结和改进建议。未配置 API 或 API 失败时，系统会自动使用本地模板兜底，保证演示流程稳定。
```
