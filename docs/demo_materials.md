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
英文课程评论来自 Hugging Face 的 MungunshagaiT/coursera-reviews 数据集。项目按 positive、neutral、negative 三类均衡抽样，每类 3000 条，共 9000 条英文 Coursera 评论。中文课程评价样本为项目组根据高校课程评价场景人工构造和标注，共 120 条，用于高校课程评价场景适配、中文 NLP 流程验证和中文输入演示。最终训练集共 9120 条。当前量化评估主要基于英文 Coursera 评论，不把中文小样本指标作为强性能结论。
```

模型说明：

```text
系统使用 TF-IDF unigram/bigram 特征和传统机器学习模型进行三分类情感识别，并比较 Dummy baseline、Logistic Regression、Naive Bayes 和 Linear SVM。当前最优模型为 Logistic Regression，accuracy 为 0.6877，macro_f1 为 0.6867。三分类任务中 macro_f1 比 accuracy 更能反映三类整体表现，因此报告中建议两个指标一起展示。
```

系统说明：

```text
传统 NLP 模块负责语言识别、文本预处理、情感分类、主题识别、关键词提取和相似评论检索；大模型模块基于结构化分析结果生成课程反馈总结和改进建议。系统支持 OpenAI-compatible API，演示默认使用 ECNU Open API 配置。未配置 API、API 失败或返回 JSON 不可解析时，系统会自动使用本地模板兜底，保证演示流程稳定。
```

## 四、答辩重点口径

中文数据定位：

```text
本项目的核心量化评估主要基于 Coursera 英文课程评论；中文部分主要用于高校课程评价场景适配、中文预处理流程验证和中文输入演示。系统通过中文分词、课程维度关键词、规则兜底和本地模板增强中文可用性，但不把当前 120 条中文小样本作为中文模型强性能结论。
```

创新点 1：

```text
面向课程评价场景的“结构化 NLP -> LLM 建议生成”链路。系统先输出语言、情感、主题、关键词和相似评论，再让 LLM 基于这些结构化结果生成总结与建议，而不是直接把原始评论交给大模型。
```

创新点 2：

```text
可解释的课程维度识别。主题识别不仅返回维度标签，还返回命中关键词和证据片段，便于教师理解系统为什么判断该评价涉及教学内容、作业任务、实验实践等方面。
```

模型结果答法：

```text
我们选择 TF-IDF + Logistic Regression，是因为它训练成本低、结果可解释、部署稳定，适合课程项目演示。后续如果补充更多真实中文课程评价，可以再引入 BERT/RoBERTa 等预训练模型做语义表示增强。
```
