# CourseInsight 课程报告

## 1. 项目概述

CourseInsight 面向课程评价文本，完成情感分析、课程维度识别、关键词提取、相似评论检索和反馈建议生成。系统支持中文高校课程评价，也支持英文在线课程评论。

项目将微调后的 multilingual BERT 作为主要情感分类器，同时保留规则校正、TF-IDF 模型回退和纯规则兜底。生成式大模型只根据结构化结果生成总结和建议，不直接决定情感标签。这样既提高了上下文语义分类效果，也保证 BERT 权重、GPU 或外部 API 不可用时系统仍能完成主要分析流程。

系统流程如下：

```text
课程评价文本
-> 语言识别
-> 文本清洗与分词
-> BERT 情感分类
-> 转折与混合评价规则校正
-> BERT 不可用时回退到 TF-IDF，再回退到规则
-> 课程维度识别、关键词提取、相似评论检索
-> 大模型或本地模板生成反馈建议
```

## 2. 数据与标签

训练集由英文 Coursera 评论和中文人工课程评价组成。

| 数据来源 | 数量 | 用途 |
|---|---:|---|
| Coursera 英文课程评论 | 9000 条 | 主要训练和量化评估 |
| 中文人工课程评价 | 120 条 | 中文流程验证和课堂演示 |
| 合计 | 9120 条 | 双语训练集 |

英文数据来自 Hugging Face 数据集 `MungunshagaiT/coursera-reviews`。项目按 `positive`、`neutral`、`negative` 三类均衡抽样，每类 3000 条。中文样本由项目组按高校课程评价场景人工构造和标注，覆盖教学内容、授课方式、作业任务、考试安排、实验实践和学习收获。

中文数据规模较小，所以报告中的模型性能主要说明英文 Coursera 数据上的效果。中文部分主要证明系统能处理中文输入、中文分词、课程维度规则和本地兜底建议，不把 120 条中文样本作为充分的中文模型训练依据。

情感标签采用三分类：

| 标签 | 判定标准 |
|---|---|
| positive | 主要表达认可、满意或表扬 |
| neutral | 有好有坏、建议型评价，或正负信号接近 |
| negative | 主要表达抱怨、不满或明显问题 |

带有“但是 / but / however”等转折信号的混合评价，默认归为 `neutral`。只有负面信号明显占主导时，才判为 `negative`。

## 3. 方法设计

文本预处理部分先清洗 URL、特殊符号和多余空白，再按语言做分词。中文使用 `jieba`，英文使用正则分词和停用词过滤。为了适配课程评价，系统额外保留“作业、考试、实验、讲解、收获”等领域词。

情感分类增加了独立的噪声文本标准化层。它将 `tbh`、`idk`、`cant`、`kinda` 等常见表达展开为完整形式，并对 `dificult`、`confusin`、`excelent` 等情感词执行保守的拼写纠错。重复拉长拼写会先在情感词集合内收缩；其他自动纠错要求候选唯一、词首前缀相同，并且最多只有一次漏字、多字或相邻字母换位。合法词形变化受到保护，短词和不规则拼写使用有限词级映射。标准化文本只送入情感分类，原文继续用于页面展示、关键词、主题和证据提取。

情感分类主模型使用在项目三分类数据上微调的 `bert-base-multilingual-cased`。模型读取标准化后的情感文本，利用上下文表示输出 `positive`、`neutral` 和 `negative` 的概率。批量分析会把多条文本合并后一次送入模型，避免逐条调用 GPU。

传统实验仍比较 Dummy、Naive Bayes、Logistic Regression 和 Linear SVM，其中 Logistic Regression 表现最好。它不再承担主分类任务，而是作为 BERT 加载或推理失败时的轻量回退，兼顾部署稳定性与可解释性。

规则模块用于处理课程评价里常见的明确情感词和转折结构。例如“老师讲得很清楚，但是作业太多”同时包含正负信息，模型容易把它推向正面或负面，规则校正会把它归为 `neutral`。

课程维度识别采用领域词典，不把它包装成复杂主题模型。系统会返回维度标签、命中关键词和原文证据片段。例如评论里出现 “assignments” 和 “deadlines”，系统会识别为“作业任务”，并保留对应证据，方便答辩时解释结果来源。

大模型模块只做反馈建议生成。输入是情感、主题、关键词、证据片段和相似评论。API 失败时，系统使用本地模板生成同样结构的输出。

BERT 推理采用懒加载方式，首次分析时加载模型，后续请求复用同一实例。设备设置为 `auto` 时优先使用 CUDA，没有 GPU 时使用 CPU。模型不可用时，系统按照“TF-IDF -> 规则”的顺序自动降级。

## 4. 实验结果

传统模型训练命令：

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

当前结果如下：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Dummy Most Frequent | 0.3333 | 0.1667 |
| Logistic Regression | 0.6904 | 0.6906 |
| Naive Bayes | 0.6772 | 0.6785 |
| Linear SVM | 0.6811 | 0.6795 |

Logistic Regression 的结果最好。三分类课程评价中，`neutral` 往往包含混合态度，边界比正面和负面更模糊，所以 Macro-F1 比单独看 Accuracy 更合适。

BERT 在相同数据划分上的当前结果为：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.6904 | 0.6906 |
| multilingual BERT | 0.7461 | 0.7457 |

BERT 的 Accuracy 提高约 5.57 个百分点，Macro-F1 提高约 5.51 个百分点。训练脚本现已拆分训练集、验证集和测试集；验证集用于选择最佳 checkpoint，测试集只在训练结束后评估一次。

消融实验命令：

```bash
python scripts\run_ablation_experiment.py --cases data\test_cases.csv --output-dir outputs\reports --model-dir models
```

当前 12 个固定测试用例结果：

| 实验版本 | 说明 | 通过情况 |
|---|---|---:|
| rule-only | 只用规则情感词和转折判断 | 12/12 |
| model-only | 只用 TF-IDF 模型 | 7/12 |
| bert-only | 只用 BERT | 11/12 |
| hybrid | BERT 预测、规则校正与自动回退 | 12/12 |

这个结果说明，BERT 明显改善了纯 TF-IDF 在中文短句和上下文语义上的判断，但固定测试集中仍有个别混合评价需要规则校正。例如“内容不错，但是考试范围不明确”会由 BERT 的负面结果校正为中性。完整流程在 12 条业务用例上达到 12/12。

## 5. 系统实现

核心代码位于 `src/`：

| 文件 | 作用 |
|---|---|
| `preprocess.py` | 文本清洗、语言识别、分词 |
| `sentiment_normalizer.py` | 缩写展开和受限情感词拼写纠错 |
| `train_model.py` | 传统模型训练和指标输出 |
| `bert_sentiment.py` | BERT 懒加载、设备选择和批量推理 |
| `nlp_analyzer.py` | BERT、规则与回退模型编排 |
| `topic_analyzer.py` | 课程维度和证据识别 |
| `keyword_extractor.py` | 关键词提取 |
| `similarity.py` | 相似评论检索 |
| `llm_client.py` | 大模型调用和本地兜底 |
| `train_bert.py` | BERT 训练、验证和最终测试 |

前端使用 Streamlit，入口是 `app.py`。页面包括项目概览、单条评价分析、批量 CSV 分析、测试用例展示、模型与技术说明。模型与技术说明页会读取传统模型指标、BERT 指标和消融实验结果；如果文件不存在，页面会提示尚未生成，不影响系统运行。

## 6. 运行与验证

安装主依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-bert.txt
```

启动系统：

```bash
streamlit run app.py
```

运行测试：

```bash
python -m unittest discover -s tests
```

本项目当前测试覆盖文本清洗、语言识别、主题证据、关键词、相似度、LLM 本地兜底、数据读取、BERT 优先选择、BERT 规则校正、批量 TF-IDF 回退、测试用例页面、传统模型指标和消融脚本。

## 7. 项目边界

当前项目的主要不足是中文真实数据规模小。系统可以处理中文输入，也能用规则和词典输出较稳定的中文演示结果，但还不能证明中文情感分类模型已经充分训练。

BERT 权重约 711 MB，无法作为普通 Git 文件直接提交；部署时需要单独下载或挂载。首次加载约需数秒，但 RTX 4060 上模型加载完成后，12 条固定测试用例的批量分析约为 0.03 秒。

后续如果继续优化，优先方向不是继续堆模型，而是补充更多真实中文课程评价、增加错误案例分析，并扩展课程维度词典和证据展示。
