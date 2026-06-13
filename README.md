# CourseInsight: 中英双语课程评价分析系统

CourseInsight 是一个自然语言处理课程项目，用来分析中文高校课程评价和英文在线课程评论。系统先做文本预处理、情感分类、课程维度识别、关键词提取和相似评论检索，再把结构化结果交给大模型生成课程反馈建议。大模型不可用时，系统会使用本地模板兜底，演示流程不会中断。

项目主线采用 BERT、规则校正和轻量模型回退的分层情感分析流程。BERT 负责主要语义分类，规则处理转折与混合评价，TF-IDF 模型和纯规则在深度模型不可用时保证系统继续运行。

## 项目功能

- 单条评价分析：识别语言、情感、置信度、课程维度、关键词、相似评论，并生成反馈建议。
- CSV 批量分析：支持常见评论字段，展示情感分布、主题分布、高频关键词和分析明细。
- 模型训练与评估：微调 multilingual BERT，同时保留传统分类器作为轻量回退。
- 消融实验：比较 rule-only、model-only、bert-only、hybrid，说明各层模型和规则校正的作用。
- BERT 实时预测：优先使用微调后的 multilingual BERT，支持 GPU/CPU 自动选择和批量推理。
- 噪声文本标准化：展开常见英文缩写，并对情感词执行受限拼写纠错，同时保留原文用于展示和证据提取。

## 目录说明

```text
app.py                         Streamlit 页面入口
src/                           NLP 分析、模型训练、LLM 调用等核心代码
scripts/                       数据处理和消融实验脚本
data/                          示例数据和处理后的训练数据
models/                        传统模型和指标文件
outputs/reports/               消融实验结果
docs/report.md                 课程报告正文
tests/                         单元测试
```

## 环境与运行

普通运行使用项目虚拟环境即可：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

启动系统：

```bash
streamlit run app.py
```

运行测试：

```bash
python -m unittest discover -s tests
```

## 数据来源

训练集由两部分组成：

| 数据 | 数量 | 用途 |
|---|---:|---|
| Coursera 英文课程评论 | 9000 条，每类 3000 条 | 主要训练和量化评估 |
| 中文人工课程评价 | 120 条，每类 40 条 | 中文流程验证和课堂演示 |

英文数据来自 Hugging Face 数据集 `MungunshagaiT/coursera-reviews`。原始文件放在 `data/raw/coursera_reviews_label_3.csv`，该目录不提交。中文数据由项目组按高校课程评价场景人工构造和标注，覆盖教学内容、授课方式、作业任务、考试安排、实验实践和学习收获。

这一点需要明确：当前模型性能主要反映英文 Coursera 评论上的效果，中文样本主要用于验证中文分词、规则识别和系统演示，不把 120 条中文样本包装成充分的中文模型训练数据。

重新构建数据：

```bash
python scripts\prepare_coursera_dataset.py --input data\raw\coursera_reviews_label_3.csv --output data\processed\coursera_reviews_sampled.csv --per-label 3000 --min-chars 20 --max-chars 1200
python scripts\build_bilingual_dataset.py --per-label 3000 --min-chars 20 --max-chars 1200
```

## 传统模型训练

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

当前一次训练结果：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Dummy Most Frequent | 0.3333 | 0.1667 |
| Logistic Regression | 0.6904 | 0.6906 |
| Naive Bayes | 0.6772 | 0.6785 |
| Linear SVM | 0.6811 | 0.6795 |

最优模型是 Logistic Regression。它不是最复杂的模型，但训练快、结果稳定、部署简单，也方便解释。三分类任务中 neutral 边界比较模糊，所以报告中同时展示 Accuracy 和 Macro-F1。

训练会生成：

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
models/model_metrics.json
```

## 消融实验

```bash
python scripts\run_ablation_experiment.py --cases data\test_cases.csv --output-dir outputs\reports --model-dir models
```

当前测试用例结果：

| 实验版本 | 说明 | 结果 |
|---|---|---|
| rule-only | 只用规则情感词和转折判断 | 12/12 |
| model-only | 只用 TF-IDF 模型 | 7/12 |
| hybrid | 模型预测加规则校正 | 12/12 |

这个结果说明：测试集中有不少混合评价和中文短句，单独依赖模型容易误判；规则校正能把明显的转折句、建议型评价和正负混合评价拉回到更合理的标签。

输出文件：

```text
outputs/reports/ablation_metrics.json
outputs/reports/ablation_errors.csv
```

## 独立压力测试

`data/stress_test_cases.csv` 与训练数据、原有 12 条业务案例分开维护，共 48 条，覆盖缩写、拼写错误、否定结构、讽刺、隐含抱怨、中英混合、长文本和正负混合评价。该文件只用于最终评估，不参与模型训练和规则调试。

运行最终模型压力测试：

```bash
python scripts\run_stress_test.py --cases data\stress_test_cases.csv --output-dir outputs\reports\stress_test_final --model-dir models
```

输出文件：

```text
outputs/reports/stress_test_final/stress_metrics.json
outputs/reports/stress_test_final/stress_predictions.csv
outputs/reports/stress_test_final/stress_errors.csv
```

指标文件包含 rule-only、TF-IDF-only、BERT-only 和 hybrid 的总体 Accuracy、Macro-F1、各压力类别指标与各语言指标。正式引用结果前，应由未参与规则开发的队友复核 `expected_sentiment`，并且不要根据这份压力集反复调整规则后继续把它称为独立测试集。

## BERT 主模型

系统默认使用 `auto` 后端：本地 BERT 权重可用时优先运行 BERT；加载或推理失败时回退到 TF-IDF，最后回退到规则。使用 BERT 前需要安装深度学习依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-bert.txt
```

训练命令：

```bash
python -m src.train_bert --data data/processed/bilingual_reviews_train.csv --model-name bert-base-multilingual-cased --output-dir outputs/bert_model --metrics-path outputs/bert_metrics.json
```

如果只检查脚本和数据流，可以用小样本：

```bash
python -m src.train_bert --sample-per-label 5 --epochs 1
```

训练流程固定拆分训练集、验证集和测试集。验证集用于选择最佳 checkpoint，测试集只在训练完成后评估一次。

情感后端配置：

```text
SENTIMENT_BACKEND=auto
BERT_MODEL_PATH=outputs/bert_model
BERT_DEVICE=auto
BERT_BATCH_SIZE=32
BERT_MAX_LENGTH=160
BERT_STRIDE=32
BERT_MAX_CHUNKS=16
```

`SENTIMENT_BACKEND` 可设置为 `auto`、`bert`、`tfidf` 或 `rule`。`BERT_DEVICE=auto` 会优先使用 CUDA，没有 GPU 时使用 CPU。批量分析会合并文本后一次推理，避免逐条调用 GPU。

超过 `BERT_MAX_LENGTH` 的评价会按模型 Token 使用滑动窗口分段，相邻窗口重叠 `BERT_STRIDE` 个 Token，再按每段有效 Token 数加权聚合分类概率。`BERT_MAX_CHUNKS` 限制单条评价的最大分段数；达到上限时，页面会明确提示仍有后续内容未进入 BERT。短文本仍只执行一次预测。

情感分类前会单独生成标准化文本，例如：

```text
Tbh this crs is dificult -> To be honest this course is difficult
I cant recomend this crs -> I can not recommend this course
borng and uncler -> boring and unclear
goooood -> good
```

缩写使用词级映射；重复拉长拼写会先在情感词集合内收缩。其他拼写纠错只面向情感词，要求唯一候选、相同前缀，并且最多只有一次漏字、多字或相邻字母换位。合法词形变化会被保护，`PyTorch`、`NumPy`、课程名等技术词不会参与自动纠错。原始评价仍用于页面展示、关键词、课程维度和证据提取。

`outputs/bert_metrics.json` 可以提交用于展示；`outputs/bert_model/` 约 711 MB，受 GitHub 单文件限制，不直接提交，应在部署时单独下载或挂载。

## 大模型配置

复制 `.env.example` 为 `.env`，填入 API 信息：

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
LLM_MODEL=ecnu-plus
LLM_TIMEOUT=20
```

系统默认配置为 ChatECNU API 地址，调用格式兼容聊天补全接口；如果换用其他同类接口，只需要调整 `LLM_BASE_URL` 和 `LLM_MODEL`。LLM 只根据情感、课程维度、关键词、证据片段和相似评论生成总结建议，不直接替代情感分类器。没有 API 或 API 返回异常时，系统使用本地模板生成 summary、problems、suggestions 和 risk_level。

## 提交说明

交付给老师时建议保留：

- 代码、README、`docs/report.md`
- `data/processed/` 下的处理后数据
- `models/model_metrics.json` 和需要演示的模型文件
- `outputs/reports/ablation_metrics.json`
- `outputs/reports/ablation_errors.csv`

不建议提交：

- `.env`
- `data/raw/`
- `outputs/bert_model/`
- `__pycache__`、测试临时文件和编辑器缓存
