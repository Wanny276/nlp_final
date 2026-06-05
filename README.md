# CourseInsight: 中英双语课程评价智能分析与改进建议系统

CourseInsight 是面向自然语言处理期末大作业的课程评价分析系统。项目采用 **传统 NLP 分析 + 大模型 API 增强** 的混合架构：先用文本预处理、情感分类、主题识别、关键词提取和相似评论检索得到结构化结果，再调用大模型生成课程反馈总结与改进建议。

系统支持两类课程反馈：

- 中文高校课程评价：用于课堂演示和体现中文 NLP 处理流程；
- 英文在线课程评论：用于接入 Coursera 等公开课程评论数据。

## 功能概览

- 单条课程评价分析：语言识别、情感倾向、置信度、主题、关键词、预处理结果、相似评论；
- CSV 批量分析：兼容中文样本和 Coursera 风格字段，展示情感分布、主题分布、高频关键词和明细表；
- 传统 NLP 模块：中文/英文分词、停用词过滤、TF-IDF、Logistic Regression、规则兜底、余弦相似度；
- 大模型增强：根据结构化结果生成总结、问题归纳和改进建议，API 失败时使用本地模板兜底；
- 测试与展示：内置测试用例页面、模型与技术说明页面，便于报告和 PPT 截图。

## 课程知识点对应

| 课上内容 | 项目体现 |
|---|---|
| L1 Introduction | 课程评价自动分析的真实应用背景 |
| L2 Text Preprocessing | 文本清洗、语言识别、中文 jieba 分词、英文正则分词、停用词过滤 |
| L3 n-gram Language Model | 使用 unigram / bigram 作为 TF-IDF 特征 |
| L4 Text Classification | 将评价划分为 `positive`、`neutral`、`negative` |
| L5 Logistic Regression | 使用 TF-IDF + Logistic Regression 训练情感分类模型 |
| L6-L8 Word Embedding / Word2Vec / Sequence Models | 作为后续拓展方向，可用于更强的语义表示和分类模型 |

## 情感标签标准

项目统一使用三分类情感标签：

| 标签 | 判定标准 |
|---|---|
| `positive` | 评价主体主要表达认可、满意或表扬，没有明显问题 |
| `neutral` | 有好有坏、转折评价、建议型评价，或正负面信号相对均衡 |
| `negative` | 评价主体主要表达抱怨、不满或明显问题，负面信号占主导 |

同时包含正负面信息并出现“但是 / but / however”等转折信号的评价，优先视为混合评价；在三分类中统一映射为 `neutral`。完整准则见 [docs/sentiment_labeling_guidelines.md](docs/sentiment_labeling_guidelines.md)。

## 目录结构

```text
.
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── .env.example
├── data/
│   ├── sample_reviews.csv
│   ├── coursera_sample_reviews.csv
│   ├── stopwords.txt
│   ├── test_cases.csv
│   └── processed/
├── docs/
│   ├── README.md
│   ├── dataset_preparation.md
│   ├── sentiment_labeling_guidelines.md
│   ├── team_work_plan.md
│   └── submission_checklist.md
├── models/
├── outputs/
├── src/
└── tests/
```

## 快速开始

建议使用 Python 3.10 或 3.11。

1. 创建并激活虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 训练双语情感分类模型：

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

4. 运行单元测试：

```bash
python -m unittest discover -s tests
```

5. 启动 Web 系统：

```bash
streamlit run app.py
```

## 数据格式

标准 CSV 字段：

```csv
id,text,course,teacher,label
```

系统也兼容常见英文课程评论字段：

```text
review, reviews, comment, content, rating, course_title, instructor
```

若上传数据包含 `rating` 字段，系统会自动转换情感标签：

```text
4-5 分 -> positive
3 分 -> neutral
1-2 分 -> negative
```

当前双语训练集为：

```text
data/processed/bilingual_reviews_train.csv
```

数据构建和 Coursera 抽样说明见 [docs/dataset_preparation.md](docs/dataset_preparation.md)。

## 训练模型

使用当前双语训练集训练 TF-IDF + 传统分类模型，并保存最优模型：

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

训练后会生成：

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
models/model_metrics.json
```

说明：

- `models/*.pkl` 是训练生成物，默认不建议提交到 GitHub；
- `models/model_metrics.json` 可用于模型效果页面展示，也可作为报告/PPT 的截图依据；
- 如果修改了 `src/preprocess.py`、训练数据、模型参数或本地 scikit-learn 版本，建议重新训练模型；
- 如果只修改 Streamlit 页面、文档或普通单元测试，一般不需要重新训练模型。

建议重新训练的情况：

```text
修改文本清洗、分词、停用词、领域词表
修改 data/processed/ 中的训练数据
修改 src/train_model.py 中的模型或特征参数
更换 Python / scikit-learn 环境后出现模型加载版本警告
```

## 大模型 API 配置

复制 `.env.example` 为 `.env`，填入自己的 API 信息：

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

如果没有配置 API，系统会自动使用本地模板生成总结和建议，保证课堂演示不会因为网络问题中断。`.env` 不要提交到仓库。

## 文档导航

- [CONTRIBUTING.md](CONTRIBUTING.md)：分支、提交和 PR 协作约定；
- [docs/README.md](docs/README.md)：项目文档目录；
- [docs/team_work_plan.md](docs/team_work_plan.md)：两人小组分工与开发计划；
- [docs/submission_checklist.md](docs/submission_checklist.md)：期末提交检查清单；
- [docs/sentiment_labeling_guidelines.md](docs/sentiment_labeling_guidelines.md)：情感标签判定与测试准则；
- [docs/dataset_preparation.md](docs/dataset_preparation.md)：数据集下载、抽样、清洗和训练说明；
- [docs/bilingual_upgrade_plan.md](docs/bilingual_upgrade_plan.md)：中英双语改造说明。

## 提交前检查

```bash
python -m unittest discover -s tests
streamlit run app.py
```

提交前确认：

- 单条中文、英文、中英混合评价都能分析；
- CSV 批量分析能展示图表和明细；
- 大模型 API 或本地兜底能生成总结建议；
- 没有提交 `.env`、`data/raw/`、无关缓存和大文件。
