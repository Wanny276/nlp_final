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
│   ├── demo_materials.md
│   ├── sentiment_labeling_guidelines.md
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

3. 准备 Coursera 抽样数据和双语训练集：

```bash
python scripts\prepare_coursera_dataset.py --input data\raw\coursera_reviews_label_3.csv --output data\processed\coursera_reviews_sampled.csv --per-label 3000 --min-chars 20 --max-chars 1200
python scripts\build_bilingual_dataset.py --per-label 3000 --min-chars 20 --max-chars 1200
```

说明：`data/raw/coursera_reviews_label_3.csv` 来自 Hugging Face `MungunshagaiT/coursera-reviews`，原始大文件不提交到仓库。

4. 训练双语情感分类模型：

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

5. 运行单元测试：

```bash
python -m unittest discover -s tests
```

6. 启动 Web 系统：

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

当前训练集规模：

```text
英文 Coursera 评论：9000 条（positive / neutral / negative 各 3000 条）
中文人工课程评价：120 条（positive / neutral / negative 各 40 条）
合计：9120 条
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
outputs/reports/classification_report.json
outputs/charts/confusion_matrix.png
outputs/charts/model_comparison.png
```

当前一次训练结果：

```text
best_model = Logistic Regression
accuracy = 0.6877
macro_f1 = 0.6867
en_accuracy = 0.6921
en_macro_f1 = 0.6912
zh_accuracy = 0.3077
zh_macro_f1 = 0.2815
```

说明：

- 仓库保留 `models/sentiment_model.pkl`、`models/tfidf_vectorizer.pkl` 和 `models/model_metrics.json`，便于老师直接运行演示；
- 如果本地 Python 或 scikit-learn 版本不兼容，可重新运行训练命令生成模型文件；
- `models/model_metrics.json` 包含总体指标、classification report、confusion matrix 和按语言分组的评估指标；
- `outputs/reports/` 和 `outputs/charts/` 是训练时生成的报告/图表目录，默认不提交生成物；
- 当前中文测试集只有 26 条，中文分组指标仅作为小样本功能验证，不作为主要量化性能结论；
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
LLM_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
LLM_MODEL=ecnu-plus
LLM_TIMEOUT=20
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1024
LLM_RETRIES=2
```

系统调用 ECNU OpenAI 兼容的 `/chat/completions` 接口，并使用 JSON schema 约束输出字段。如果没有配置 API、接口异常或返回 JSON 不可解析，系统会自动使用本地模板生成总结和建议，保证课堂演示不会因为网络问题中断。`.env` 不要提交到仓库。

## 文档导航

- [CONTRIBUTING.md](CONTRIBUTING.md)：分支、提交和 PR 协作约定；
- [docs/README.md](docs/README.md)：项目文档目录；
- [docs/demo_materials.md](docs/demo_materials.md)：演示输入、截图清单和报告/PPT 表述；
- [docs/submission_checklist.md](docs/submission_checklist.md)：期末提交检查清单；
- [docs/sentiment_labeling_guidelines.md](docs/sentiment_labeling_guidelines.md)：情感标签判定与测试准则；
- [docs/dataset_preparation.md](docs/dataset_preparation.md)：数据集下载、抽样、清洗和训练说明。

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
