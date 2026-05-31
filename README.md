# CourseInsight: 中英双语课程评价智能分析与改进建议系统

面向自然语言处理期末大作业的课程评价分析系统。项目采用“传统 NLP 分析 + 大模型 API 增强”的混合架构：先用文本预处理、情感分类、主题识别、关键词提取和相似评论检索得到结构化结果，再调用大模型生成课程反馈总结与改进建议。

系统当前定位为中英双语课程评论分析：

- 中文高校课程评价：用于课堂演示和体现中文 NLP 处理流程；
- 英文在线课程评论：用于接入 Coursera 等公开课程评论数据集。

## 当前状态

本仓库目前是团队协作开发骨架，已经包含：

- Streamlit Web 入口：`app.py`
- NLP 核心模块：`src/`
- 示例课程评价数据：`data/sample_reviews.csv`
- Coursera 格式示例数据：`data/coursera_sample_reviews.csv`
- 测试用例数据：`data/test_cases.csv`
- 环境依赖：`requirements.txt`
- API 配置模板：`.env.example`
- 单元测试：`tests/`
- GitHub Actions CI：`.github/workflows/ci.yml`

## 主要功能规划

必做功能：

1. 单条课程评价分析
2. CSV 批量评价分析
3. 中英双语文本清洗、分词、停用词过滤
4. TF-IDF + Logistic Regression 情感分类
5. 中英双语主题关键词规则识别
6. 高频关键词提取
7. 情感分布和主题分布可视化
8. 大模型 API 生成总结和建议
9. 至少 5 个测试用例与结果说明

加分功能：

1. 多模型对比
2. 混淆矩阵展示
3. 相似评论检索
4. 词云图
5. 分析报告导出
6. 大模型 JSON 结构化输出
7. API 失败后的本地模板兜底

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
│   └── test_cases.csv
├── models/
│   └── .gitkeep
├── outputs/
│   ├── charts/.gitkeep
│   └── reports/.gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── keyword_extractor.py
│   ├── llm_client.py
│   ├── nlp_analyzer.py
│   ├── preprocess.py
│   ├── similarity.py
│   ├── topic_analyzer.py
│   ├── train_model.py
│   └── visualizer.py
└── tests/
    └── test_core.py
```

## 快速开始

建议使用 Python 3.10 或 3.11。Python 3.13 也可以先跑基础测试，但部分第三方包可能需要等待兼容版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

运行 Web 系统：

```bash
streamlit run app.py
```

系统支持上传包含 `text`、`review`、`reviews`、`comment` 或 `content` 字段的 CSV。若上传 Coursera 风格数据并包含 `rating` 字段，系统会自动转换情感标签：

```text
4-5 分 -> positive
3 分 -> neutral
1-2 分 -> negative
```

运行测试：

```bash
python -m unittest discover -s tests
```

训练情感分类模型：

```bash
python -m src.train_model --data data/sample_reviews.csv --model-dir models
```

## 大模型 API 配置

复制 `.env.example` 为 `.env`，填入自己的 API 信息：

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

如果没有配置 API，系统会自动使用本地模板生成总结和建议，保证课堂演示不会因为网络问题中断。

## 团队分工建议

| 角色 | 主要任务 |
|---|---|
| A | 文本预处理、情感分类模型训练、模型评估 |
| B | 大模型 API、Prompt 设计、异常兜底 |
| C | Streamlit 界面、可视化、测试用例、演示视频 |

两人小组建议：

| 成员 | 主要任务 |
|---|---|
| A | 系统框架、Streamlit 页面、传统 NLP 分析、主题识别、关键词提取、相似评论检索、可视化和系统整合 |
| B | 中文数据扩充、Coursera 数据抽样、测试用例、大模型 API、Prompt、演示案例、报告/PPT 中测试和 LLM 部分 |

## GitHub 协作流程

1. 每个人从 `main` 新建自己的功能分支，例如 `feature/sentiment-model`。
2. 提交前先运行测试：`python -m unittest discover -s tests`。
3. 通过 Pull Request 合并，避免直接改 `main`。
4. PR 描述中说明改了什么、怎么测试、是否影响演示。

## 后续待办

- [ ] 扩充真实或人工审核后的课程评价数据
- [ ] 抽样并清洗 Coursera 英文课程评论数据
- [ ] 完成 Logistic Regression 模型训练和评估截图
- [ ] 在 Streamlit 页面展示混淆矩阵和模型指标
- [ ] 接入可用的大模型 API
- [ ] 补充报告导出功能
- [ ] 整理 5 个以上测试用例的运行结果截图
