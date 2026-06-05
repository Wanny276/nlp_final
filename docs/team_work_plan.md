# CourseInsight 两人小组分工与开发计划

本项目为 **CourseInsight 中英双语课程评价智能分析系统**。系统支持中文高校课程评价和英文在线课程评论，完成情感分析、主题识别、关键词提取、相似评论检索、可视化展示和大模型总结建议。

常用入口：

- 项目运行和训练命令见 [../README.md](../README.md)；
- 情感标签判定以 [sentiment_labeling_guidelines.md](sentiment_labeling_guidelines.md) 为准；
- 数据集来源和构建流程见 [dataset_preparation.md](dataset_preparation.md)；
- 期末提交检查见 [submission_checklist.md](submission_checklist.md)。

## 一、总体分工

| 成员 | 负责方向 | 贡献度 |
|---|---|---:|
| 成员 A | 系统主线、Streamlit 界面、传统 NLP 分析、模型接入、可视化、系统整合与演示 | 50% |
| 成员 B | 数据集构建、Coursera 数据处理、大模型 API、Prompt 设计、测试用例、报告/PPT 中数据和 LLM 部分 | 50% |

## 二、成员 A 详细任务

成员 A 主要负责系统代码主线和最终演示效果。

### 1. 系统框架与 GitHub 协作

- 维护 GitHub 仓库结构；
- 合并 Pull Request；
- 保证 `main` 分支代码可以运行；
- 处理代码冲突；
- 维护项目运行说明和提交检查清单。

涉及文件：

```text
README.md
CONTRIBUTING.md
docs/submission_checklist.md
```

### 2. Streamlit 界面开发

负责完善网页演示界面，使系统适合课堂展示。

主要任务：

- 完善系统标题和项目介绍；
- 优化单条评价分析页面；
- 优化批量 CSV 分析页面；
- 增加测试用例展示区域；
- 增加模型效果展示区域；
- 增加图表和分析结果表格；
- 保证中文和英文示例都能正常展示。

涉及文件：

```text
app.py
```

页面需要展示的内容：

```text
语言类型
情感倾向
置信度
主题类别
关键词
预处理结果
相似评论
大模型总结与建议
情感分布图
主题分布图
高频关键词图
```

### 3. 传统 NLP 分析模块

负责系统的传统 NLP 分析能力。

主要任务：

- 优化文本预处理流程；
- 完善中文和英文分词；
- 优化情感分析兜底规则；
- 完善主题识别关键词表；
- 优化关键词提取；
- 优化相似评论检索；
- 将分析结果统一输出给前端页面。

涉及文件：

```text
src/preprocess.py
src/nlp_analyzer.py
src/topic_analyzer.py
src/keyword_extractor.py
src/similarity.py
```

### 4. 可视化展示

负责批量分析结果的可视化。

主要任务：

- 展示情感分布；
- 展示主题分布；
- 展示高频关键词；
- 可选：生成词云图；
- 可选：保存图表到 `outputs/charts/`。

涉及文件：

```text
src/visualizer.py
app.py
```

### 5. 模型训练与接入

当前项目已有双语训练集，成员 A 负责训练和接入情感分类模型。

训练命令：

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

目标生成：

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
models/model_metrics.json
```

接入要求：

- 系统优先使用训练好的模型预测情感；
- 如果模型不存在，则使用规则版情感分析兜底；
- 记录模型准确率、Macro-F1 等指标，用于报告和 PPT；
- 如果修改 `src/preprocess.py`，需要重新训练模型。

涉及文件：

```text
src/train_model.py
src/nlp_analyzer.py
```

## 三、成员 B 详细任务

成员 B 主要负责数据、大模型和测试材料。

### 1. 中英双语数据集构建

负责扩充项目训练和演示数据。

涉及文件：

```text
data/sample_reviews.csv
data/coursera_sample_reviews.csv
```

标准字段：

```csv
id,text,course,teacher,label
```

标签规则以 [sentiment_labeling_guidelines.md](sentiment_labeling_guidelines.md) 为准。简要原则：

```text
positive：主要表达认可、满意或表扬
neutral：有好有坏、转折评价、建议型评价，或正负面信号相对均衡
negative：主要表达抱怨、不满或明显问题
```

建议数据规模：

| 类型 | positive | neutral | negative | 小计 |
|---|---:|---:|---:|---:|
| 中文课程评价 | 30 | 30 | 30 | 90 |
| 英文课程评论 | 30 | 30 | 30 | 90 |
| 合计 | 60 | 60 | 60 | 180 |

最低要求：

```text
至少 100 条中英混合课程评价数据
positive / neutral / negative 尽量均衡
中文和英文都要有样本
```

### 2. Coursera 数据处理

如果使用 Coursera 数据集，成员 B 负责抽样和清洗。

推荐流程：

1. 下载 Coursera Course Reviews 数据集；
2. 保留评论文本、课程名、评分字段；
3. 抽取 300-1000 条样本用于分析；
4. 转换为项目兼容格式；
5. 只提交小样本，不提交过大的原始数据文件。

Coursera 评分转标签规则：

```text
4-5 分 -> positive
3 分 -> neutral
1-2 分 -> negative
```

项目已经支持以下字段：

```text
text
review
reviews
comment
content
rating
course_title
instructor
```

### 3. 测试用例设计

负责整理至少 10 个正式测试用例。

涉及文件：

```text
data/test_cases.csv
```

建议测试类型：

| 类型 | 数量 |
|---|---:|
| 中文正面评价 | 1-2 |
| 中文负面评价 | 1-2 |
| 中文中性评价 | 1-2 |
| 英文正面评论 | 1-2 |
| 英文负面评论 | 1-2 |
| 英文中性评论 | 1-2 |
| 中英混合评论 | 1 |
| 批量 CSV 分析 | 1 |
| 大模型总结 | 1 |
| API 失败兜底 | 1 |

每个测试用例需要记录：

```text
输入文本
预期情感
预期主题
实际系统输出
是否符合预期
截图或结果表格
```

### 4. 大模型 API 与 Prompt 设计

负责完善大模型调用和 Prompt。

涉及文件：

```text
src/llm_client.py
.env.example
docs/bilingual_upgrade_plan.md
```

主要任务：

- 选择大模型 API，例如 DeepSeek、通义千问或 OpenAI 兼容接口；
- 配置 `.env` 文件；
- 设计单条评价总结 Prompt；
- 设计批量评价总结 Prompt；
- 保证输出结构稳定；
- 保证 API 失败时本地兜底可以正常工作。

本地 `.env` 示例：

```text
LLM_API_KEY=自己的 key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

注意：

```text
.env 不能提交到 GitHub
API Key 不能写进代码
```

### 5. 报告和 PPT 中的数据/LLM 部分

成员 B 负责整理以下内容：

- 数据来源说明；
- 中文数据构建方法；
- Coursera 数据抽样方法；
- 标签标注规则；
- 测试用例表；
- 大模型 API 选型；
- Prompt 设计；
- API 输出示例；
- API 失败兜底说明。

## 四、开发顺序

### 阶段 0：同步最新版本

本地更新：

```bash
git checkout main
git pull origin main
python -m unittest discover -s tests
streamlit run app.py
```

目标：

```text
两个人电脑上都能打开 Streamlit 页面
页面标题显示“中英双语课程评价智能分析系统”
中文、英文和中英混合单条评价都能分析
```

### 阶段 1：并行启动

成员 A：

```text
熟悉 app.py 和 src/ 代码
优化页面展示
检查中英双语分析流程
```

成员 B：

```text
扩充 sample_reviews.csv
整理 coursera_sample_reviews.csv
扩充 test_cases.csv
草拟大模型 Prompt
```

### 阶段 2：数据和界面成型

成员 A：

```text
完善单条分析页面
完善批量分析页面
完善图表展示
优化主题识别和关键词提取
```

成员 B：

```text
完成至少 100 条数据
保证三类情感标签均衡
完成 10 个以上测试用例
准备 Coursera 数据说明
```

### 阶段 3：模型和大模型接入

成员 A：

```text
训练 TF-IDF + Logistic Regression 模型
把模型接入 nlp_analyzer.py
整理模型指标
```

成员 B：

```text
接入大模型 API
完善 Prompt
保存 LLM 输出示例
测试 API 失败兜底
```

### 阶段 4：整体验收

两个人一起完成：

```text
单条中文评价演示
单条英文评价演示
CSV 批量分析演示
大模型总结演示
至少 5 个测试用例结果说明
系统截图
演示视频
```

### 阶段 5：报告、PPT 和视频

成员 A 负责：

```text
系统功能
系统架构
界面展示
传统 NLP 模块
可视化结果
程序演示流程
```

成员 B 负责：

```text
数据集说明
Coursera 数据处理
大模型 API
Prompt 设计
测试用例
测试结果
```

## 五、Git 分支建议

成员 A 分支：

```bash
git checkout -b feature/ui-nlp-integration
```

成员 B 分支：

```bash
git checkout -b feature/dataset-llm
```

每次提交前运行：

```bash
python -m unittest discover -s tests
```

提交示例：

```bash
git add .
git commit -m "feat: improve bilingual analysis page"
git push -u origin feature/ui-nlp-integration
```

数据提交示例：

```bash
git add data/
git commit -m "data: expand bilingual review dataset"
git push -u origin feature/dataset-llm
```

大模型提交示例：

```bash
git add src/llm_client.py docs/bilingual_upgrade_plan.md
git commit -m "feat: improve llm prompt and fallback"
git push -u origin feature/dataset-llm
```

## 六、最终贡献度写法

报告中可以这样写：

```text
成员 A：负责系统框架搭建、Streamlit 界面开发、传统 NLP 分析模块、主题识别、关键词提取、相似评论检索、可视化展示、模型接入与系统整合，贡献度 50%。

成员 B：负责中英双语课程评价数据集构建、Coursera 数据抽样与清洗、测试用例设计、大模型 API 接入、Prompt 设计、演示案例整理、报告与 PPT 中数据和大模型相关内容，贡献度 50%。
```

## 七、当前最优先事项

按照优先级执行：

1. 两个人都拉取最新代码并运行项目；
2. 成员 A 重新训练双语情感模型，并确认模型效果页面可截图；
3. 成员 A 检查中文、英文、中英混合单条分析和批量 CSV 分析；
4. 成员 B 完成大模型 API / Prompt / 本地兜底测试；
5. 成员 B 整理测试用例结果和截图；
6. 两人共同整理报告、PPT、演示视频和最终提交材料。
