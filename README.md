# CourseInsight：面向课程评论的中英双语智能分析系统

CourseInsight 是一个面向高校课程评论场景的自然语言处理应用。系统支持中文、英文和中英混合评价，
可以完成情感分类、课程维度识别、关键词提取、证据定位、批量统计和教学建议生成。

情感分析采用分层混合流程：优先使用微调后的 multilingual BERT，结合否定、转折、建议型表达和
混合情感规则进行校正；BERT 不可用时自动回退到 TF-IDF Logistic Regression，传统模型仍不可用时
继续使用规则。大模型只负责根据结构化证据整理教学建议，API 不可用时由本地模板兜底。

## 当前状态

- Streamlit 系统包含首页概览、单条分析、批量分析、固定案例验证和模型评估五个页面。
- 传统模型、BERT、规则校正、自动回退和长文本滑动窗口均已集成。
- 12 条固定案例、48 条独立压力测试和自动化测试均已建立。
- LaTeX 期末报告已完成，最新版位于 `report/main.pdf`，共 28 页 A4。
- 报告已包含目录、系统架构图、数据流图、实验图表、消融实验、真实界面截图、部署测试和贡献度。
- 小组成员为王佳妮、柯文丽，贡献度各 50%。

## 主要功能

- **单条评价分析**：输出语言、情感、置信度、模型来源、课程维度、关键词、原文证据、和教学建议。
- **CSV 批量分析**：支持常见评价字段，展示情感分布、课程维度、高频关键词和逐条结果，并可下载明细。
- **固定案例验证**：一键运行 12 条中英文固定案例，显示预期值、实际值和通过情况。
- **模型评估**：展示传统模型、BERT、分类报告、混淆矩阵、消融实验和当前运行后端。
- **噪声文本处理**：展开英文缩写，收缩重复字母，并仅对高置信度情感词执行保守拼写纠错。
- **长文本推理**：使用带重叠的滑动窗口处理长评价，并按有效 Token 数加权聚合概率。
- **结构化建议生成**：LLM 只整理已识别的情感、维度和证据；服务异常时自动使用本地模板。

## 项目结构

```text
app.py                         Streamlit 应用入口
src/                           NLP 分析、BERT、传统模型、LLM 和训练代码
scripts/                       数据处理、消融实验、压力测试和报告图表脚本
tests/                         自动化测试
data/                          示例数据、处理后数据、固定案例和压力测试
models/                        TF-IDF 模型、向量器和传统模型指标
outputs/                       BERT 指标、实验结果和本地大模型权重
report/                        LaTeX 报告源码、图表、真实界面截图和 main.pdf
docs/report.md                 早期文字稿，仅供历史参考，不是最终提交版
```

## 快速开始

以下命令适用于 Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

需要运行或训练 BERT 时，再安装深度学习依赖：

```powershell
pip install -r requirements-bert.txt
```

复制环境配置并按需填写：

```powershell
Copy-Item .env.example .env
```

启动应用：

```powershell
streamlit run app.py
```

运行测试：

```powershell
python -m unittest discover -s tests
```

自动化测试覆盖核心流程，提交前请以实际运行结果为准。

## 环境配置

情感后端相关配置：

```text
SENTIMENT_BACKEND=AUTO
BERT_MODEL_PATH=outputs/bert_model_final
BERT_DEVICE=AUTO
BERT_BATCH_SIZE=32
BERT_MAX_LENGTH=160
BERT_STRIDE=32
BERT_MAX_CHUNKS=16
```

`SENTIMENT_BACKEND` 可设为 `AUTO`、`BERT`、`TFIDF` 或 `RULE`。`AUTO` 会按照
BERT、TF-IDF、规则的顺序自动降级。`BERT` 表示优先尝试 BERT；如果 BERT 权重、依赖或设备不可用，
系统仍会安全降级，并在模型评估页展示实际运行后端。`BERT_DEVICE=AUTO` 会优先使用 CUDA，没有 GPU 时使用 CPU。

仓库保留 BERT 配置、Tokenizer 和指标文件，但约 711 MB 的 `model.safetensors` 不提交到 Git。
队友运行最终模型时，需要将权重单独放入 `outputs/bert_model_final/`，并设置上述
`BERT_MODEL_PATH`。没有权重时，系统仍可通过 TF-IDF 或规则运行。

LLM 配置：

```text
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
LLM_MODEL=ecnu-plus
LLM_TIMEOUT=20
```

不要提交包含真实密钥的 `.env`。没有 API Key 或接口异常时，系统会自动生成本地模板建议。

## 数据与划分

训练数据由 9000 条 Coursera 英文课程评论和 120 条人工构造的中文课程评价组成，标签均为
`negative`、`neutral`、`positive`。中文样本覆盖教学内容、授课方式、作业任务、考试安排、
实验实践和学习收获等课程维度。

| 数据 | 数量 | 用途 |
|---|---:|---|
| Coursera 英文课程评论 | 9000 条，每类 3000 条 | 模型训练与量化评估 |
| 中文课程评价 | 120 条，每类 40 条 | 双语训练、中文流程验证和演示 |
| 固定案例 | 12 条 | 演示流程验证与关键业务场景验证 |
| 独立压力测试 | 48 条，每类场景 6 条 | 复杂表达和鲁棒性评估 |

BERT 训练固定拆分为 5472 条训练集、1368 条验证集和 2280 条测试集。验证集用于选择最佳
checkpoint，测试集只在训练完成后评估一次。

重新构建数据：

```powershell
python scripts\prepare_coursera_dataset.py --input data\raw\coursera_reviews_label_3.csv --output data\processed\coursera_reviews_sampled.csv --per-label 3000 --min-chars 20 --max-chars 1200
python scripts\build_bilingual_dataset.py --per-label 3000 --min-chars 20 --max-chars 1200
```

英文原始数据来自 Hugging Face 数据集 `MungunshagaiT/coursera-reviews`。`data/raw/` 不提交到 Git。

## 模型训练

训练传统模型：

```powershell
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| Dummy Most Frequent | 0.3333 | 0.1667 |
| Logistic Regression | **0.6904** | **0.6906** |
| Naive Bayes | 0.6772 | 0.6785 |
| Linear SVM | 0.6811 | 0.6795 |

传统模型会生成：

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
models/model_metrics.json
```

训练 multilingual BERT：

```powershell
python -m src.train_bert --data data/processed/bilingual_reviews_train.csv --model-name bert-base-multilingual-cased --output-dir outputs/bert_model_final --metrics-path outputs/bert_metrics_final.json
```

最终 BERT 测试结果：

| 模型 | Accuracy | Macro-F1 |
|---|---:|---:|
| multilingual BERT | **0.7461** | **0.7457** |

## 消融实验

运行 12 条固定案例：

```powershell
python scripts\run_ablation_experiment.py --cases data\test_cases.csv --output-dir outputs\reports\final_model --model-dir models
```

| 实验版本 | Accuracy | Macro-F1 | 通过数 |
|---|---:|---:|---:|
| rule-only | 1.0000 | 1.0000 | 12/12 |
| model-only（TF-IDF） | 0.5833 | 0.5424 | 7/12 |
| bert-only | 0.9167 | 0.9221 | 11/12 |
| hybrid | **1.0000** | **1.0000** | **12/12** |

固定案例用于验证演示流程和关键业务场景，其中包含较多规则能够覆盖的典型表达，因此不能用 12/12
替代独立泛化能力结论。

输出文件：

```text
outputs/reports/final_model/ablation_metrics.json
outputs/reports/final_model/ablation_errors.csv
```

## 独立压力测试

`data/stress_test_cases.csv` 共 48 条，覆盖缩写、拼写错误、否定、讽刺、隐含抱怨、中英混合、
长文本和混合情感。该数据不参与模型训练，也不用于固定业务规则的反复调试。

```powershell
python scripts\run_stress_test.py --cases data\stress_test_cases.csv --output-dir outputs\reports\stress_test_final --model-dir models
```

| 实验版本 | Accuracy | Macro-F1 | 通过数 |
|---|---:|---:|---:|
| rule-only | 0.4583 | 0.4705 | 22/48 |
| tfidf-only | 0.5417 | 0.5282 | 26/48 |
| bert-only | 0.5417 | 0.5416 | 26/48 |
| hybrid | **0.7083** | **0.7217** | **34/48** |

混合流程在中英混合类别通过 6/6，在缩写、拼写、否定和长文本类别均通过 5/6。讽刺和隐含抱怨
仍是主要改进方向。

输出文件：

```text
outputs/reports/stress_test_final/stress_metrics.json
outputs/reports/stress_test_final/stress_predictions.csv
outputs/reports/stress_test_final/stress_errors.csv
```

## 报告维护

最终报告不再使用 `docs/report.md`，统一维护 `report/` 下的 LaTeX 文件：

```text
report/main.tex                    报告入口
report/info.tex                    封面成员与日期
report/report.sty                  字体、字号、行距、目录和封面格式
report/sections/                   独立章节
report/figures/                    实验图表和真实系统截图
report/main.pdf                    最新可提交 PDF
```

重新生成实验图表：

```powershell
.\.venv\Scripts\python.exe scripts\generate_report_charts.py
```

重新编译报告：

```powershell
Set-Location report
.\build-report.ps1
```

中间文件位于被忽略的 `report/build/`，脚本会把最终 PDF 复制到可提交、可推送的
`report/main.pdf`。详细协作规则见 `report/README.md`。

## 提交清单

建议提交：

- 完整源代码、`README.md`、依赖文件和 `.env.example`
- `data/processed/`、`data/test_cases.csv`、`data/stress_test_cases.csv`
- `models/` 下的传统模型、向量器和指标
- `outputs/bert_metrics_final.json` 与 `outputs/reports/` 下的实验结果
- `report/` 下的 LaTeX 源码、图表、截图和 `main.pdf`
- 单独制作的项目展示 PPT 和项目演示视频

不要提交：

- `.env` 和任何真实 API Key
- `data/raw/`
- BERT 的 `model.safetensors`、运行时压缩包和其他超大文件
- `.venv/`、`__pycache__/`、测试临时文件和编辑器缓存
