# 期末提交检查清单

提交前建议先阅读根目录 [README.md](../README.md)。情感标签判定和测试准则以 [sentiment_labeling_guidelines.md](sentiment_labeling_guidelines.md) 为准。

## 项目本身

- [ ] Streamlit 系统可以运行
- [ ] 单条评价分析可以演示
- [ ] CSV 批量分析可以演示
- [ ] 中文评价和英文评论都可以演示
- [ ] 情感分类结果可解释
- [ ] 混合评价判定符合 `neutral` 标签准则
- [ ] 主题识别结果可展示
- [ ] 关键词提取结果可展示
- [ ] 大模型总结或本地兜底可展示
- [ ] 至少 5 个测试用例有结果截图或表格
- [ ] 模型与技术说明页面可截图

## 代码材料

- [ ] `README.md` 包含安装和运行说明
- [ ] `requirements.txt` 完整
- [ ] `.env.example` 提供 API 配置模板
- [ ] 示例数据已放入 `data/`
- [ ] `models/model_metrics.json` 可由训练命令生成
- [ ] 修改预处理、训练数据、模型参数或 scikit-learn 环境后已重新训练模型
- [ ] 测试代码已放入 `tests/`
- [ ] 不提交 `.env`
- [ ] 不提交 `data/raw/`、无关缓存和大文件
- [ ] 默认不提交 `models/*.pkl`，除非最终压缩包演示需要

模型训练命令：

```bash
python -m src.train_model --data data/processed/bilingual_reviews_train.csv --model-dir models
```

## 汇报材料

- [ ] 报告标明贡献度
- [ ] PPT 是报告的简化版
- [ ] 演示视频包含单条分析和批量分析
- [ ] 报告说明课程知识点对应：预处理、n-gram/TF-IDF、文本分类、Logistic Regression
- [ ] 报告说明 API 失败时的本地兜底方案
- [ ] 准备 3 到 5 个答辩问题回答
