# 项目文档目录

本目录保存 CourseInsight 的开发、数据、测试和提交材料说明。根目录 [README.md](../README.md) 作为项目入口，本文档用于快速定位细节材料。

## 推荐阅读顺序

1. [../README.md](../README.md)：项目简介、运行方式、训练命令和文档导航；
2. [sentiment_labeling_guidelines.md](sentiment_labeling_guidelines.md)：情感标签判定标准，报告和测试以此为准；
3. [dataset_preparation.md](dataset_preparation.md)：中文人工数据、Coursera 数据、双语训练集构建说明；
4. [demo_materials.md](demo_materials.md)：固定演示输入、截图清单和报告/PPT 可用表述；
5. [submission_checklist.md](submission_checklist.md)：期末提交前逐项检查。

## 文档职责

| 文档 | 用途 |
|---|---|
| `README.md` | 项目入口、快速开始、训练和运行说明 |
| `sentiment_labeling_guidelines.md` | positive / neutral / negative 的统一判定规则 |
| `dataset_preparation.md` | 数据来源、抽样、清洗和训练集构建 |
| `demo_materials.md` | 演示输入、截图清单、报告/PPT 文案 |
| `submission_checklist.md` | 最终提交材料和演示检查 |

## 维护原则

- 安装、运行、训练等高频入口信息放在根目录 `README.md`；
- 详细数据、标签、演示材料和提交要求放在 `docs/`；
- 情感标签定义以 `sentiment_labeling_guidelines.md` 为唯一准则；
- 已过期的阶段计划、分支建议和旧数据规模说明不保留，避免和当前事实冲突。
