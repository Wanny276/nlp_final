# 项目文档目录

本目录保存 CourseInsight 的开发、数据、测试和提交材料说明。根目录 [README.md](../README.md) 作为项目入口，本文档用于快速定位细节材料。

## 推荐阅读顺序

1. [../README.md](../README.md)：项目简介、运行方式、训练命令和文档导航；
2. [sentiment_labeling_guidelines.md](sentiment_labeling_guidelines.md)：情感标签判定标准，报告和测试以此为准；
3. [dataset_preparation.md](dataset_preparation.md)：中文人工数据、Coursera 数据、双语训练集构建说明；
4. [team_work_plan.md](team_work_plan.md)：两人小组分工和开发阶段安排；
5. [submission_checklist.md](submission_checklist.md)：期末提交前逐项检查；
6. [bilingual_upgrade_plan.md](bilingual_upgrade_plan.md)：中英双语版本的设计说明；
7. [github_setup.md](github_setup.md)：GitHub 仓库创建和推送说明；
8. [next_steps_work_plan.txt](next_steps_work_plan.txt)：阶段性交接计划，保留作历史参考。

## 文档职责

| 文档 | 用途 |
|---|---|
| `README.md` | 项目入口、快速开始、训练和运行说明 |
| `sentiment_labeling_guidelines.md` | positive / neutral / negative 的统一判定规则 |
| `dataset_preparation.md` | 数据来源、抽样、清洗和训练集构建 |
| `team_work_plan.md` | 成员 A/B 分工、开发顺序、演示分工 |
| `submission_checklist.md` | 最终提交材料和演示检查 |
| `bilingual_upgrade_plan.md` | 中英双语功能改造说明 |
| `github_setup.md` | GitHub 初始化和协作推送说明 |
| `next_steps_work_plan.txt` | 已完成阶段的交接记录 |

## 维护原则

- 安装、运行、训练等高频入口信息放在根目录 `README.md`；
- 详细数据、标签、分工、提交要求放在 `docs/`；
- 情感标签定义以 `sentiment_labeling_guidelines.md` 为唯一准则；
- 阶段性计划可以保留，但当前任务以 `team_work_plan.md` 和 `submission_checklist.md` 为准。

