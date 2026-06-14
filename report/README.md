# CourseInsight 期末报告

本目录保存期末报告的 LaTeX 源码、实验图表、真实系统截图和最终 PDF。正式提交版为
`report/main.pdf`；`docs/report.md` 是早期文字稿，不再作为最终报告维护。

## 当前状态

- 报告共 28 页，A4 版式。
- 摘要已按要求删除，正文前包含自动目录。
- 正文为宋体小四、1.5 倍行距。
- 普通正文与标题后首段统一首行缩进 2 字符，列表、图表、公式、代码和参考文献不使用正文式首行缩进。
- 一级标题为加粗宋体三号，二级标题为加粗宋体小三，三级标题为加粗宋体四号。
- 标题层级不超过三级，图表均带编号和文字说明。
- 创新点占 2 页，界面展示占 3 页，思考与总结占 3 页。
- 界面章节使用单条分析、批量分析和固定案例验证三个真实运行截图。
- 封面成员信息和贡献度已经填写：王佳妮、柯文丽，各 50%。
- 最新 PDF 位于 `main.pdf`，可以提交到 Git 供成员预览。

## 目录结构

```text
report/
├─ main.tex                         报告入口，一般不需要修改
├─ main.pdf                         最新编译版，可提交和推送
├─ build-report.ps1                 统一编译脚本
├─ info.tex                         封面标题、成员和日期
├─ report.sty                       与教师 Word 模板对应的统一格式
├─ sections/
│  ├─ 01-background.tex            项目背景
│  ├─ 02-functions.tex             主要功能
│  ├─ 03-technical-solution.tex     技术方案、架构与数据流
│  ├─ 04-algorithms.tex             关键算法与模型原理
│  ├─ 04a-experiment-visualization.tex
│  │                                  实验结果可视化
│  ├─ 05-innovation.tex             创新点、消融与技术贡献
│  ├─ 06-interface.tex              界面设计与真实截图
│  ├─ 07-deployment-and-testing.tex 部署、源码范围和测试
│  ├─ 08-summary.tex                思考与总结
│  ├─ 09-contributions.tex          小组分工与贡献度
│  └─ 10-references.tex             参考文献
├─ figures/                         实验图表和系统截图
└─ build/                           LaTeX 中间文件，已被忽略
```

## 协作规则

每位成员只编辑自己负责的 `sections/*.tex` 文件。不要在章节中设置页边距、字体、行距或标题
样式，全局格式统一由 `report.sty` 管理。

- 章节只能使用 `\section`、`\subsection` 和 `\subsubsection`。
- 图表必须使用 `\caption{...}` 和 `\label{...}`。
- 引用图表时使用 `\ref{...}`，不要手工填写编号。
- 目录由 `main.tex` 自动生成，不要手工填写标题或页码。
- 修改实验数据后，应同时更新图表、正文指标和最终 PDF。
- 修改报告后应提交对应的 `.tex`、图片和 `main.pdf`，不要提交 `build/`。

`build-report.ps1` 会在编译前检查是否出现超过三级的标题命令。

## 图表与截图

实验图表由项目根目录下的脚本生成：

```powershell
.\.venv\Scripts\python.exe scripts\generate_report_charts.py
```

当前自动图表包括：

```text
dataset_distribution.png
model_performance.png
traditional_confusion_matrix.png
bert_confusion_matrix.png
bert_training_loss.png
bert_validation_metrics.png
ablation_results.png
stress_test_overall.png
stress_test_categories.png
```

报告使用的聚焦界面截图包括：

```text
ui-single-focus.png
ui-batch-focus.png
ui-test-focus.png
```

聚焦截图由以下脚本从完整运行截图中裁剪生成：

```powershell
.\.venv\Scripts\python.exe scripts\crop_report_screenshots.py
```

如果前端发生明显变化，应先替换完整截图，再重新运行裁剪脚本和报告编译脚本。

## 编译

本机需要安装带 XeLaTeX、latexmk 和中文字体支持的 TeX Live。在 `report` 目录执行：

```powershell
.\build-report.ps1
```

脚本会：

1. 检查标题层级是否超过三级。
2. 使用 XeLaTeX 编译 `main.tex`。
3. 将中间文件保存在 `build/`。
4. 将最终 PDF 更新到可提交的 `report/main.pdf`。

编译完成后，应检查：

- `main.pdf` 与 `build/main.pdf` 是否一致；
- 目录页码和交叉引用是否正确；
- 图表、表格和长 URL 是否超出版心；
- 封面成员信息、日期和贡献度是否仍然正确。

## 已覆盖的课程要求

- 项目背景、问题描述与实际需求
- 核心功能、模块划分、系统架构图、数据流图、场景与用户
- 技术选型及理由、关键算法与模型原理
- 创新性设计、消融实验、独立压力测试和技术贡献
- 3 页界面展示与界面设计理念
- 环境要求、安装步骤、源码提交范围和 8 个代表性测试用例
- 开发问题与解决方案、局限性、未来方向和个人反思
- 小组分工与 50%/50% 贡献度

PPT、10 分钟现场演示流程和演示视频是独立交付物，不放在本报告目录中。
