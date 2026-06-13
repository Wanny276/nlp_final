# CourseInsight 期末报告

```text
report/
├─ main.tex          报告入口，一般不需要修改
├─ main.pdf          最新编译版，可提交到 Git 供成员预览
├─ build-report.ps1  统一编译脚本
├─ info.tex          封面标题、成员和日期
├─ report.sty        与教师 Word 模板对应的统一格式
├─ sections/         各成员独立编辑的报告板块
└─ figures/          界面截图和自动生成的实验图表
```

## 协作方式

每位成员只编辑自己负责的 `sections/*.tex` 文件。不要在章节文件中修改
页边距、字体和标题格式；全局格式统一由 `report.sty` 管理。

报告排版统一遵循教师模板：

- 正文使用宋体小四、1.5 倍行距；
- 一级标题为加粗宋体三号；
- 二级标题为加粗宋体小三；
- 三级标题为加粗宋体四号；
- 标题层级不得超过三级；
- 图表必须使用 `\caption{...}` 编号并附文字说明；
- 目录由 `main.tex` 自动生成，不要手工填写标题或页码。

提交前需要完成两处人工确认：

1. 在 `info.tex` 中填写真实姓名、学号和学院。
2. 在 `sections/09-contributions.tex` 中确认分工与贡献比例，比例合计为 100%。
3. 单独准备项目展示 PPT、10 分钟演示流程和项目演示视频。

## 编译

如实验 JSON、训练日志或数据集发生变化，先在项目根目录重新生成图表：

```powershell
.venv\Scripts\python.exe scripts\generate_report_charts.py
```

在 `report` 目录执行：

```powershell
.\build-report.ps1
```

脚本会把 LaTeX 中间文件保留在被忽略的 `build/` 目录，并将最新版报告更新到
可提交的 `report/main.pdf`。修改报告后，请重新运行脚本并一同提交对应的
`sections/*.tex` 和 `main.pdf`。
