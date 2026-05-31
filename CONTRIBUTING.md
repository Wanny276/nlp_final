# 团队协作约定

## 分支命名

- `feature/xxx`：新增功能
- `fix/xxx`：修复问题
- `docs/xxx`：文档修改
- `test/xxx`：测试相关

示例：

```bash
git checkout -b feature/topic-analyzer
```

## 提交信息

建议使用简洁中文或英文：

```text
feat: add topic analyzer
fix: handle empty csv input
docs: update deployment steps
test: add preprocess tests
```

## Pull Request 检查项

提交 PR 前确认：

- 代码可以运行
- 单元测试通过
- README 或文档已同步更新
- 没有提交 `.env`、模型大文件或无关缓存

## 代码风格

- 业务逻辑尽量放在 `src/`
- `app.py` 只负责页面展示和调用模块
- 新功能尽量配一个最小测试
- API Key 只能放在本地 `.env`，不要提交到仓库
