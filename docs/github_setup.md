# GitHub 仓库创建与推送说明

当前机器没有安装 GitHub CLI，因此推荐先在浏览器创建远程仓库，再把本地仓库推上去。

## 1. 在 GitHub 创建仓库

建议仓库名：

```text
course-feedback-nlp-llm
```

创建时建议：

- Visibility：按课程要求选择 Public 或 Private
- 不要勾选 Initialize this repository with README
- 不要添加 `.gitignore`
- 不要添加 License

因为本地项目已经包含这些文件。

## 2. 本地初始化并提交

在项目根目录执行：

```bash
git init
git add .
git commit -m "Initial project scaffold"
```

如果提示没有配置用户名和邮箱，先执行：

```bash
git config user.name "你的名字"
git config user.email "你的邮箱"
```

## 3. 关联远程仓库

把下面 URL 换成你自己的 GitHub 仓库地址：

```bash
git remote add origin https://github.com/your-name/course-feedback-nlp-llm.git
git branch -M main
git push -u origin main
```

## 4. 邀请队友

进入 GitHub 仓库页面：

```text
Settings -> Collaborators -> Add people
```

输入队友 GitHub 用户名或邮箱并发送邀请。

## 5. 队友拉取项目

队友接受邀请后执行：

```bash
git clone https://github.com/your-name/course-feedback-nlp-llm.git
cd course-feedback-nlp-llm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 6. 推荐协作方式

每个人新建功能分支：

```bash
git checkout -b feature/your-task
```

完成后提交并推送：

```bash
git add .
git commit -m "feat: describe your change"
git push -u origin feature/your-task
```

然后在 GitHub 上创建 Pull Request，经过队友检查后合并到 `main`。
