# 07 · 从零复现与 FAQ

## 7.1 完整复现步骤

```bash
# 1) 克隆仓库
git clone <your-repo-url>
cd AFAC2026

# 2) 安装依赖
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) 配置密钥
cp .env.example .env             # 填入 DASHSCOPE_API_KEY

# 4) 环境自检
python scripts/check_env.py

# 5) 放置官方数据（docs/03 说明目录结构）
#    data/raw/raw/<领域>/...

# 6) 解析 + 分块
python -m data_processing.mineru --limit 1    # PDF→Markdown（可选）
python scripts/split_chunks.py --apply         # 二次切分

# 7) 构建索引
python scripts/build_index.py build

# 8) 检索自检
python scripts/build_index.py search "身故保险金如何计算"

# 9) 跑 Agent（先小批量联调）
python scripts/run_agent.py --split B --domain insurance --limit 3

# 10) 全量 + 生成提交
python scripts/run_agent.py --split B
python scripts/build_submission.py --answers output/answers.json --out output/submission.csv
```

## 7.2 离线冒烟测试

仓库自带不依赖网络和 API Key 的测试：

```bash
python -m pytest tests/ -v
```

覆盖：BM25 小语料检索、答案格式化、Markdown 分块、提交 CSV 构建、官方题目 schema。

## 7.3 FAQ

### Q1：`check_env.py` 提示缺少分块文件怎么办？

`data/chunks/` 是生成产物，未纳入 Git。先放好原始数据，跑分块脚本，再 `python scripts/build_index.py build`。

### Q2：调用 DashScope 报网络/代理错误？

`agent/qa_agent.py` 在导入时会主动清理 HTTP 代理环境变量（`HTTP_PROXY`/`HTTPS_PROXY` 等）。如果本地确实需要通过代理访问，请调整该段代码而不是删掉它（这是排查"直连失败"的常用开关）。

### Q3：`jieba` 报 `pkg_resources` 弃用警告？

是上游依赖的已知警告，不影响运行，可忽略。

### Q4：换模型怎么改？

编辑 `config/config.yaml` 的 `model.name`，或在 `.env` 里设 `AFAC_MODEL`。注意赛题要求使用官方指定模型。

### Q5：5 线程并发内存不够？

每个线程会独立加载一次 BM25 索引（约 270MB），内存占用约等于 `270MB × 线程数`。机器内存紧张时用 `--workers 1`，或参考 `agent/settings.py` 调整索引加载方式。

### Q6：CSV 里答案带 BOM 或乱码？

提交文件用 `utf-8-sig` 写出（Excel 友好）。读取答案文件时脚本同样兼容 BOM。

### Q7：`.env` 会不小心提交吗？

不会。`.gitignore` 已排除 `.env`。提交前可用 `git status` 复核，**泄露 API Key 的补救方式是立刻在百炼控制台吊销重建**。

### Q8：为什么 `data/raw/`、`data/index/` 不在仓库里？

原始数据体积大（约 340MB）且版权属于赛题方；索引（约 270MB）和分块是生成产物。仓库遵循"源码 + 文档 + 小体积官方文件"的原则，详见 `data/README.md`。

### Q9：提交时 `reasoning` 列要不要？

不要。官方格式只有 8 列，`reasoning` 只用于本地审计（`--with-reasoning`）。

## 7.4 已知限制

| 项目 | 说明 |
|------|------|
| 中文数字转换 | `_cn2int` 对"二十"这类写法支持有限，分块时可能误判编号 |
| 标题约定检测 | 只取前 3 个标题判断约定 A/B，边缘 case 可能误判 |
| Milvus Lite（Windows） | `create_index` 偶发文件锁问题，已绕过（实验路径使用 FAISS） |
| Token 估算 | 本地估算是近似值，以平台统计为准 |
