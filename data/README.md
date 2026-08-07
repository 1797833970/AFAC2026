# 数据目录说明

```text
data/
├── README.md                  # 本文件
├── upload_b/                  # 官方 B 榜文件（随项目提供）
│   ├── readme.md              # 官方提交说明
│   ├── submit.csv             # 官方提交模板
│   └── question_b/            # B 榜题目（5 领域 × 20 题）
├── doc_registry.json          # 文档注册表：doc_id → 标题 / chunk 数
├── raw/                       # 原始数据（gitignore，自行下载）
├── chunks/                    # 分块产物（gitignore，脚本生成）
├── index/                     # BM25 索引（gitignore，脚本生成）
├── processed_md/              # MinerU 解析产物（gitignore）
├── processed/                 # 章节树解析产物（gitignore）
└── temp/                      # 临时目录（gitignore）
```

## 随项目提供的部分

- `upload_b/`：官方 B 榜题目与提交模板，体积小（约 60KB），随项目提供，方便读者直接跑 `build_submission.py` 等流程。
- `doc_registry.json`：文档元数据，Agent 的 Prompt 构造依赖它。

## 需自行准备的部分

| 目录 | 原因 | 如何获得 |
|------|------|----------|
| `raw/` | 原始数据约 340MB，版权属于赛题方 | 从赛题页面下载后放入 |
| `chunks/` | 生成产物约 60MB | `chunker_md.py` + `split_chunks.py` |
| `index/` | 生成产物约 270MB | `scripts/build_index.py build` |
| `processed_md/` | 生成产物 | `data_processing/mineru.py` |
| `processed/` | 生成产物 | `data_processing/document_parser.py` |

完整复现步骤见 [docs/07-复现.md](../docs/07-复现.md)。
