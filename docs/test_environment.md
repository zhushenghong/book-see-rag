# 检索准确率测试环境

## 测试时间

2026-04-23

---

## 硬件配置

### 本地开发机 (Mac)

| 项目 | 参数 |
|------|------|
| **型号** | Mac |
| **操作系统** | macOS 26.2 (arm64) |
| **CPU** | Apple M4 (4 性能核 + 4 能效核) |
| **内存** | 24 GB |
| **磁盘** | 228 GB (已用 44%) |
| **GPU** | Apple M4 集成 GPU (10 核, Metal 4) |

### IP 地址

| 接口 | 地址 |
|------|------|
| 主网口 | 178.18.1.78 |
| Docker 内网 | 198.18.0.1 |

---

## 服务依赖

### 向量数据库

| 项目 | 地址 |
|------|------|
| **Milvus** | 192.168.11.135:19530 |
| **Collection** | book_see_rag |

### LLM 服务

| 项目 | 地址 |
|------|------|
| **vLLM** | 192.168.11.199:8988/v1 |
| **模型** | qwen-rag / qwen-coder |

### 缓存 & 消息队列

| 项目 | 地址 |
|------|------|
| **Redis** | 192.168.11.135:6379/0 |

---

## 模型配置

### Embedding & Reranker

| 模型 | 设备 | 配置 |
|------|------|------|
| BAAI/bge-large-zh-v1.5 | MPS (Apple GPU) | 1024 维向量 |
| BAAI/bge-reranker-base | MPS | Cross-Encoder |

### 检索参数

| 参数 | 值 |
|------|---|
| rerank_top_n (召回) | 10 |
| rerank_top_k (精排) | 3 |
| enable_rerank | false |
| chunk_size | 512 |
| chunk_overlap | 128 |

---

## 测试数据集

| 文件 | 说明 |
|------|------|
| `docs/rag_quality_eval_set.json` | 评估问题集 |
| `docs/rag_quality_test.md` | 测试文档 |

---

## 测试脚本

```bash
python scripts/run_rag_benchmark.py
```

输出结果: `benchmark_results/rag_benchmark_<timestamp>.md`
