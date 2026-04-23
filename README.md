# book-see-rag

基于 RAG 的文档问答系统，支持多格式文档解析、向量化存储、多轮对话和异步任务处理。

## 功能特性

- **多格式文档解析**：支持 PDF、DOCX、TXT、MD 格式
- **两阶段检索**：ANN 召回 + Cross-Encoder 精排
- **向量化模型**：BGE-large-zh-v1.5 (1024 维)
- **重排序模型**：bge-reranker-v2-m3
- **多轮对话**：基于 Redis 的 Session 历史管理
- **异步任务**：Celery + Redis 处理文档 ingestion
- **多种 LLM 支持**：vLLM / Claude / GPT-4

## 系统架构

```
                    ┌─────────────┐
                    │  Streamlit  │
                    │      UI     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   FastAPI   │
                    │    API      │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
 ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
 │  Celery     │    │   Milvus    │    │   Redis     │
 │  Worker     │    │  VectorDB   │    │  Memory     │
 │  (Ingest)   │    │             │    │             │
 └─────────────┘    └─────────────┘    └─────────────┘
```

## 目录结构

```
book-see-rag/
├── src/book_see_rag/
│   ├── api/              # FastAPI 路由
│   ├── chains/           # LangChain 链 (QA/Chat/Summary)
│   ├── embedding/        # BGE 向量化和重排序
│   ├── ingestion/        # 文档解析和分块
│   ├── llm/              # LLM 工厂
│   ├── memory/           # Redis Session 记忆
│   ├── retriever/       # 检索器
│   ├── tasks/            # Celery 异步任务
│   ├── vectorstore/      # Milvus 存储
│   └── config.py         # 配置管理
├── tests/                # 单元测试
├── ui/                   # Streamlit 前端
├── docker/               # Docker 文件
└── docker-compose.yml    # 容器编排
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- Milvus 2.4+
- Redis 7+
- GPU (推荐)

### 2. 安装依赖

```bash
pip install -e .
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 (vllm/claude/openai) | vllm |
| `VLLM_BASE_URL` | vLLM 服务地址 | http://localhost:8000/v1 |
| `MILVUS_HOST` | Milvus 主机地址 | localhost |
| `MILVUS_PORT` | Milvus 端口 | 19530 |
| `EMBED_MODEL` | Embedding 模型 | BAAI/bge-large-zh-v1.5 |
| `RERANKER_MODEL` | 重排序模型 | BAAI/bge-reranker-v2-m3 |
| `BGE_DEVICE` | BGE 设备 (cuda/cpu/mps) | cuda |
| `REDIS_URL` | Redis 连接地址 | redis://localhost:6379/0 |

### 4. 启动服务

**本地开发：**

```bash
# 启动 API 服务
uvicorn book_see_rag.api.main:app --reload

# 启动 Celery Worker
celery -A book_see_rag.tasks.ingest_task worker --loglevel=info

# 启动 Streamlit UI (新终端)
streamlit run ui/app.py
```

**Docker Compose：**

```bash
# 启动基础设施 (Milvus, Redis, etcd, MinIO)
docker-compose --profile infra up -d

# 启动应用
docker-compose --profile app up -d
```

### 5. 访问

- API 文档：http://127.0.0.1:8000/docs
- Streamlit UI：http://127.0.0.1:8501

## API 接口

### 文档摄入

```bash
# 上传文档
POST /api/ingest
Content-Type: multipart/form-data

file: <your-document.pdf>

# 查询任务状态
GET /api/ingest/{task_id}
```

### 问答

```bash
# 单轮问答
POST /api/query
{
    "question": "这本书讲了什么？",
    "doc_ids": ["optional-doc-id-filter"]
}

# 多轮对话
POST /api/chat
{
    "session_id": "user-session-id",
    "message": "深度学习怎么入门？"
}
```

### 文档管理

```bash
# 获取文档列表
GET /api/documents

# 删除文档
DELETE /api/documents/{doc_id}
```

## 检索流程

```
用户查询
    │
    ▼
1. 查询向量化 (BGE)
    │
    ▼
2. ANN 召回 (Milvus HNSW, Top-50)
    │
    ▼
3. Cross-Encoder 精排 (Top-10)
    │
    ▼
4. LLM 生成答案
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `chunk_size` | 分块大小 (tokens) | 512 |
| `chunk_overlap` | 分块重叠 (tokens) | 128 |
| `rerank_top_n` | 召回数量 | 50 |
| `rerank_top_k` | 精排后返回数量 | 10 |
| `enable_rerank` | 是否启用重排序 | true |

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码格式化

```bash
make format
make lint
```

## 技术栈

- **框架**：LangChain, FastAPI, Streamlit
- **向量数据库**：Milvus
- **缓存/消息队列**：Redis, Celery
- **文档解析**：pdfplumber, marker-pdf
- **Embedding**：FlagEmbedding (BGE)
- **LLM**：vLLM, Anthropic, OpenAI

## License

MIT
