# 硬隔离多租户落地说明

## 入口路由

采用子域名路由：`https://{tenant_slug}.{TENANT_BASE_DOMAIN}`。

网关负责把外部请求路由到该租户独立的 API 栈；应用默认支持两种身份来源：

- `AUTH_MODE=headers`：本地开发或受信任网关模式，使用 `X-Tenant-Id`、`X-User-Id`、`X-Role`、`X-Department`。
- `AUTH_MODE=jwt`：生产鉴权模式，从 `Authorization: Bearer <token>` 中读取用户与租户信息，默认租户 claim 为 `tenant_id`。

控制面生产环境必须使用 `AUTH_MODE=jwt`。Header 模式只允许本地调试，并且必须显式设置：

```bash
export ALLOW_INSECURE_CONTROL_PLANE_HEADERS=true
```

数据面默认会检查 `CONTROL_PLANE_DATA_DIR` 中已存在租户的状态：如果租户被禁用则拒绝访问。若需要强制所有请求租户都必须先在控制面注册，可设置：

```bash
export REQUIRE_TENANT_REGISTERED=true
```

## 数据面隔离

每个租户独立部署：

- API
- Celery worker
- Milvus
- Redis
- metadata DB（MySQL 推荐）
- MinIO/S3 等对象存储

`docker/docker-compose.tenant.example.yml` 提供单租户全栈部署模板。每个租户使用独立 env、独立卷、独立数据库连接与独立向量库。

模板中的数据库、MinIO、JWT 公钥等均通过环境变量注入；不要把明文密码写进 compose 文件。可从 `docker/tenant.env.example` 复制出每个租户自己的 env 文件。

## Control Plane

控制面入口：

```bash
uvicorn book_see_rag.control_plane.main:app --host 0.0.0.0 --port 8080
```

或：

```bash
docker-compose --profile control-plane up -d
```

最小 API：

- `GET /health`
- `GET /tenants`
- `GET /tenants/{tenant_id}`
- `POST /tenants`
- `POST /tenants/{tenant_id}/disable`
- `POST /tenants/{tenant_id}/enable`
- `GET /tenants/{tenant_id}/audit`
- `DELETE /tenants/{tenant_id}`

创建、禁用、删除租户时，如果配置了 `TENANT_DEPLOY_WEBHOOK_URL`，控制面会向部署系统发送 webhook：

```json
{
  "event": "tenant.created",
  "payload": {
    "tenant_id": "...",
    "slug": "acme",
    "domain": "acme.rag.example.com",
    "status": "active"
  }
}
```

## 元数据迁移

默认仍使用 JSON 元数据，避免影响本地开发。启用 SQL 元数据：

```bash
export METADATA_BACKEND=sql
export METADATA_DB_URL='mysql+pymysql://rag:rag@mysql:3306/book_see_rag'
export DEFAULT_TENANT_ID='tenant-acme'
python scripts/migrate_metadata_json_to_db.py
```

迁移脚本会把 `data/metadata/knowledge_bases.json` 与 `data/metadata/documents.json` 导入当前租户数据库。

虽然硬隔离部署下每租户独立 DB 已经是主要边界，应用层仍会在知识库、文档和会话上携带 `tenant_id`。这样即使本地开发或误配置时多个租户短暂共用一套存储，也会被应用层过滤。

## PR 拆分边界

- PR1：`access_control.py`、配置项、鉴权/租户上下文。
- PR2：`metadata_sql.py`、`metadata_store.py`、JSON 到 SQL 迁移脚本。
- PR3：`control_plane/`、控制面 Dockerfile、控制面 compose profile。
- PR4：审计、部署模板与文档。

严格避免修改 `src/book_see_rag/chains/**` 与 `src/book_see_rag/retrieval.py`，降低与检索链路并行开发的冲突。

