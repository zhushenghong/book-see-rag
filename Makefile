.PHONY: dev-setup dev-api dev-worker dev-ui dev infra-up infra-down app-up app-down

# ── Mac 本地开发（原生 Python，推荐调试用）─────────────────────

dev-setup:
	@echo "安装依赖..."
	pip install -e ".[dev]"
	@echo "复制 Mac 环境配置..."
	@test -f .env || cp .env.mac .env
	@echo "请编辑 .env，将 <SERVER_IP> 替换为服务器实际 IP"
	mkdir -p data/uploads

dev-api:
	@echo "启动 FastAPI（热重载）..."
	PYTHONPATH=src uvicorn book_see_rag.api.main:app --reload --host 0.0.0.0 --port 8000

dev-worker:
	@echo "启动 Celery Worker..."
	PYTHONPATH=src celery -A book_see_rag.tasks.ingest_task.celery_app worker --pool=solo --loglevel=info --concurrency=1

dev-ui:
	@echo "启动 Streamlit UI..."
	streamlit run ui/app.py --server.port 8501

# 同时启动三个进程（需要 tmux 或分开三个终端）
dev:
	@echo "请在三个独立终端分别运行："
	@echo "  make dev-api"
	@echo "  make dev-worker"
	@echo "  make dev-ui"

# ── Docker 操作 ────────────────────────────────────────────────

# 服务器：启动基础设施
infra-up:
	docker-compose --profile infra up -d

infra-down:
	docker-compose --profile infra down

# Mac：用 Docker 启动应用层（连接远端基础设施）
app-up:
	docker-compose --profile app up -d

app-down:
	docker-compose --profile app down

# 生产：全部启动
prod-up:
	docker-compose --profile infra --profile app up -d

# ── 测试 ───────────────────────────────────────────────────────

test:
	PYTHONPATH=src pytest tests/ -v

test-cov:
	PYTHONPATH=src pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
