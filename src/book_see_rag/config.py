from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = "vllm"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen2.5-72B-Instruct"
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    openai_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "book_see_rag"

    # Embedding & Reranker
    embed_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    bge_device: str = "cuda"
    enable_rerank: bool = True
    rerank_top_n: int = 50
    rerank_top_k: int = 10
    retrieval_prefilter_top_k: int = 24
    retrieval_backend: str = "llamaindex"
    retrieval_keyword_pool_limit: int = 1000
    retrieval_window_top_k: int = 24
    llamaindex_candidate_limit: int = 80
    llamaindex_top_k: int = 24
    query_embedding_cache_size: int = 256

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    session_ttl: int = 86400  # 24h
    chat_history_window: int = 12
    followup_rewrite_history: int = 6

    # OCR
    marker_device: str = "cuda"

    # Storage
    upload_dir: str = "./data/uploads"
    metadata_dir: str = "./data/metadata"
    query_profiles_path: str = "./data/query_profiles.json"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Auth / Tenant
    # For production, prefer gateway or JWT-based auth. Header-based context is kept for local/dev compatibility.
    auth_mode: str = "headers"  # headers | jwt
    jwt_issuer: str = ""
    jwt_audience: str = ""
    jwt_secret: str = ""  # HS256 (dev). For RS256, use jwt_public_key_pem.
    jwt_public_key_pem: str = ""  # RS256
    tenant_header: str = "x-tenant-id"
    tenant_claim: str = "tenant_id"
    default_tenant_id: str = "public"
    allow_insecure_control_plane_headers: bool = False
    enforce_tenant_status: bool = True
    require_tenant_registered: bool = False

    # Metadata store
    metadata_backend: str = "json"  # json | sql
    metadata_db_url: str = ""  # e.g. mysql+pymysql://user:pass@host:3306/db

    # Control plane (tenant admin)
    control_plane_data_dir: str = "./data/control_plane"
    tenant_base_domain: str = "rag.example.com"
    tenant_deploy_webhook_url: str = ""  # if set, control plane will POST on tenant create/disable/delete

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 128
    ocr_min_chars_per_page: int = 50  # 低于此值触发 OCR
    noisy_char_ratio_threshold: float = 0.35
    repeated_char_threshold: int = 6
    min_clean_chunk_chars: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
