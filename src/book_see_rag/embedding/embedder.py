from functools import lru_cache
import logging
import time
from FlagEmbedding import FlagModel
from book_see_rag.config import get_settings

logger = logging.getLogger("uvicorn.error")


@lru_cache(maxsize=1)
def _load_model() -> FlagModel:
    settings = get_settings()
    preferred_device = settings.bge_device
    fallback_devices = []
    if preferred_device != "cpu":
        fallback_devices.append("cpu")

    for device in [preferred_device, *fallback_devices]:
        use_fp16 = device.startswith("cuda")
        logger.info(
            "Loading embedding model model=%s device=%s fp16=%s",
            settings.embed_model,
            device,
            use_fp16,
        )
        try:
            return FlagModel(
                settings.embed_model,
                use_fp16=use_fp16,
                device=device,
            )
        except Exception:
            logger.exception(
                "Loading embedding model failed model=%s device=%s",
                settings.embed_model,
                device,
            )

    raise RuntimeError(f"无法加载 embedding 模型: {settings.embed_model}")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """文档向量化（不加查询前缀）"""
    logger.info("Embedding %s chunks", len(texts))
    model = _load_model()
    return model.encode(texts, batch_size=32).tolist()


def embed_query(query: str) -> list[float]:
    """查询向量化（加 bge 查询指令前缀）"""
    return list(_embed_query_cached(query.strip()))


@lru_cache(maxsize=256)
def _embed_query_cached(query: str) -> tuple[float, ...]:
    started = time.perf_counter()
    logger.info("Embedding query started query=%r", query[:80])
    model = _load_model()
    vector = tuple(model.encode_queries([query])[0].tolist())
    logger.info("Embedding query finished elapsed=%.3fs", time.perf_counter() - started)
    return vector
