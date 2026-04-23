from functools import lru_cache
import logging
import time
from FlagEmbedding import FlagReranker
from book_see_rag.config import get_settings

logger = logging.getLogger("uvicorn.error")


@lru_cache(maxsize=1)
def _load_reranker() -> FlagReranker:
    settings = get_settings()
    preferred_device = settings.bge_device
    fallback_devices = []
    if preferred_device != "cpu":
        fallback_devices.append("cpu")

    for device in [preferred_device, *fallback_devices]:
        use_fp16 = device.startswith("cuda")
        logger.info(
            "Loading reranker model model=%s device=%s fp16=%s",
            settings.reranker_model,
            device,
            use_fp16,
        )
        try:
            return FlagReranker(
                settings.reranker_model,
                use_fp16=use_fp16,
                device=device,
            )
        except Exception:
            logger.exception(
                "Loading reranker model failed model=%s device=%s",
                settings.reranker_model,
                device,
            )

    raise RuntimeError(f"无法加载 reranker 模型: {settings.reranker_model}")


def rerank(query: str, chunks: list[str], top_k: int | None = None) -> list[str]:
    """
    Cross-Encoder 精排。
    返回按相关度降序排列的 top_k 个 chunk 文本。
    """
    if not chunks:
        return []

    settings = get_settings()
    top_k = top_k or settings.rerank_top_k

    started = time.perf_counter()
    logger.info("Reranking started chunks=%s top_k=%s", len(chunks), top_k)
    reranker = _load_reranker()
    pairs = [[query, chunk] for chunk in chunks]
    scores = reranker.compute_score(pairs, normalize=True)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    logger.info("Reranking finished elapsed=%.3fs", time.perf_counter() - started)
    return [chunk for _, chunk in ranked[:top_k]]
