from __future__ import annotations

import logging
from typing import Any

from book_see_rag.config import get_settings
from book_see_rag.embedding.embedder import embed_documents, embed_query
from book_see_rag.retrieval import keyword_rank_hits, merge_ranked_hits, section_window_hits, sentence_window_hits
from book_see_rag.retrievers.scoped_documents import load_hits_from_uploads
from book_see_rag.vectorstore.milvus_store import SearchHit, get_doc_hits

logger = logging.getLogger("uvicorn.error")


def _vector_subqueries(query: str) -> list[str]:
    """
    Build focused embedding queries: first user line + 检索关注 sub-clauses.
    Multi-line expansion improves recall for contrast/definition questions without
    polluting the primary keyword/BM25 path (still uses full query string there).
    """
    raw = (query or "").strip()
    if not raw:
        return []
    primary = raw.split("\n", 1)[0].strip() or raw
    out: list[str] = [primary]
    if "\n" not in raw:
        return out
    tail = raw.split("\n", 1)[1]
    if "检索关注：" not in tail:
        return out
    blob = tail.split("检索关注：", 1)[1].strip()
    for piece in blob.replace(";", "；").split("；"):
        p = piece.strip()
        if p and p not in out:
            out.append(p)
    return out


class LlamaIndexUnavailable(RuntimeError):
    pass


def _load_llamaindex() -> tuple[Any, Any, Any]:
    try:
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import BaseEmbedding
    except ImportError as exc:
        raise LlamaIndexUnavailable(
            "LlamaIndex 未安装。请安装 llama-index-core 后再启用 retrieval_backend=llamaindex。"
        ) from exc
    return Document, VectorStoreIndex, BaseEmbedding


def _build_embedding_model(base_embedding: Any):
    class ExistingEmbedding(base_embedding):
        def _get_query_embedding(self, query: str) -> list[float]:
            return embed_query(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return embed_documents([text])[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return embed_documents(texts)

    return ExistingEmbedding()


def search_hits_with_llamaindex(query: str, doc_ids: list[str] | None = None) -> list[SearchHit]:
    """
    Experimental LlamaIndex retriever.

    It intentionally adapts back to the project's SearchHit shape so API/UI,
    permission filtering, citations, and memory can stay unchanged.
    """
    settings = get_settings()
    if not doc_ids:
        logger.info("LlamaIndex retriever skipped because scoped doc_ids are required")
        return []

    Document, VectorStoreIndex, BaseEmbedding = _load_llamaindex()
    keyword_pool_limit = max(
        getattr(settings, "retrieval_keyword_pool_limit", 1000),
        settings.llamaindex_candidate_limit,
        settings.rerank_top_n * 10,
    )
    try:
        keyword_pool = get_doc_hits(doc_ids, limit=keyword_pool_limit)
    except Exception:
        logger.exception("LlamaIndex failed to load chunks from Milvus, using upload fallback")
        keyword_pool = []
    source_hits = keyword_pool[:settings.llamaindex_candidate_limit]
    if not source_hits:
        source_hits = load_hits_from_uploads(doc_ids, limit=settings.llamaindex_candidate_limit)
        keyword_pool = source_hits
    if not source_hits:
        return []

    documents = [
        Document(
            text=item["content"],
            metadata={
                "doc_id": item["doc_id"],
                "filename": item["filename"],
                "page": item["page"],
            },
        )
        for item in source_hits
    ]
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=_build_embedding_model(BaseEmbedding),
        show_progress=False,
    )
    retriever = index.as_retriever(similarity_top_k=settings.llamaindex_top_k)
    vector_hits: list[SearchHit] = []
    for subq in _vector_subqueries(query):
        batch: list[SearchHit] = []
        for node_with_score in retriever.retrieve(subq):
            node = node_with_score.node
            metadata = node.metadata or {}
            batch.append(
                {
                    "doc_id": metadata.get("doc_id", ""),
                    "filename": metadata.get("filename", ""),
                    "page": int(metadata.get("page") or 0),
                    "content": node.get_content(metadata_mode="none"),
                    "score": float(node_with_score.score or 0.0),
                }
            )
        vector_hits = merge_ranked_hits(vector_hits, batch)
    window_top_k = getattr(settings, "retrieval_window_top_k", settings.llamaindex_top_k)
    section_hits = section_window_hits(query, keyword_pool, window_top_k)
    sentence_hits = sentence_window_hits(query, keyword_pool, window_top_k)
    keyword_hits = keyword_rank_hits(query, keyword_pool, window_top_k)
    hits = merge_ranked_hits(section_hits, sentence_hits, keyword_hits, vector_hits)
    logger.info(
        "LlamaIndex hybrid retriever finished source_hits=%s keyword_pool=%s section_hits=%s sentence_hits=%s keyword_hits=%s vector_hits=%s hits=%s",
        len(source_hits),
        len(keyword_pool),
        len(section_hits),
        len(sentence_hits),
        len(keyword_hits),
        len(vector_hits),
        len(hits),
    )
    return hits
