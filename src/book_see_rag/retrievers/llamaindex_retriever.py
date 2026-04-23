from __future__ import annotations

import logging
from typing import Any

from book_see_rag.config import get_settings
from book_see_rag.embedding.embedder import embed_documents, embed_query
from book_see_rag.retrievers.scoped_documents import load_hits_from_uploads
from book_see_rag.vectorstore.milvus_store import SearchHit, get_doc_hits

logger = logging.getLogger("uvicorn.error")


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
    try:
        source_hits = get_doc_hits(doc_ids, limit=settings.llamaindex_candidate_limit)
    except Exception:
        logger.exception("LlamaIndex failed to load chunks from Milvus, using upload fallback")
        source_hits = []
    if not source_hits:
        source_hits = load_hits_from_uploads(doc_ids, limit=settings.llamaindex_candidate_limit)
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
    nodes = index.as_retriever(similarity_top_k=settings.llamaindex_top_k).retrieve(query)

    hits: list[SearchHit] = []
    for node_with_score in nodes:
        node = node_with_score.node
        metadata = node.metadata or {}
        hits.append(
            {
                "doc_id": metadata.get("doc_id", ""),
                "filename": metadata.get("filename", ""),
                "page": int(metadata.get("page") or 0),
                "content": node.get_content(metadata_mode="none"),
                "score": float(node_with_score.score or 0.0),
            }
        )
    logger.info("LlamaIndex retriever finished source_hits=%s hits=%s", len(source_hits), len(hits))
    return hits
