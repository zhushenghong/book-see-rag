import uuid
from functools import lru_cache
import logging
import time
from typing import TypedDict
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from langchain_core.documents import Document
from book_see_rag.config import get_settings
from book_see_rag.embedding.embedder import embed_documents, embed_query

logger = logging.getLogger("uvicorn.error")

COLLECTION_NAME = None  # 从 settings 读取


class SearchHit(TypedDict):
    doc_id: str
    filename: str
    page: int
    content: str
    score: float


def _normalize_content(content: str) -> str:
    return " ".join(content.split())


@lru_cache(maxsize=1)
def _connect():
    settings = get_settings()
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    settings = get_settings()
    name = settings.milvus_collection
    _connect()

    if not utility.has_collection(name):
        _create_collection(name)

    col = Collection(name)
    col.load()
    return col


def _create_collection(name: str):
    fields = [
        FieldSchema("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema("doc_id",   DataType.VARCHAR, max_length=64),
        FieldSchema("dense",    DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema("content",  DataType.VARCHAR, max_length=4096),
        FieldSchema("filename", DataType.VARCHAR, max_length=512),
        FieldSchema("page",     DataType.INT64),
        FieldSchema("is_ocr",   DataType.BOOL),
    ]
    schema = CollectionSchema(fields, description="book-see-rag chunks")
    col = Collection(name, schema)

    col.create_index(
        "dense",
        {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    return col


def insert_chunks(doc_id: str, filename: str, chunks: list[Document]) -> int:
    """向 Milvus 批量写入 chunks，返回写入数量"""
    col = _get_collection()

    texts = [c.page_content for c in chunks]
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    pages = [c.metadata.get("page", 0) for c in chunks]
    is_ocr_flags = [c.metadata.get("is_ocr", False) for c in chunks]
    logger.info("Preparing vectors doc_id=%s filename=%s chunks=%s", doc_id, filename, len(chunks))
    embed_started = time.perf_counter()
    vectors = embed_documents(texts)
    embed_elapsed = time.perf_counter() - embed_started
    logger.info(
        "Vectors ready doc_id=%s filename=%s vectors=%s elapsed=%.3fs",
        doc_id,
        filename,
        len(vectors),
        embed_elapsed,
    )

    entities = [
        chunk_ids,
        [doc_id] * len(chunks),
        vectors,
        [t[:4096] for t in texts],
        [filename] * len(chunks),
        pages,
        is_ocr_flags,
    ]
    logger.info("Inserting into Milvus doc_id=%s filename=%s rows=%s", doc_id, filename, len(chunks))
    insert_started = time.perf_counter()
    col.insert(entities)
    col.flush()
    insert_elapsed = time.perf_counter() - insert_started
    logger.info(
        "Milvus insert finished doc_id=%s filename=%s elapsed=%.3fs",
        doc_id,
        filename,
        insert_elapsed,
    )
    return len(chunks)


def search_hits(query: str, doc_ids: list[str] | None = None) -> list[SearchHit]:
    """
    两阶段检索：
    1. Milvus ANN 召回 top-N
    2. 由调用方调用 reranker 精排（解耦）
    返回结构化命中结果
    """
    started = time.perf_counter()
    col = _get_collection()
    settings = get_settings()
    if doc_ids == []:
        logger.info("Milvus search skipped query=%r because doc scope is empty", query[:80])
        return []

    vector = embed_query(query)
    expr = f"doc_id in {doc_ids}" if doc_ids else None

    results = col.search(
        data=[vector],
        anns_field="dense",
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=settings.rerank_top_n,
        expr=expr,
        output_fields=["doc_id", "filename", "page", "content"],
    )
    seen: set[tuple[str, int, str]] = set()
    chunks: list[SearchHit] = []
    for hit in results[0]:
        content = _normalize_content(hit.entity.get("content") or "")
        if not content:
            continue
        item: SearchHit = {
            "doc_id": hit.entity.get("doc_id") or "",
            "filename": hit.entity.get("filename") or "",
            "page": int(hit.entity.get("page") or 0),
            "content": content,
            "score": float(getattr(hit, "distance", 0.0) or 0.0),
        }
        dedupe_key = (item["doc_id"], item["page"], item["content"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        chunks.append(item)
    logger.info(
        "Milvus search finished query=%r doc_filter=%s hits=%s elapsed=%.3fs",
        query[:80],
        bool(doc_ids),
        len(chunks),
        time.perf_counter() - started,
    )
    return chunks


def search(query: str, doc_ids: list[str] | None = None) -> list[str]:
    return [item["content"] for item in search_hits(query, doc_ids=doc_ids)]


def get_doc_hits(doc_ids: list[str], limit: int = 12) -> list[SearchHit]:
    col = _get_collection()
    if not doc_ids:
        return []

    expr = f"doc_id in {doc_ids}"
    results = col.query(
        expr=expr,
        output_fields=["doc_id", "filename", "page", "content"],
        limit=limit,
    )
    seen: set[tuple[str, int, str]] = set()
    hits: list[SearchHit] = []
    for row in results:
        content = _normalize_content(row.get("content") or "")
        if not content:
            continue
        item: SearchHit = {
            "doc_id": row.get("doc_id") or "",
            "filename": row.get("filename") or "",
            "page": int(row.get("page") or 0),
            "content": content,
            "score": 0.0,
        }
        dedupe_key = (item["doc_id"], item["page"], item["content"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        hits.append(item)
    logger.info("Milvus doc fallback fetched doc_ids=%s hits=%s", len(doc_ids), len(hits))
    return hits


def delete_by_doc_id(doc_id: str):
    col = _get_collection()
    col.delete(f'doc_id == "{doc_id}"')
    col.flush()


def list_doc_ids() -> list[str]:
    col = _get_collection()
    results = col.query(expr="doc_id != ''", output_fields=["doc_id", "filename"], limit=10000)
    seen = {}
    for r in results:
        seen[r["doc_id"]] = r["filename"]
    return [{"doc_id": k, "filename": v} for k, v in seen.items()]
