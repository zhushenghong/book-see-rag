from __future__ import annotations

import logging
from pathlib import Path

from book_see_rag.config import get_settings
from book_see_rag.ingestion.loader import load_document
from book_see_rag.ingestion.splitter import split_documents
from book_see_rag.metadata_store import list_documents
from book_see_rag.vectorstore.milvus_store import SearchHit

logger = logging.getLogger("uvicorn.error")


def load_hits_from_uploads(doc_ids: list[str], limit: int) -> list[SearchHit]:
    if not doc_ids:
        return []

    settings = get_settings()
    doc_map = {item["doc_id"]: item for item in list_documents()}
    hits: list[SearchHit] = []
    for doc_id in doc_ids:
        meta = doc_map.get(doc_id)
        if not meta:
            continue
        suffix = Path(meta["filename"]).suffix
        path = Path(settings.upload_dir) / f"{doc_id}{suffix}"
        if not path.exists():
            logger.warning("Scoped upload fallback skipped missing file path=%s", path)
            continue
        try:
            chunks = split_documents(load_document(str(path)))
        except Exception:
            logger.exception("Scoped upload fallback failed path=%s", path)
            continue
        for chunk in chunks:
            hits.append(
                {
                    "doc_id": doc_id,
                    "filename": meta["filename"],
                    "page": int(chunk.metadata.get("page") or 0),
                    "content": chunk.page_content,
                    "score": 0.0,
                }
            )
            if len(hits) >= limit:
                logger.info("Scoped upload fallback loaded hits=%s", len(hits))
                return hits

    logger.info("Scoped upload fallback loaded hits=%s", len(hits))
    return hits
