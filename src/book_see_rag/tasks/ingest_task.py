from celery import Celery
import logging
import time
from book_see_rag.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "book_see_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=86400,
)


@celery_app.task(bind=True, name="ingest_document")
def ingest_document(self, doc_id: str, file_path: str, filename: str) -> dict:
    """
    Celery 异步任务：加载 → 分块 → 向量化 → 写入 Milvus
    可通过 task_id 查询状态：pending / started / success / failure
    """
    try:
        task_started = time.perf_counter()
        logger.info(
            "Ingest task started task_id=%s doc_id=%s filename=%s path=%s",
            self.request.id,
            doc_id,
            filename,
            file_path,
        )
        self.update_state(state="PROGRESS", meta={"step": "loading"})

        from book_see_rag.ingestion.loader import load_document
        from book_see_rag.ingestion.splitter import split_documents
        from book_see_rag.vectorstore.milvus_store import insert_chunks

        load_started = time.perf_counter()
        documents = load_document(file_path)
        load_elapsed = time.perf_counter() - load_started
        logger.info(
            "Document loaded doc_id=%s filename=%s documents=%s elapsed=%.3fs",
            doc_id,
            filename,
            len(documents),
            load_elapsed,
        )
        self.update_state(state="PROGRESS", meta={"step": "splitting"})

        split_started = time.perf_counter()
        chunks = split_documents(documents)
        split_elapsed = time.perf_counter() - split_started
        logger.info(
            "Documents split doc_id=%s filename=%s chunks=%s elapsed=%.3fs",
            doc_id,
            filename,
            len(chunks),
            split_elapsed,
        )
        self.update_state(state="PROGRESS", meta={"step": "embedding", "total_chunks": len(chunks)})

        logger.info(
            "Starting embedding and insert doc_id=%s filename=%s chunks=%s",
            doc_id,
            filename,
            len(chunks),
        )
        embed_insert_started = time.perf_counter()
        count = insert_chunks(doc_id, filename, chunks)
        embed_insert_elapsed = time.perf_counter() - embed_insert_started
        total_elapsed = time.perf_counter() - task_started
        logger.info(
            "Ingest task finished task_id=%s doc_id=%s filename=%s inserted=%s load=%.3fs split=%.3fs embed_insert=%.3fs total=%.3fs",
            self.request.id,
            doc_id,
            filename,
            count,
            load_elapsed,
            split_elapsed,
            embed_insert_elapsed,
            total_elapsed,
        )
        return {
            "status": "done",
            "doc_id": doc_id,
            "chunks": count,
            "timings": {
                "load": round(load_elapsed, 3),
                "split": round(split_elapsed, 3),
                "embed_insert": round(embed_insert_elapsed, 3),
                "total": round(total_elapsed, 3),
            },
        }

    except Exception as exc:
        logger.exception(
            "Ingest task failed task_id=%s doc_id=%s filename=%s error=%s",
            self.request.id,
            doc_id,
            filename,
            exc,
        )
        raise
