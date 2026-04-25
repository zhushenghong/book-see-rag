from __future__ import annotations

import argparse
import json
from pathlib import Path

from book_see_rag.config import get_settings
from book_see_rag.metadata_sql import ensure_schema, get_engine
from book_see_rag.metadata_store import DEFAULT_KNOWLEDGE_BASES
from sqlalchemy import text


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate metadata JSON (knowledge_bases/documents) into SQL metadata store.")
    parser.add_argument("--metadata-dir", default=None, help="Path to metadata dir (default: settings.metadata_dir)")
    args = parser.parse_args()

    settings = get_settings()
    metadata_dir = Path(args.metadata_dir or settings.metadata_dir)
    kb_file = metadata_dir / "knowledge_bases.json"
    doc_file = metadata_dir / "documents.json"
    ingest_task_file = metadata_dir / "ingest_tasks.json"

    ensure_schema()
    engine = get_engine()
    tenant_id = (settings.default_tenant_id or "public").strip() or "public"

    # Seed defaults first (idempotent)
    from book_see_rag.metadata_sql import seed_knowledge_bases, upsert_document, create_knowledge_base, list_knowledge_bases

    seed_knowledge_bases([{**kb, "tenant_id": kb.get("tenant_id") or tenant_id} for kb in DEFAULT_KNOWLEDGE_BASES])

    if kb_file.exists():
        items = json.loads(kb_file.read_text(encoding="utf-8"))
        existing_ids = {(kb.get("tenant_id") or tenant_id, kb["kb_id"]) for kb in list_knowledge_bases()}
        for kb in items:
            kb["tenant_id"] = kb.get("tenant_id") or tenant_id
            if (kb["tenant_id"], kb["kb_id"]) in existing_ids:
                continue
            create_knowledge_base(kb)

    if doc_file.exists():
        docs = json.loads(doc_file.read_text(encoding="utf-8"))
        for doc in docs:
            doc_id = str(doc["doc_id"])
            filename = str(doc.get("filename") or "")
            kb_id = str(doc.get("kb_id") or "kb_public")
            upsert_document(doc_id, filename, kb_id, str(doc.get("tenant_id") or tenant_id))

    if ingest_task_file.exists():
        tasks = json.loads(ingest_task_file.read_text(encoding="utf-8"))
        from book_see_rag.metadata_sql import upsert_ingest_task

        for task in tasks:
            upsert_ingest_task(
                str(task["task_id"]),
                str(task["doc_id"]),
                str(task.get("tenant_id") or tenant_id),
            )

    # quick sanity: count rows
    with engine.connect() as conn:
        kb_count = conn.execute(text("select count(*) from knowledge_bases")).scalar_one()
        doc_count = conn.execute(text("select count(*) from documents")).scalar_one()
        task_count = conn.execute(text("select count(*) from ingest_tasks")).scalar_one()
    print(f"OK: knowledge_bases={kb_count} documents={doc_count} ingest_tasks={task_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

