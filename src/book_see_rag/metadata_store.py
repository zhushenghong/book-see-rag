from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from book_see_rag.access_control import UserContext
from book_see_rag.config import get_settings


DEFAULT_KNOWLEDGE_BASES = [
    {
        "kb_id": "kb_public",
        "name": "公共知识库",
        "visibility": "public",
        "query_profile": None,
        "departments": [],
        "roles": [],
        "user_ids": [],
    },
    {
        "kb_id": "kb_rd",
        "name": "研发知识库",
        "visibility": "department",
        "query_profile": None,
        "departments": ["rd", "engineering", "tech"],
        "roles": [],
        "user_ids": [],
    },
    {
        "kb_id": "kb_hr",
        "name": "人事知识库",
        "visibility": "department",
        "query_profile": None,
        "departments": ["hr"],
        "roles": ["hr_admin"],
        "user_ids": [],
    },
]

def _backend() -> str:
    settings = get_settings()
    return (settings.metadata_backend or "json").strip().lower()


def _default_tenant_id() -> str:
    settings = get_settings()
    return (settings.default_tenant_id or "public").strip() or "public"


def _record_tenant_id(record: dict[str, Any]) -> str:
    return (record.get("tenant_id") or _default_tenant_id()).strip() or _default_tenant_id()


def _matches_tenant(user: UserContext, record: dict[str, Any]) -> bool:
    return _record_tenant_id(record) == user.tenant_id


def _default_knowledge_bases() -> list[dict[str, Any]]:
    tenant_id = _default_tenant_id()
    return [{**kb, "tenant_id": kb.get("tenant_id") or tenant_id} for kb in DEFAULT_KNOWLEDGE_BASES]


def _sql():
    try:
        import book_see_rag.metadata_sql as metadata_sql
    except ImportError as exc:
        raise RuntimeError("SQL metadata backend requires SQLAlchemy and a DB driver") from exc
    return metadata_sql


def _metadata_dir() -> Path:
    settings = get_settings()
    base = Path(settings.metadata_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _kb_file() -> Path:
    return _metadata_dir() / "knowledge_bases.json"


def _doc_file() -> Path:
    return _metadata_dir() / "documents.json"


def _ingest_task_file() -> Path:
    return _metadata_dir() / "ingest_tasks.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_seed_data() -> None:
    if _backend() == "sql":
        _sql().seed_knowledge_bases(_default_knowledge_bases())
        return
    kb_path = _kb_file()
    if not kb_path.exists():
        _write_json(kb_path, _default_knowledge_bases())
    doc_path = _doc_file()
    if not doc_path.exists():
        _write_json(doc_path, [])
    ingest_task_path = _ingest_task_file()
    if not ingest_task_path.exists():
        _write_json(ingest_task_path, [])


def list_knowledge_bases() -> list[dict[str, Any]]:
    _ensure_seed_data()
    if _backend() == "sql":
        return _sql().list_knowledge_bases()
    return _read_json(_kb_file(), [])


def list_knowledge_bases_for_tenant(tenant_id: str) -> list[dict[str, Any]]:
    normalized_tenant_id = (tenant_id or _default_tenant_id()).strip() or _default_tenant_id()
    if _backend() == "sql":
        _ensure_seed_data()
        return _sql().list_knowledge_bases(normalized_tenant_id)
    return [kb for kb in list_knowledge_bases() if _record_tenant_id(kb) == normalized_tenant_id]


def create_knowledge_base(
    kb_id: str,
    name: str,
    visibility: str = "public",
    query_profile: str | None = None,
    tenant_id: str | None = None,
    departments: list[str] | None = None,
    roles: list[str] | None = None,
    user_ids: list[str] | None = None,
) -> dict[str, Any]:
    _ensure_seed_data()
    normalized_kb_id = kb_id.strip()
    normalized_name = name.strip()
    normalized_visibility = visibility.strip().lower() or "public"
    if not normalized_kb_id:
        raise ValueError("kb_id 不能为空")
    if not normalized_name:
        raise ValueError("name 不能为空")
    if normalized_visibility not in {"public", "department", "private"}:
        raise ValueError("visibility 仅支持 public / department / private")

    record = {
        "kb_id": normalized_kb_id,
        "tenant_id": (tenant_id or _default_tenant_id()).strip() or _default_tenant_id(),
        "name": normalized_name,
        "visibility": normalized_visibility,
        "query_profile": (query_profile or "").strip() or None,
        "departments": sorted({item.strip() for item in (departments or []) if item and item.strip()}),
        "roles": sorted({item.strip() for item in (roles or []) if item and item.strip()}),
        "user_ids": sorted({item.strip() for item in (user_ids or []) if item and item.strip()}),
    }
    if _backend() == "sql":
        return _sql().create_knowledge_base(record)
    items = list_knowledge_bases()
    if any(kb["kb_id"] == normalized_kb_id and _record_tenant_id(kb) == record["tenant_id"] for kb in items):
        raise ValueError(f"知识库 {normalized_kb_id} 已存在")
    items.append(record)
    _write_json(_kb_file(), items)
    return record


def get_knowledge_base(kb_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
    if _backend() == "sql":
        _ensure_seed_data()
        return _sql().get_knowledge_base(kb_id, tenant_id)
    for kb in list_knowledge_bases():
        if kb["kb_id"] == kb_id and (tenant_id is None or _record_tenant_id(kb) == tenant_id):
            return kb
    return None


def user_can_access_kb(user: UserContext, kb: dict[str, Any]) -> bool:
    if not _matches_tenant(user, kb):
        return False
    visibility = kb.get("visibility", "public")
    if visibility == "public":
        return True
    if user.user_id in kb.get("user_ids", []):
        return True
    if user.role in kb.get("roles", []):
        return True
    if user.department in kb.get("departments", []):
        return True
    return False


def list_knowledge_bases_for_user(user: UserContext) -> list[dict[str, Any]]:
    return [kb for kb in list_knowledge_bases_for_tenant(user.tenant_id) if user_can_access_kb(user, kb)]


def register_document(doc_id: str, filename: str, kb_id: str, tenant_id: str | None = None) -> None:
    _ensure_seed_data()
    resolved_tenant_id = (tenant_id or _default_tenant_id()).strip() or _default_tenant_id()
    if _backend() == "sql":
        _sql().upsert_document(doc_id, filename, kb_id, resolved_tenant_id)
        return
    docs = [
        doc
        for doc in _read_json(_doc_file(), [])
        if not (doc["doc_id"] == doc_id and _record_tenant_id(doc) == resolved_tenant_id)
    ]
    docs.append({"doc_id": doc_id, "filename": filename, "kb_id": kb_id, "tenant_id": resolved_tenant_id})
    _write_json(_doc_file(), docs)


def delete_document(doc_id: str, tenant_id: str | None = None) -> None:
    _ensure_seed_data()
    if _backend() == "sql":
        _sql().delete_document(doc_id, tenant_id)
        return
    docs = [
        doc
        for doc in _read_json(_doc_file(), [])
        if not (doc["doc_id"] == doc_id and (tenant_id is None or _record_tenant_id(doc) == tenant_id))
    ]
    _write_json(_doc_file(), docs)


def list_documents() -> list[dict[str, Any]]:
    _ensure_seed_data()
    if _backend() == "sql":
        return _sql().list_documents()
    return _read_json(_doc_file(), [])


def list_documents_for_tenant(tenant_id: str) -> list[dict[str, Any]]:
    normalized_tenant_id = (tenant_id or _default_tenant_id()).strip() or _default_tenant_id()
    _ensure_seed_data()
    if _backend() == "sql":
        return _sql().list_documents(normalized_tenant_id)
    return [doc for doc in _read_json(_doc_file(), []) if _record_tenant_id(doc) == normalized_tenant_id]


def register_ingest_task(task_id: str, doc_id: str, tenant_id: str | None = None) -> None:
    _ensure_seed_data()
    resolved_tenant_id = (tenant_id or _default_tenant_id()).strip() or _default_tenant_id()
    if _backend() == "sql":
        _sql().upsert_ingest_task(task_id, doc_id, resolved_tenant_id)
        return
    tasks = [
        task
        for task in _read_json(_ingest_task_file(), [])
        if not (task["task_id"] == task_id and _record_tenant_id(task) == resolved_tenant_id)
    ]
    tasks.append({"task_id": task_id, "doc_id": doc_id, "tenant_id": resolved_tenant_id})
    _write_json(_ingest_task_file(), tasks)


def get_ingest_task(task_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
    _ensure_seed_data()
    if _backend() == "sql":
        return _sql().get_ingest_task(task_id, tenant_id)
    for task in _read_json(_ingest_task_file(), []):
        if task["task_id"] == task_id and (tenant_id is None or _record_tenant_id(task) == tenant_id):
            return task
    return None


def list_documents_for_user(user: UserContext) -> list[dict[str, Any]]:
    allowed_kbs = {kb["kb_id"] for kb in list_knowledge_bases_for_user(user)}
    return [doc for doc in list_documents_for_tenant(user.tenant_id) if doc["kb_id"] in allowed_kbs]


def resolve_allowed_doc_ids(
    user: UserContext,
    requested_doc_ids: list[str] | None = None,
    requested_kb_ids: list[str] | None = None,
) -> list[str]:
    allowed_kbs = {kb["kb_id"] for kb in list_knowledge_bases_for_user(user)}
    if requested_kb_ids:
        allowed_kbs &= set(requested_kb_ids)

    allowed_docs = {
        doc["doc_id"]
        for doc in list_documents_for_tenant(user.tenant_id)
        if doc["kb_id"] in allowed_kbs
    }
    if requested_doc_ids:
        return [doc_id for doc_id in requested_doc_ids if doc_id in allowed_docs]
    return sorted(allowed_docs)
