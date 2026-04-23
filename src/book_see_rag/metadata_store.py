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
        "departments": [],
        "roles": [],
        "user_ids": [],
    },
    {
        "kb_id": "kb_rd",
        "name": "研发知识库",
        "visibility": "department",
        "departments": ["rd", "engineering", "tech"],
        "roles": [],
        "user_ids": [],
    },
    {
        "kb_id": "kb_hr",
        "name": "人事知识库",
        "visibility": "department",
        "departments": ["hr"],
        "roles": ["hr_admin"],
        "user_ids": [],
    },
]


def _metadata_dir() -> Path:
    settings = get_settings()
    base = Path(settings.metadata_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _kb_file() -> Path:
    return _metadata_dir() / "knowledge_bases.json"


def _doc_file() -> Path:
    return _metadata_dir() / "documents.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_seed_data() -> None:
    kb_path = _kb_file()
    if not kb_path.exists():
        _write_json(kb_path, DEFAULT_KNOWLEDGE_BASES)
    doc_path = _doc_file()
    if not doc_path.exists():
        _write_json(doc_path, [])


def list_knowledge_bases() -> list[dict[str, Any]]:
    _ensure_seed_data()
    return _read_json(_kb_file(), [])


def create_knowledge_base(
    kb_id: str,
    name: str,
    visibility: str = "public",
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

    items = list_knowledge_bases()
    if any(kb["kb_id"] == normalized_kb_id for kb in items):
        raise ValueError(f"知识库 {normalized_kb_id} 已存在")

    record = {
        "kb_id": normalized_kb_id,
        "name": normalized_name,
        "visibility": normalized_visibility,
        "departments": sorted({item.strip() for item in (departments or []) if item and item.strip()}),
        "roles": sorted({item.strip() for item in (roles or []) if item and item.strip()}),
        "user_ids": sorted({item.strip() for item in (user_ids or []) if item and item.strip()}),
    }
    items.append(record)
    _write_json(_kb_file(), items)
    return record


def get_knowledge_base(kb_id: str) -> dict[str, Any] | None:
    for kb in list_knowledge_bases():
        if kb["kb_id"] == kb_id:
            return kb
    return None


def user_can_access_kb(user: UserContext, kb: dict[str, Any]) -> bool:
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
    return [kb for kb in list_knowledge_bases() if user_can_access_kb(user, kb)]


def register_document(doc_id: str, filename: str, kb_id: str) -> None:
    _ensure_seed_data()
    docs = [doc for doc in _read_json(_doc_file(), []) if doc["doc_id"] != doc_id]
    docs.append({"doc_id": doc_id, "filename": filename, "kb_id": kb_id})
    _write_json(_doc_file(), docs)


def delete_document(doc_id: str) -> None:
    _ensure_seed_data()
    docs = [doc for doc in _read_json(_doc_file(), []) if doc["doc_id"] != doc_id]
    _write_json(_doc_file(), docs)


def list_documents() -> list[dict[str, Any]]:
    _ensure_seed_data()
    return _read_json(_doc_file(), [])


def list_documents_for_user(user: UserContext) -> list[dict[str, Any]]:
    allowed_kbs = {kb["kb_id"] for kb in list_knowledge_bases_for_user(user)}
    return [doc for doc in list_documents() if doc["kb_id"] in allowed_kbs]


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
        for doc in list_documents()
        if doc["kb_id"] in allowed_kbs
    }
    if requested_doc_ids:
        return [doc_id for doc_id in requested_doc_ids if doc_id in allowed_docs]
    return sorted(allowed_docs)
