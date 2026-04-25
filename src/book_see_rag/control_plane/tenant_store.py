from __future__ import annotations

import json
import re
import time
from threading import Lock
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from book_see_rag.config import get_settings

_STORE_LOCK = Lock()
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


class TenantRecord(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    slug: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    status: str = Field(default="active")  # active | disabled | deleted
    created_at: int = Field(default_factory=lambda: int(time.time()))
    plan: str = Field(default="standard")

    @property
    def domain(self) -> str:
        settings = get_settings()
        return f"{self.slug}.{settings.tenant_base_domain.strip()}"


def _data_dir() -> Path:
    settings = get_settings()
    base = Path(settings.control_plane_data_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _tenants_file() -> Path:
    return _data_dir() / "tenants.json"


def _read_all() -> list[dict[str, Any]]:
    path = _tenants_file()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_all(items: list[dict[str, Any]]) -> None:
    _tenants_file().write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _audit_file() -> Path:
    return _data_dir() / "tenant_audit.jsonl"


def append_audit_event(action: str, actor_user_id: str, tenant_id: str, detail: dict[str, Any] | None = None) -> None:
    event = {
        "ts": int(time.time()),
        "action": action,
        "actor_user_id": actor_user_id,
        "tenant_id": tenant_id,
        "detail": detail or {},
    }
    with _audit_file().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def list_audit_events(tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    path = _audit_file()
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if tenant_id and event.get("tenant_id") != tenant_id:
            continue
        events.append(event)
    return events[-max(limit, 1):]


def _validate_slug(slug: str) -> str:
    normalized = slug.strip().lower()
    if not _SLUG_RE.match(normalized):
        raise ValueError("slug 仅支持小写字母、数字和连字符，且不能以连字符开头或结尾")
    return normalized


def _emit_webhook(event: str, payload: dict[str, Any]) -> None:
    settings = get_settings()
    url = (settings.tenant_deploy_webhook_url or "").strip()
    if not url:
        return
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        append_audit_event("tenant.webhook.rejected", "system", payload.get("tenant_id", ""), {"url": url})
        return
    body = {"event": event, "payload": payload}
    try:
        response = httpx.post(url, json=body, timeout=10.0)
        if response.status_code >= 400:
            append_audit_event(
                "tenant.webhook.failed",
                "system",
                payload.get("tenant_id", ""),
                {"event": event, "status_code": response.status_code},
            )
    except Exception as exc:
        append_audit_event(
            "tenant.webhook.failed",
            "system",
            payload.get("tenant_id", ""),
            {"event": event, "error": str(exc)},
        )
        return


def list_tenants() -> list[TenantRecord]:
    return [TenantRecord(**item) for item in _read_all()]


def get_tenant(tenant_id: str) -> TenantRecord | None:
    for item in list_tenants():
        if item.tenant_id == tenant_id:
            return item
    return None


def create_tenant(record: TenantRecord) -> TenantRecord:
    record = record.model_copy(update={"slug": _validate_slug(record.slug)})
    with _STORE_LOCK:
        items = _read_all()
        if any(it.get("tenant_id") == record.tenant_id for it in items):
            raise ValueError("tenant_id 已存在")
        if any((it.get("slug") or "").lower() == record.slug.lower() for it in items):
            raise ValueError("slug 已存在")
        items.append(record.model_dump())
        _write_all(items)
    payload = record.model_dump()
    payload["domain"] = record.domain
    _emit_webhook("tenant.created", payload)
    return record


def disable_tenant(tenant_id: str) -> TenantRecord | None:
    rec: TenantRecord | None = None
    with _STORE_LOCK:
        items = _read_all()
        for it in items:
            if it.get("tenant_id") == tenant_id:
                it["status"] = "disabled"
                _write_all(items)
                rec = TenantRecord(**it)
                break
    if not rec:
        return None
    payload = rec.model_dump()
    payload["domain"] = rec.domain
    _emit_webhook("tenant.disabled", payload)
    return rec


def enable_tenant(tenant_id: str) -> TenantRecord | None:
    rec: TenantRecord | None = None
    with _STORE_LOCK:
        items = _read_all()
        for it in items:
            if it.get("tenant_id") == tenant_id:
                it["status"] = "active"
                _write_all(items)
                rec = TenantRecord(**it)
                break
    if not rec:
        return None
    payload = rec.model_dump()
    payload["domain"] = rec.domain
    _emit_webhook("tenant.enabled", payload)
    return rec


def delete_tenant(tenant_id: str) -> bool:
    removed: dict[str, Any] | None = None
    with _STORE_LOCK:
        items = _read_all()
        for i, it in enumerate(items):
            if it.get("tenant_id") == tenant_id:
                removed = items.pop(i)
                _write_all(items)
                break
    if not removed:
        return False
    rec = TenantRecord(**removed)
    payload = rec.model_dump()
    payload["domain"] = rec.domain
    _emit_webhook("tenant.deleted", payload)
    return True

