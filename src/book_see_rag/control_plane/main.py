from __future__ import annotations

import time
import uuid
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from book_see_rag.access_control import UserContext, get_current_user
from book_see_rag.config import get_settings
from book_see_rag.control_plane.tenant_store import (
    TenantRecord,
    append_audit_event,
    create_tenant,
    delete_tenant,
    disable_tenant,
    enable_tenant,
    get_tenant,
    list_audit_events,
    list_tenants,
)


app = FastAPI(
    title="book-see-rag-control-plane",
    description="多租户控制面：租户创建/禁用/删除/查看/列表，并触发部署流水线",
    version="0.1.0",
)


class TenantCreateRequest(BaseModel):
    slug: str = Field(..., min_length=1, max_length=64, description="租户标识，用于子域名，如 acme -> acme.rag.example.com")
    name: str = Field(..., min_length=1, max_length=128)
    plan: str = Field(default="standard")


class TenantResponse(BaseModel):
    tenant_id: str
    slug: str
    name: str
    status: str
    created_at: int
    plan: str
    domain: str


class TenantAuditEvent(BaseModel):
    ts: int
    action: str
    actor_user_id: str
    tenant_id: str
    detail: dict


def _require_admin(user: UserContext) -> None:
    settings = get_settings()
    auth_mode = (settings.auth_mode or "headers").lower()
    if auth_mode == "headers" and not settings.allow_insecure_control_plane_headers:
        raise HTTPException(status_code=403, detail="控制面必须启用 JWT 鉴权，或显式允许本地 Header 模式")
    if user.role != "platform_admin":
        raise HTTPException(status_code=403, detail="仅平台管理员可操作租户")


def _to_response(record: TenantRecord) -> TenantResponse:
    payload = record.model_dump()
    payload["domain"] = record.domain
    return TenantResponse(**payload)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tenants", response_model=list[TenantResponse])
async def tenants_list(user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    return [_to_response(t) for t in list_tenants()]


@app.get("/tenants/{tenant_id}", response_model=TenantResponse)
async def tenants_get(tenant_id: str, user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    item = get_tenant(tenant_id)
    if not item:
        raise HTTPException(status_code=404, detail="租户不存在")
    return _to_response(item)


@app.post("/tenants", response_model=TenantResponse)
async def tenants_create(payload: TenantCreateRequest, user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    tenant_id = str(uuid.uuid4())
    now = int(time.time())
    record = TenantRecord(
        tenant_id=tenant_id,
        slug=payload.slug.strip().lower(),
        name=payload.name.strip(),
        status="active",
        created_at=now,
        plan=payload.plan.strip() or "standard",
    )
    try:
        created = create_tenant(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_audit_event("tenant.created", user.user_id, created.tenant_id, {"slug": created.slug, "domain": created.domain})
    return _to_response(created)


@app.post("/tenants/{tenant_id}/disable", response_model=TenantResponse)
async def tenants_disable(tenant_id: str, user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    item = disable_tenant(tenant_id)
    if not item:
        raise HTTPException(status_code=404, detail="租户不存在")
    append_audit_event("tenant.disabled", user.user_id, item.tenant_id, {"slug": item.slug, "domain": item.domain})
    return _to_response(item)


@app.post("/tenants/{tenant_id}/enable", response_model=TenantResponse)
async def tenants_enable(tenant_id: str, user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    item = enable_tenant(tenant_id)
    if not item:
        raise HTTPException(status_code=404, detail="租户不存在")
    append_audit_event("tenant.enabled", user.user_id, item.tenant_id, {"slug": item.slug, "domain": item.domain})
    return _to_response(item)


@app.delete("/tenants/{tenant_id}", response_model=dict)
async def tenants_delete(tenant_id: str, user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    ok = delete_tenant(tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail="租户不存在")
    append_audit_event("tenant.deleted", user.user_id, tenant_id)
    return {"deleted": True, "tenant_id": tenant_id}


@app.get("/tenants/{tenant_id}/audit", response_model=list[TenantAuditEvent])
async def tenants_audit(tenant_id: str, limit: int = 100, user: UserContext = Depends(get_current_user)):
    _require_admin(user)
    return [TenantAuditEvent(**event) for event in list_audit_events(tenant_id=tenant_id, limit=limit)]

