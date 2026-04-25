from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException, Request

from book_see_rag.config import get_settings


@dataclass(frozen=True)
class UserContext:
    tenant_id: str
    user_id: str
    role: str
    department: str


def build_user_context(
    tenant_id: str | None = None,
    x_user_id: str | None = None,
    x_role: str | None = None,
    x_department: str | None = None,
) -> UserContext:
    return UserContext(
        tenant_id=(tenant_id or "public").strip() or "public",
        user_id=(x_user_id or "guest").strip() or "guest",
        role=(x_role or "employee").strip() or "employee",
        department=(x_department or "general").strip() or "general",
    )


def _decode_jwt(token: str) -> dict[str, Any]:
    try:
        import jwt
        from jwt import PyJWTError
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="PyJWT 未安装，无法启用 JWT 鉴权") from exc

    settings = get_settings()
    if settings.jwt_public_key_pem.strip():
        algorithms = ["RS256"]
        key = settings.jwt_public_key_pem
    else:
        algorithms = ["HS256"]
        key = settings.jwt_secret
        if not key:
            raise HTTPException(status_code=500, detail="JWT_SECRET 未配置")

    options = {
        "verify_signature": True,
        "verify_aud": bool(settings.jwt_audience),
        "verify_iss": bool(settings.jwt_issuer),
    }
    try:
        payload = jwt.decode(
            token,
            key=key,
            algorithms=algorithms,
            audience=(settings.jwt_audience or None),
            issuer=(settings.jwt_issuer or None),
            options=options,
        )
    except PyJWTError as exc:
        raise HTTPException(status_code=401, detail="无效的登录凭证") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="无效的登录凭证")
    return payload


def _ensure_tenant_active(tenant_id: str) -> None:
    settings = get_settings()
    if not settings.enforce_tenant_status:
        return
    try:
        from book_see_rag.control_plane.tenant_store import get_tenant
    except ModuleNotFoundError as exc:
        if exc.name == "book_see_rag.control_plane.tenant_store":
            return
        raise

    tenant = get_tenant(tenant_id)
    if not tenant:
        if settings.require_tenant_registered:
            raise HTTPException(status_code=403, detail="租户未注册")
        return
    if tenant.status != "active":
        raise HTTPException(status_code=403, detail="租户已被禁用")


async def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_user_id: str | None = Header(default=None),
    x_role: str | None = Header(default=None),
    x_department: str | None = Header(default=None),
) -> UserContext:
    settings = get_settings()
    auth_mode = (settings.auth_mode or "headers").lower()

    if auth_mode == "jwt":
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="缺少 Authorization: Bearer <token>")
        token = authorization.split(" ", 1)[1].strip()
        payload = _decode_jwt(token)

        tenant_claim = (settings.tenant_claim or "tenant_id").strip() or "tenant_id"
        tenant_id = str(payload.get(tenant_claim) or payload.get("tid") or payload.get("tenant") or "").strip() or "public"
        user_id = str(payload.get("sub") or payload.get("user_id") or payload.get("uid") or "").strip() or "guest"
        role = str(payload.get("role") or "").strip() or "employee"
        department = str(payload.get("department") or payload.get("dept") or "").strip() or "general"
        _ensure_tenant_active(tenant_id)
        return build_user_context(tenant_id=tenant_id, x_user_id=user_id, x_role=role, x_department=department)

    # headers mode (dev / behind trusted gateway)
    tenant_header = (settings.tenant_header or "x-tenant-id").lower()
    tenant_id = (request.headers.get(tenant_header) or x_tenant_id or "public").strip() or "public"
    _ensure_tenant_active(tenant_id)
    return build_user_context(tenant_id=tenant_id, x_user_id=x_user_id, x_role=x_role, x_department=x_department)
