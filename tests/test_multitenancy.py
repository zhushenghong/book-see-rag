import asyncio
import json

from fastapi.testclient import TestClient
from fastapi import HTTPException
from starlette.requests import Request


def test_build_user_context_includes_tenant():
    from book_see_rag.access_control import build_user_context

    user = build_user_context(tenant_id="tenant-acme", x_user_id="alice", x_role="employee", x_department="rd")

    assert user.tenant_id == "tenant-acme"
    assert user.user_id == "alice"
    assert user.role == "employee"
    assert user.department == "rd"


def test_header_mode_accepts_tenant_header(tmp_path, monkeypatch):
    from book_see_rag.access_control import get_current_user
    from book_see_rag.config import get_settings

    monkeypatch.setenv("CONTROL_PLANE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (b"x-tenant-id", b"tenant-acme"),
        ],
    }
    request = Request(scope)
    user = asyncio.run(
        get_current_user(
            request=request,
            authorization=None,
            x_tenant_id=None,
            x_user_id="alice",
            x_role=None,
            x_department=None,
        )
    )

    assert user.tenant_id == "tenant-acme"
    assert user.user_id == "alice"
    get_settings.cache_clear()


def test_control_plane_create_disable_delete_tenant(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings

    monkeypatch.setenv("CONTROL_PLANE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TENANT_BASE_DOMAIN", "rag.example.test")
    monkeypatch.setenv("ALLOW_INSECURE_CONTROL_PLANE_HEADERS", "true")
    get_settings.cache_clear()

    from book_see_rag.control_plane.main import app

    client = TestClient(app)
    headers = {"X-Role": "platform_admin", "X-User-Id": "ops"}

    created = client.post("/tenants", json={"slug": "acme", "name": "Acme"}, headers=headers)
    assert created.status_code == 200
    body = created.json()
    assert body["slug"] == "acme"
    assert body["domain"] == "acme.rag.example.test"

    tenant_id = body["tenant_id"]
    disabled = client.post(f"/tenants/{tenant_id}/disable", headers=headers)
    assert disabled.status_code == 200
    assert disabled.json()["status"] == "disabled"

    enabled = client.post(f"/tenants/{tenant_id}/enable", headers=headers)
    assert enabled.status_code == 200
    assert enabled.json()["status"] == "active"

    audit = client.get(f"/tenants/{tenant_id}/audit", headers=headers)
    assert audit.status_code == 200
    assert [event["action"] for event in audit.json()] == ["tenant.created", "tenant.disabled", "tenant.enabled"]

    deleted = client.delete(f"/tenants/{tenant_id}", headers=headers)
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "tenant_id": tenant_id}

    get_settings.cache_clear()


def test_control_plane_rejects_invalid_slug(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings

    monkeypatch.setenv("CONTROL_PLANE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOW_INSECURE_CONTROL_PLANE_HEADERS", "true")
    get_settings.cache_clear()

    from book_see_rag.control_plane.main import app

    client = TestClient(app)
    r = client.post(
        "/tenants",
        json={"slug": "-bad-slug", "name": "Bad"},
        headers={"X-Role": "platform_admin", "X-User-Id": "ops"},
    )

    assert r.status_code == 400
    assert "slug" in r.json()["detail"]
    get_settings.cache_clear()


def test_control_plane_rejects_non_platform_admin(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings

    monkeypatch.setenv("CONTROL_PLANE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOW_INSECURE_CONTROL_PLANE_HEADERS", "true")
    get_settings.cache_clear()

    from book_see_rag.control_plane.main import app

    client = TestClient(app)
    r = client.post(
        "/tenants",
        json={"slug": "hr", "name": "HR"},
        headers={"X-Role": "hr_admin", "X-User-Id": "hr"},
    )

    assert r.status_code == 403
    get_settings.cache_clear()


def test_control_plane_rejects_header_mode_unless_explicitly_allowed(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings

    monkeypatch.setenv("CONTROL_PLANE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ALLOW_INSECURE_CONTROL_PLANE_HEADERS", raising=False)
    get_settings.cache_clear()

    from book_see_rag.control_plane.main import app

    client = TestClient(app)
    r = client.get("/tenants", headers={"X-Role": "platform_admin"})

    assert r.status_code == 403
    assert "JWT" in r.json()["detail"]
    get_settings.cache_clear()


def test_disabled_tenant_is_rejected_by_user_context(tmp_path, monkeypatch):
    from book_see_rag.access_control import get_current_user
    from book_see_rag.config import get_settings
    from book_see_rag.control_plane.tenant_store import TenantRecord, create_tenant, disable_tenant

    monkeypatch.setenv("CONTROL_PLANE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    create_tenant(TenantRecord(tenant_id="tenant-disabled", slug="disabled", name="Disabled Tenant"))
    disable_tenant("tenant-disabled")

    request = Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"x-tenant-id", b"tenant-disabled")]})
    try:
        asyncio.run(
            get_current_user(
                request=request,
                authorization=None,
                x_tenant_id=None,
                x_user_id="alice",
                x_role=None,
                x_department=None,
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("disabled tenant should be rejected")
    get_settings.cache_clear()


def test_json_metadata_filters_by_tenant(tmp_path, monkeypatch):
    from book_see_rag.access_control import build_user_context
    from book_see_rag.config import get_settings
    from book_see_rag.metadata_store import list_documents_for_user, register_document

    monkeypatch.setenv("METADATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEFAULT_TENANT_ID", "tenant-a")
    get_settings.cache_clear()

    user_a = build_user_context(tenant_id="tenant-a")
    user_b = build_user_context(tenant_id="tenant-b")

    register_document("doc-a", "a.txt", "kb_public", tenant_id="tenant-a")

    assert [doc["doc_id"] for doc in list_documents_for_user(user_a)] == ["doc-a"]
    assert list_documents_for_user(user_b) == []
    get_settings.cache_clear()


def test_json_metadata_allows_same_kb_id_per_tenant(tmp_path, monkeypatch):
    from book_see_rag.access_control import build_user_context
    from book_see_rag.config import get_settings
    from book_see_rag.metadata_store import create_knowledge_base, get_knowledge_base, list_knowledge_bases_for_user

    monkeypatch.setenv("METADATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEFAULT_TENANT_ID", "tenant-a")
    get_settings.cache_clear()

    create_knowledge_base("kb_shared", "A Shared", tenant_id="tenant-a")
    create_knowledge_base("kb_shared", "B Shared", tenant_id="tenant-b")

    user_a = build_user_context(tenant_id="tenant-a")
    user_b = build_user_context(tenant_id="tenant-b")

    assert get_knowledge_base("kb_shared", tenant_id="tenant-a")["name"] == "A Shared"
    assert get_knowledge_base("kb_shared", tenant_id="tenant-b")["name"] == "B Shared"
    assert [kb["name"] for kb in list_knowledge_bases_for_user(user_a) if kb["kb_id"] == "kb_shared"] == ["A Shared"]
    assert [kb["name"] for kb in list_knowledge_bases_for_user(user_b) if kb["kb_id"] == "kb_shared"] == ["B Shared"]
    get_settings.cache_clear()


def test_delete_document_is_tenant_scoped(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings
    from book_see_rag.metadata_store import delete_document, list_documents, register_document

    monkeypatch.setenv("METADATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEFAULT_TENANT_ID", "tenant-a")
    get_settings.cache_clear()

    register_document("doc-1", "a.txt", "kb_public", tenant_id="tenant-a")
    register_document("doc-1", "b.txt", "kb_public", tenant_id="tenant-b")
    delete_document("doc-1", tenant_id="tenant-a")

    assert list_documents() == [{"doc_id": "doc-1", "filename": "b.txt", "kb_id": "kb_public", "tenant_id": "tenant-b"}]
    get_settings.cache_clear()


def test_sql_metadata_is_tenant_scoped(tmp_path, monkeypatch):
    from book_see_rag.access_control import build_user_context
    from book_see_rag.config import get_settings
    from book_see_rag.metadata_sql import get_engine
    from book_see_rag.metadata_store import (
        create_knowledge_base,
        get_knowledge_base,
        list_documents_for_user,
        register_document,
    )

    monkeypatch.setenv("METADATA_BACKEND", "sql")
    monkeypatch.setenv("METADATA_DB_URL", f"sqlite:///{tmp_path / 'metadata.db'}")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "tenant-a")
    get_settings.cache_clear()
    get_engine.cache_clear()

    create_knowledge_base("kb_shared", "A Shared", tenant_id="tenant-a")
    create_knowledge_base("kb_shared", "B Shared", tenant_id="tenant-b")
    register_document("doc-a", "a.txt", "kb_shared", tenant_id="tenant-a")
    register_document("doc-b", "b.txt", "kb_shared", tenant_id="tenant-b")

    assert get_knowledge_base("kb_shared", tenant_id="tenant-a")["name"] == "A Shared"
    assert get_knowledge_base("kb_shared", tenant_id="tenant-b")["name"] == "B Shared"
    assert [doc["doc_id"] for doc in list_documents_for_user(build_user_context(tenant_id="tenant-a"))] == ["doc-a"]
    assert [doc["doc_id"] for doc in list_documents_for_user(build_user_context(tenant_id="tenant-b"))] == ["doc-b"]

    get_engine.cache_clear()
    get_settings.cache_clear()


def test_ingest_task_mapping_is_tenant_scoped(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings
    from book_see_rag.metadata_store import get_ingest_task, register_ingest_task

    monkeypatch.setenv("METADATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEFAULT_TENANT_ID", "tenant-a")
    get_settings.cache_clear()

    register_ingest_task("task-1", "doc-a", tenant_id="tenant-a")

    assert get_ingest_task("task-1", tenant_id="tenant-a") == {
        "task_id": "task-1",
        "doc_id": "doc-a",
        "tenant_id": "tenant-a",
    }
    assert get_ingest_task("task-1", tenant_id="tenant-b") is None
    get_settings.cache_clear()


def test_metadata_migration_keeps_same_kb_id_across_tenants(tmp_path, monkeypatch):
    from book_see_rag.config import get_settings
    from book_see_rag.metadata_sql import get_engine, list_knowledge_bases
    from scripts.migrate_metadata_json_to_db import main

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "knowledge_bases.json").write_text(
        json.dumps(
            [
                {"tenant_id": "tenant-a", "kb_id": "kb_shared", "name": "A", "visibility": "public"},
                {"tenant_id": "tenant-b", "kb_id": "kb_shared", "name": "B", "visibility": "public"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "documents.json").write_text("[]", encoding="utf-8")

    monkeypatch.setenv("METADATA_BACKEND", "sql")
    monkeypatch.setenv("METADATA_DB_URL", f"sqlite:///{tmp_path / 'metadata.db'}")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "tenant-a")
    monkeypatch.setattr("sys.argv", ["migrate", "--metadata-dir", str(metadata_dir)])
    get_settings.cache_clear()
    get_engine.cache_clear()

    assert main() == 0

    shared = sorted(
        (kb["tenant_id"], kb["name"])
        for kb in list_knowledge_bases()
        if kb["kb_id"] == "kb_shared"
    )
    assert shared == [("tenant-a", "A"), ("tenant-b", "B")]

    get_engine.cache_clear()
    get_settings.cache_clear()

