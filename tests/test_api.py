import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    from book_see_rag.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_unsupported_format(client):
    r = client.post(
        "/api/ingest",
        files={"file": ("test.xlsx", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "不支持的文件格式" in r.json()["detail"]


def test_ingest_valid_txt(client, tmp_path):
    task_mock = MagicMock()
    task_mock.id = "task-abc"

    with patch("book_see_rag.api.routes.ingest.ingest_document") as mock_task, \
         patch("book_see_rag.api.routes.ingest.register_document") as mock_register:
        mock_task.delay.return_value = task_mock
        r = client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"Hello World content", "text/plain")},
            data={"kb_id": "kb_public"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == "task-abc"
    assert data["filename"] == "test.txt"
    assert data["kb_id"] == "kb_public"
    mock_register.assert_called_once()


def test_query_qa(client):
    with patch("book_see_rag.api.routes.query.answer") as mock_answer, \
         patch("book_see_rag.api.routes.query.resolve_allowed_doc_ids", return_value=["d1"]) as mock_scope:
        mock_answer.return_value = {"answer": "测试答案", "sources": ["chunk1"]}
        r = client.post("/api/query", json={"question": "什么是RAG？", "mode": "qa"})
    assert r.status_code == 200
    assert r.json()["result"] == "测试答案"
    mock_scope.assert_called_once()


def test_query_invalid_mode(client):
    r = client.post("/api/query", json={"question": "test", "mode": "invalid"})
    assert r.status_code == 422


def test_chat_endpoint(client):
    with patch("book_see_rag.api.routes.chat.chat") as mock_chat, \
         patch("book_see_rag.api.routes.chat.resolve_allowed_doc_ids", return_value=["d1"]) as mock_scope, \
         patch("book_see_rag.api.routes.chat.set_session_scope") as mock_set_scope:
        mock_set_scope.return_value = {"doc_ids": ["d1"], "kb_ids": []}
        mock_chat.return_value = {"answer": "你好！", "sources": [], "scope": {"doc_ids": ["d1"], "kb_ids": []}}
        r = client.post("/api/chat", json={"session_id": "s1", "message": "你好", "doc_ids": ["d1"]})
    assert r.status_code == 200
    assert r.json()["answer"] == "你好！"
    assert r.json()["session_id"] == "s1"
    assert r.json()["scope"] == {"doc_ids": ["d1"], "kb_ids": []}
    mock_scope.assert_called_once()
    mock_set_scope.assert_called_once_with("s1", ["d1"], [])


def test_chat_endpoint_uses_stored_scope_when_request_omits_filters(client):
    with patch("book_see_rag.api.routes.chat.chat") as mock_chat, \
         patch("book_see_rag.api.routes.chat.get_session_scope", return_value={"doc_ids": ["d2"], "kb_ids": ["kb_rd"]}) as mock_get_scope, \
         patch("book_see_rag.api.routes.chat.resolve_allowed_doc_ids", return_value=["d2"]) as mock_scope:
        mock_chat.return_value = {"answer": "你好！", "sources": [], "scope": {"doc_ids": ["d2"], "kb_ids": ["kb_rd"]}}
        r = client.post("/api/chat", json={"session_id": "s1", "message": "继续说"})
    assert r.status_code == 200
    assert r.json()["scope"] == {"doc_ids": ["d2"], "kb_ids": ["kb_rd"]}
    mock_get_scope.assert_called_once_with("s1")
    assert mock_scope.call_args.args[1:] == (["d2"], ["kb_rd"])


def test_get_session_scope(client):
    scope = {"doc_ids": ["d1"], "kb_ids": ["kb_public"]}
    with patch("book_see_rag.api.routes.chat.get_session_scope", return_value=scope):
        r = client.get("/api/chat/sessions/session-001/scope")
    assert r.status_code == 200
    assert r.json() == scope


def test_clear_session(client):
    with patch("book_see_rag.api.routes.chat.delete_session") as mock_del:
        r = client.delete("/api/chat/sessions/session-001")
    assert r.status_code == 200
    mock_del.assert_called_once_with("session-001")


def test_get_session_messages(client):
    messages = [
        {"role": "user", "content": "你好", "sources": [], "scope": {"doc_ids": [], "kb_ids": []}},
        {
            "role": "assistant",
            "content": "你好！",
            "sources": [{"doc_id": "d1", "filename": "简历.pdf", "page": 1, "content": "chunk1"}],
            "scope": {"doc_ids": ["d1"], "kb_ids": ["kb_public"]},
        },
    ]
    with patch("book_see_rag.api.routes.chat.list_session_messages", return_value=messages) as mock_list:
        r = client.get("/api/chat/sessions/session-001")
    assert r.status_code == 200
    assert r.json() == messages
    mock_list.assert_called_once_with("session-001")


def test_list_documents(client):
    with patch("book_see_rag.api.routes.documents.list_documents_for_user") as mock_list:
        mock_list.return_value = [{"doc_id": "d1", "filename": "test.pdf", "kb_id": "kb_public"}]
        r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json()[0]["filename"] == "test.pdf"


def test_list_knowledge_bases(client):
    items = [{"kb_id": "kb_public", "name": "公共知识库", "visibility": "public"}]
    with patch("book_see_rag.api.routes.knowledge_bases.list_knowledge_bases_for_user", return_value=items):
        r = client.get("/api/knowledge-bases")
    assert r.status_code == 200
    assert r.json() == items


def test_create_knowledge_base_requires_admin(client):
    r = client.post(
        "/api/knowledge-bases",
        json={"kb_id": "kb_finance", "name": "财务知识库", "visibility": "department"},
        headers={"X-Role": "employee"},
    )
    assert r.status_code == 403
    assert "仅管理员可创建知识库" in r.json()["detail"]


def test_create_knowledge_base_success(client):
    item = {"kb_id": "kb_finance", "name": "财务知识库", "visibility": "department"}
    with patch("book_see_rag.api.routes.knowledge_bases.create_knowledge_base", return_value=item) as mock_create:
        r = client.post(
            "/api/knowledge-bases",
            json={
                "kb_id": "kb_finance",
                "name": "财务知识库",
                "visibility": "department",
                "departments": ["finance"],
                "roles": ["hr_admin"],
                "user_ids": ["alice"],
            },
            headers={"X-Role": "hr_admin"},
        )
    assert r.status_code == 200
    assert r.json() == item
    mock_create.assert_called_once_with(
        kb_id="kb_finance",
        name="财务知识库",
        visibility="department",
        departments=["finance"],
        roles=["hr_admin"],
        user_ids=["alice"],
    )


def test_create_knowledge_base_duplicate_returns_400(client):
    with patch("book_see_rag.api.routes.knowledge_bases.create_knowledge_base", side_effect=ValueError("知识库 kb_finance 已存在")):
        r = client.post(
            "/api/knowledge-bases",
            json={"kb_id": "kb_finance", "name": "财务知识库"},
            headers={"X-Role": "hr_admin"},
        )
    assert r.status_code == 400
    assert "已存在" in r.json()["detail"]
