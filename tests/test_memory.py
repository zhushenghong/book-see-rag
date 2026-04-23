import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def mock_redis():
    store = {}
    r = MagicMock()
    r.get.side_effect = lambda k: store.get(k)
    r.setex.side_effect = lambda k, ttl, v: store.update({k: v})
    r.delete.side_effect = lambda k: store.pop(k, None)
    return r, store


def test_redis_memory_empty_session(mock_redis):
    r, _ = mock_redis
    with patch("book_see_rag.memory.redis_memory._get_redis", return_value=r):
        from book_see_rag.memory.redis_memory import RedisMessageHistory
        history = RedisMessageHistory("session-001")
        assert history.messages == []


def test_redis_memory_add_and_load(mock_redis):
    r, _ = mock_redis
    with patch("book_see_rag.memory.redis_memory._get_redis", return_value=r):
        from book_see_rag.memory.redis_memory import RedisMessageHistory
        history = RedisMessageHistory("session-002")
        history.add_message(HumanMessage(content="你好"))
        history.add_message(AIMessage(content="你好！有什么可以帮你？"))

        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].content == "你好"
        assert msgs[1].content == "你好！有什么可以帮你？"


def test_redis_memory_clear(mock_redis):
    r, store = mock_redis
    with patch("book_see_rag.memory.redis_memory._get_redis", return_value=r):
        from book_see_rag.memory.redis_memory import RedisMessageHistory
        history = RedisMessageHistory("session-003")
        history.add_message(HumanMessage(content="test"))
        history.clear()
        assert history.messages == []


def test_delete_session(mock_redis):
    r, store = mock_redis
    store["session:del-001"] = "data"
    with patch("book_see_rag.memory.redis_memory._get_redis", return_value=r):
        from book_see_rag.memory.redis_memory import delete_session
        delete_session("del-001")
        assert "session:del-001" not in store


def test_list_session_messages_includes_sources(mock_redis):
    r, _ = mock_redis
    with patch("book_see_rag.memory.redis_memory._get_redis", return_value=r):
        from book_see_rag.memory.redis_memory import append_ai_message, append_user_message, list_session_messages

        append_user_message("session-004", "第一问")
        append_ai_message("session-004", "第一答", [
            {"doc_id": "d1", "filename": "简历.pdf", "page": 2, "content": "片段A"},
            {"doc_id": "d1", "filename": "简历.pdf", "page": 3, "content": "片段B"},
        ], scope={"doc_ids": ["d1"], "kb_ids": ["kb_public"]})

        messages = list_session_messages("session-004")
        assert messages == [
            {"role": "user", "content": "第一问", "sources": [], "scope": {"doc_ids": [], "kb_ids": []}},
            {
                "role": "assistant",
                "content": "第一答",
                "sources": [
                    {"doc_id": "d1", "filename": "简历.pdf", "page": 2, "content": "片段A"},
                    {"doc_id": "d1", "filename": "简历.pdf", "page": 3, "content": "片段B"},
                ],
                "scope": {"doc_ids": ["d1"], "kb_ids": ["kb_public"]},
            },
        ]


def test_session_scope_roundtrip(mock_redis):
    r, _ = mock_redis
    with patch("book_see_rag.memory.redis_memory._get_redis", return_value=r):
        from book_see_rag.memory.redis_memory import get_session_scope, set_session_scope

        set_session_scope("session-005", ["d1", "d2"], ["kb_public"])
        scope = get_session_scope("session-005")
        assert scope == {"doc_ids": ["d1", "d2"], "kb_ids": ["kb_public"]}
