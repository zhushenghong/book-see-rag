import json
from typing import Any
import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, messages_from_dict, messages_to_dict
from book_see_rag.config import get_settings


def _get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


class RedisMessageHistory(BaseChatMessageHistory):
    """
    基于 Redis 的多用户对话记忆，按 session_id 隔离，TTL 自动过期。
    实现 LangChain BaseChatMessageHistory 接口，可直接用于 RunnableWithMessageHistory。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._redis = _get_redis()
        self._key = f"session:{session_id}"
        settings = get_settings()
        self._ttl = settings.session_ttl

    @property
    def messages(self) -> list[BaseMessage]:
        raw = self._redis.get(self._key)
        if not raw:
            return []
        return messages_from_dict(json.loads(raw))

    def add_message(self, message: BaseMessage) -> None:
        current = messages_to_dict(self.messages)
        current.append(messages_to_dict([message])[0])
        self._redis.setex(self._key, self._ttl, json.dumps(current, ensure_ascii=False))

    def clear(self) -> None:
        self._redis.delete(self._key)


def get_session_history(session_id: str) -> RedisMessageHistory:
    return RedisMessageHistory(session_id)


def _session_scope_key(session_id: str) -> str:
    return f"session_scope:{session_id}"


def get_recent_messages(session_id: str, limit: int | None = None) -> list[BaseMessage]:
    messages = RedisMessageHistory(session_id).messages
    if limit is None or limit <= 0:
        return messages
    return messages[-limit:]


def list_session_messages(session_id: str) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in RedisMessageHistory(session_id).messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        sources = msg.additional_kwargs.get("sources", []) if isinstance(msg, AIMessage) else []
        scope = msg.additional_kwargs.get("scope", {"doc_ids": [], "kb_ids": []}) if isinstance(msg, AIMessage) else {"doc_ids": [], "kb_ids": []}
        serialized.append(
            {
                "role": role,
                "content": msg.content,
                "sources": sources,
                "scope": scope,
            }
        )
    return serialized


def append_user_message(session_id: str, content: str) -> None:
    RedisMessageHistory(session_id).add_message(HumanMessage(content=content))


def append_ai_message(
    session_id: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
    scope: dict[str, list[str]] | None = None,
) -> None:
    RedisMessageHistory(session_id).add_message(
        AIMessage(content=content, additional_kwargs={"sources": sources or [], "scope": scope or {"doc_ids": [], "kb_ids": []}})
    )


def get_session_scope(session_id: str) -> dict[str, list[str]]:
    r = _get_redis()
    raw = r.get(_session_scope_key(session_id))
    if not raw:
        return {"doc_ids": [], "kb_ids": []}
    data = json.loads(raw)
    return {
        "doc_ids": data.get("doc_ids", []),
        "kb_ids": data.get("kb_ids", []),
    }


def set_session_scope(session_id: str, doc_ids: list[str], kb_ids: list[str]) -> dict[str, list[str]]:
    r = _get_redis()
    settings = get_settings()
    payload = {"doc_ids": doc_ids, "kb_ids": kb_ids}
    r.setex(_session_scope_key(session_id), settings.session_ttl, json.dumps(payload, ensure_ascii=False))
    return payload


def clear_session_scope(session_id: str) -> None:
    r = _get_redis()
    r.delete(_session_scope_key(session_id))


def delete_session(session_id: str) -> None:
    r = _get_redis()
    r.delete(f"session:{session_id}")
    r.delete(_session_scope_key(session_id))
