from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from book_see_rag.access_control import UserContext, get_current_user
from book_see_rag.chains.chat_chain import chat
from book_see_rag.memory.redis_memory import delete_session, get_session_scope, list_session_messages, set_session_scope
from book_see_rag.metadata_store import resolve_allowed_doc_ids

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    doc_ids: list[str] | None = None
    kb_ids: list[str] | None = None


class Citation(BaseModel):
    doc_id: str
    filename: str
    page: int
    content: str


class SessionScope(BaseModel):
    doc_ids: list[str] = []
    kb_ids: list[str] = []


class SessionScopeMessage(BaseModel):
    doc_ids: list[str] = []
    kb_ids: list[str] = []


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Citation]
    scope: SessionScope


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: list[Citation] = []
    scope: SessionScopeMessage = SessionScopeMessage()


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest, user: UserContext = Depends(get_current_user)):
    try:
        incoming_scope = req.doc_ids is not None or req.kb_ids is not None
        if incoming_scope:
            effective_doc_ids = resolve_allowed_doc_ids(user, req.doc_ids, req.kb_ids)
            effective_scope = set_session_scope(req.session_id, effective_doc_ids, req.kb_ids or [])
        else:
            stored_scope = get_session_scope(req.session_id)
            effective_doc_ids = resolve_allowed_doc_ids(user, stored_scope["doc_ids"], stored_scope["kb_ids"])
            effective_scope = {"doc_ids": effective_doc_ids, "kb_ids": stored_scope["kb_ids"]}
        result = chat(req.session_id, req.message, doc_ids=effective_doc_ids, scope=effective_scope)
        return ChatResponse(
            session_id=req.session_id,
            answer=result["answer"],
            sources=result["sources"],
            scope=effective_scope,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/sessions/{session_id}", response_model=list[ChatMessage])
async def get_session_messages(session_id: str):
    return list_session_messages(session_id)


@router.get("/sessions/{session_id}/scope", response_model=SessionScope)
async def get_scope(session_id: str):
    return get_session_scope(session_id)


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    delete_session(session_id)
    return {"message": f"会话 {session_id} 记忆已清除"}
