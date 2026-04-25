from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from book_see_rag.access_control import UserContext, get_current_user
from book_see_rag.metadata_store import create_knowledge_base, list_knowledge_bases_for_user

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


class KnowledgeBaseInfo(BaseModel):
    kb_id: str
    name: str
    visibility: str
    query_profile: str | None = None


class KnowledgeBaseCreateRequest(BaseModel):
    kb_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    visibility: str = "public"
    query_profile: str | None = None
    departments: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    user_ids: list[str] = Field(default_factory=list)


@router.get("", response_model=list[KnowledgeBaseInfo])
async def list_accessible_knowledge_bases(user: UserContext = Depends(get_current_user)):
    return list_knowledge_bases_for_user(user)


@router.post("", response_model=KnowledgeBaseInfo)
async def create_kb(payload: KnowledgeBaseCreateRequest, user: UserContext = Depends(get_current_user)):
    if user.role not in {"platform_admin", "kb_admin", "hr_admin"}:
        raise HTTPException(status_code=403, detail="仅管理员可创建知识库")
    try:
        return create_knowledge_base(
            kb_id=payload.kb_id,
            name=payload.name,
            visibility=payload.visibility,
            query_profile=payload.query_profile,
            tenant_id=user.tenant_id,
            departments=payload.departments,
            roles=payload.roles,
            user_ids=payload.user_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
