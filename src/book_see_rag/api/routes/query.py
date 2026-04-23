from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from book_see_rag.access_control import UserContext, get_current_user
from book_see_rag.chains.qa_chain import answer
from book_see_rag.chains.summary_chain import summarize
from book_see_rag.chains.extraction_chain import extract
from book_see_rag.metadata_store import resolve_allowed_doc_ids

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    mode: Literal["qa", "summary", "extraction"] = "qa"
    doc_ids: list[str] | None = None
    kb_ids: list[str] | None = None


class QueryResponse(BaseModel):
    mode: str
    result: str | dict
    sources: list[str] | None = None


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest, user: UserContext = Depends(get_current_user)):
    try:
        effective_doc_ids = resolve_allowed_doc_ids(user, req.doc_ids, req.kb_ids)
        match req.mode:
            case "qa":
                out = answer(req.question, doc_ids=effective_doc_ids)
                return QueryResponse(mode="qa", result=out["answer"], sources=out["sources"])

            case "summary":
                text = summarize(doc_ids=effective_doc_ids, topic=req.question)
                return QueryResponse(mode="summary", result=text)

            case "extraction":
                item = extract(query=req.question, doc_ids=effective_doc_ids)
                return QueryResponse(mode="extraction", result=item.model_dump())

    except Exception as e:
        raise HTTPException(500, str(e))
