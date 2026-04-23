from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from book_see_rag.access_control import UserContext, get_current_user
from book_see_rag.metadata_store import delete_document as delete_document_metadata, list_documents_for_user
from book_see_rag.vectorstore.milvus_store import delete_by_doc_id

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    kb_id: str


@router.get("", response_model=list[DocumentInfo])
async def list_documents(user: UserContext = Depends(get_current_user)):
    return list_documents_for_user(user)


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user: UserContext = Depends(get_current_user)):
    try:
        allowed_docs = {doc["doc_id"] for doc in list_documents_for_user(user)}
        if doc_id not in allowed_docs:
            raise HTTPException(403, "当前用户无权删除该文档")
        delete_by_doc_id(doc_id)
        delete_document_metadata(doc_id)
        return {"message": f"文档 {doc_id} 已删除"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(500, str(e))
