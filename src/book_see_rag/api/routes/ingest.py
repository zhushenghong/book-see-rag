import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult
from book_see_rag.config import get_settings
from book_see_rag.metadata_store import get_knowledge_base, register_document
from book_see_rag.tasks.ingest_task import celery_app, ingest_document

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    task_id: str
    doc_id: str
    filename: str
    kb_id: str
    message: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    detail: dict | None = None


ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@router.post("", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), kb_id: str = Form("kb_public")):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"不支持的文件格式：{suffix}")
    if not get_knowledge_base(kb_id):
        raise HTTPException(400, f"未知知识库：{kb_id}")

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    dest = upload_dir / f"{doc_id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    register_document(doc_id, file.filename, kb_id)
    task = ingest_document.delay(doc_id, str(dest), file.filename)
    return IngestResponse(
        task_id=task.id,
        doc_id=doc_id,
        filename=file.filename,
        kb_id=kb_id,
        message="文档已接收，正在后台处理",
    )


@router.get("/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    status_map = {
        "PENDING":  "pending",
        "STARTED":  "processing",
        "PROGRESS": "processing",
        "SUCCESS":  "done",
        "FAILURE":  "failed",
    }
    try:
        state = result.state
    except ValueError:
        # Redis 中存储的旧任务结果格式不兼容，返回失败状态
        return TaskStatus(
            task_id=task_id,
            status="failed",
            detail={"error": "任务状态无法读取，可能是旧数据格式问题"},
        )
    return TaskStatus(
        task_id=task_id,
        status=status_map.get(state, state.lower()),
        detail=result.info if isinstance(result.info, dict) else None,
    )
