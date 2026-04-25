from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Iterable

from sqlalchemy import ForeignKeyConstraint, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from book_see_rag.config import get_settings


class Base(DeclarativeBase):
    pass


class KnowledgeBaseRow(Base):
    __tablename__ = "knowledge_bases"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    kb_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False, default="public")
    query_profile: Mapped[str | None] = mapped_column(String(256), nullable=True)

    departments_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    roles_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    user_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    documents: Mapped[list["DocumentRow"]] = relationship(back_populates="kb", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "visibility": self.visibility,
            "query_profile": self.query_profile,
            "departments": json.loads(self.departments_json or "[]"),
            "roles": json.loads(self.roles_json or "[]"),
            "user_ids": json.loads(self.user_ids_json or "[]"),
        }

    @staticmethod
    def from_record(record: dict[str, Any]) -> "KnowledgeBaseRow":
        return KnowledgeBaseRow(
            kb_id=record["kb_id"],
            tenant_id=record.get("tenant_id") or "public",
            name=record["name"],
            visibility=record.get("visibility") or "public",
            query_profile=record.get("query_profile"),
            departments_json=json.dumps(record.get("departments") or [], ensure_ascii=False),
            roles_json=json.dumps(record.get("roles") or [], ensure_ascii=False),
            user_ids_json=json.dumps(record.get("user_ids") or [], ensure_ascii=False),
        )


class DocumentRow(Base):
    __tablename__ = "documents"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "kb_id"],
            ["knowledge_bases.tenant_id", "knowledge_bases.kb_id"],
        ),
    )

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    kb_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    kb: Mapped[KnowledgeBaseRow] = relationship(back_populates="documents")

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "filename": self.filename, "kb_id": self.kb_id, "tenant_id": self.tenant_id}


class IngestTaskRow(Base):
    __tablename__ = "ingest_tasks"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "doc_id": self.doc_id, "tenant_id": self.tenant_id}


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    if not settings.metadata_db_url.strip():
        raise ValueError("metadata_db_url 未配置")
    return create_engine(settings.metadata_db_url, pool_pre_ping=True)


def ensure_schema() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


def seed_knowledge_bases(default_kbs: Iterable[dict[str, Any]]) -> None:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        rows = session.execute(select(KnowledgeBaseRow.tenant_id, KnowledgeBaseRow.kb_id)).all()
        existing = {(tenant_id, kb_id) for tenant_id, kb_id in rows}
        missing = [kb for kb in default_kbs if ((kb.get("tenant_id") or "public"), kb["kb_id"]) not in existing]
        if not missing:
            return
        for kb in missing:
            session.add(KnowledgeBaseRow.from_record(kb))
        session.commit()


def list_knowledge_bases(tenant_id: str | None = None) -> list[dict[str, Any]]:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(KnowledgeBaseRow)
        if tenant_id:
            stmt = stmt.where(KnowledgeBaseRow.tenant_id == tenant_id)
        rows = session.scalars(stmt).all()
        return [row.to_dict() for row in rows]


def create_knowledge_base(record: dict[str, Any]) -> dict[str, Any]:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        tenant_id = record.get("tenant_id") or "public"
        exists = session.get(KnowledgeBaseRow, (tenant_id, record["kb_id"]))
        if exists:
            raise ValueError(f"知识库 {record['kb_id']} 已存在")
        row = KnowledgeBaseRow.from_record(record)
        session.add(row)
        session.commit()
        return row.to_dict()


def get_knowledge_base(kb_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        if tenant_id:
            row = session.get(KnowledgeBaseRow, (tenant_id, kb_id))
        else:
            row = session.scalars(select(KnowledgeBaseRow).where(KnowledgeBaseRow.kb_id == kb_id)).first()
        return row.to_dict() if row else None


def upsert_document(doc_id: str, filename: str, kb_id: str, tenant_id: str = "public") -> None:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        row = session.get(DocumentRow, (tenant_id, doc_id))
        if row:
            row.tenant_id = tenant_id
            row.filename = filename
            row.kb_id = kb_id
        else:
            session.add(DocumentRow(doc_id=doc_id, tenant_id=tenant_id, filename=filename, kb_id=kb_id))
        session.commit()


def delete_document(doc_id: str, tenant_id: str | None = None) -> None:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        if tenant_id:
            rows = [session.get(DocumentRow, (tenant_id, doc_id))]
        else:
            rows = list(session.scalars(select(DocumentRow).where(DocumentRow.doc_id == doc_id)).all())
        for row in rows:
            if row:
                session.delete(row)
        session.commit()


def list_documents(tenant_id: str | None = None) -> list[dict[str, Any]]:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(DocumentRow)
        if tenant_id:
            stmt = stmt.where(DocumentRow.tenant_id == tenant_id)
        rows = session.scalars(stmt).all()
        return [row.to_dict() for row in rows]


def upsert_ingest_task(task_id: str, doc_id: str, tenant_id: str = "public") -> None:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        row = session.get(IngestTaskRow, (tenant_id, task_id))
        if row:
            row.doc_id = doc_id
        else:
            session.add(IngestTaskRow(tenant_id=tenant_id, task_id=task_id, doc_id=doc_id))
        session.commit()


def get_ingest_task(task_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
    ensure_schema()
    engine = get_engine()
    with Session(engine) as session:
        if tenant_id:
            row = session.get(IngestTaskRow, (tenant_id, task_id))
        else:
            row = session.scalars(select(IngestTaskRow).where(IngestTaskRow.task_id == task_id)).first()
        return row.to_dict() if row else None

