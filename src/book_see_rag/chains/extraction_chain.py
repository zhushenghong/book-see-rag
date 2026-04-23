from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import logging
import time
from book_see_rag.config import get_settings
from book_see_rag.llm.factory import create_llm
from book_see_rag.vectorstore.milvus_store import search
from book_see_rag.embedding.reranker import rerank

logger = logging.getLogger("uvicorn.error")


class KnowledgeItem(BaseModel):
    entities: list[str] = Field(description="文中出现的关键实体（人物、地点、组织、概念等）")
    key_facts: list[str] = Field(description="文档中的关键事实或结论，每条一句话")
    relationships: list[str] = Field(description="实体间的关系描述，格式：A → 关系 → B")


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的信息提取助手。请从以下文档内容中提取结构化知识。\n\n文档内容：\n{context}"),
    ("human", "请按要求提取实体、关键事实和关系"),
])


def extract(query: str = "主要知识点", doc_ids: list[str] | None = None) -> KnowledgeItem:
    """
    知识提取：检索相关 chunks → 结构化输出（Pydantic）
    """
    started = time.perf_counter()
    settings = get_settings()
    t0 = time.perf_counter()
    candidates = search(query, doc_ids=doc_ids)
    t1 = time.perf_counter()
    if settings.enable_rerank:
        ranked_chunks = rerank(query, candidates)
    else:
        ranked_chunks = candidates[:settings.rerank_top_k]
        logger.info("Extraction rerank disabled using top_k=%s from search results", len(ranked_chunks))
    t2 = time.perf_counter()
    context = "\n\n---\n\n".join(ranked_chunks)

    llm = create_llm()
    structured_llm = llm.with_structured_output(KnowledgeItem)
    chain = _PROMPT | structured_llm
    result = chain.invoke({"context": context})
    t3 = time.perf_counter()
    logger.info(
        "Extraction timings search=%.3fs rerank=%.3fs llm=%.3fs total=%.3fs candidates=%s ranked=%s",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t3 - started,
        len(candidates),
        len(ranked_chunks),
    )
    return result
