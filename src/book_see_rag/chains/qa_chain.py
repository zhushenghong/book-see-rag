from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import time
from book_see_rag.chains.answer_cleanup import clean_answer_text
from book_see_rag.chains.answer_guardrails import find_unsupported_numbers
from book_see_rag.chains.answer_quality import inspect_answer_quality
from book_see_rag.config import get_settings
from book_see_rag.llm.factory import create_llm
from book_see_rag.retrieval import filter_meta_evaluation_chunks, prefilter_hits
from book_see_rag.vectorstore.milvus_store import get_doc_hits, search_hits
from book_see_rag.embedding.reranker import rerank
from book_see_rag.ingestion.splitter import is_noisy_chunk
from book_see_rag.retrievers.llamaindex_retriever import LlamaIndexUnavailable, search_hits_with_llamaindex
from book_see_rag.retrievers.scoped_documents import load_hits_from_uploads

logger = logging.getLogger("uvicorn.error")

_QUALITY_FAILURE_MESSAGE = "检索到相关证据，但生成结果未通过质量校验。请重试或缩小问题范围。"

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的文档分析助手。根据以下参考内容回答用户问题，"
     "若参考内容不足以回答，请如实说明。"
     "如果用户一次问多个问题，必须逐项回答，不能遗漏子问题。"
     "涉及日期、人数、比例、文件大小、技术名词时，必须严格照抄参考内容中的原文数字和名称，不要自行换算、修正或补全。"
     "不要把“推荐测试问题”或“标准答案要点”当成事实依据，除非用户明确询问测试题或标准答案。"
     "回答必须使用简洁、干净的中文，不要输出乱码、替换字符、重复标点或无意义符号。\n\n参考内容：\n{context}"),
    ("human", "{question}"),
])


_STRICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个严谨的文档事实抽取助手。只能根据参考内容回答。"
     "答案中的日期、人数、比例、文件大小、技术名词必须从参考内容中逐字复制。"
     "如果某个数字或术语在参考内容中没有原样出现，禁止写入答案。"
     "不要添加参考内容没有的背景、解释或推断。"
     "不要使用“推荐测试问题”或“标准答案要点”作为事实依据。"
     "只输出最终答案，不要输出 user、assistant、system、调试文本或无关问题。"
     "不要输出异常英文碎片、重复词、残缺列表或无意义符号。"
     "回答要简洁、完整、无乱码。\n\n参考内容：\n{context}"),
    ("human", "{question}"),
])


def _inspect_guardrails(answer_text: str, context: str) -> tuple[list[str], list[str]]:
    unsupported_numbers = find_unsupported_numbers(answer_text, context)
    quality_issues = inspect_answer_quality(answer_text)
    return unsupported_numbers, quality_issues


def _retrieve_hits(question: str, doc_ids: list[str] | None):
    settings = get_settings()
    if settings.retrieval_backend.lower() != "llamaindex":
        return search_hits(question, doc_ids=doc_ids)
    try:
        hits = search_hits_with_llamaindex(question, doc_ids=doc_ids)
        if hits:
            return hits
    except LlamaIndexUnavailable as exc:
        logger.warning("LlamaIndex unavailable, falling back to Milvus: %s", exc)
    except Exception:
        logger.exception("LlamaIndex retriever failed, falling back to Milvus")
    return search_hits(question, doc_ids=doc_ids)


def answer(question: str, doc_ids: list[str] | None = None) -> dict:
    """
    问答：两阶段检索（ANN + rerank）→ LLM 生成答案
    返回 {"answer": str, "sources": list[str]}
    """
    started = time.perf_counter()
    settings = get_settings()
    t0 = time.perf_counter()
    candidates = _retrieve_hits(question, doc_ids=doc_ids)
    t1 = time.perf_counter()
    candidates = [item for item in candidates if not is_noisy_chunk(item["content"])]
    candidates = filter_meta_evaluation_chunks(candidates)
    candidates = prefilter_hits(question, candidates, settings.retrieval_prefilter_top_k)
    if not candidates and doc_ids:
        try:
            fallback = get_doc_hits(doc_ids, limit=settings.retrieval_prefilter_top_k)
        except Exception:
            logger.exception("QA doc fallback failed to load chunks from Milvus, using upload fallback")
            fallback = []
        if not fallback:
            fallback = load_hits_from_uploads(doc_ids, limit=settings.retrieval_prefilter_top_k)
        candidates = [item for item in fallback if not is_noisy_chunk(item["content"])]
    candidate_texts = [item["content"] for item in candidates]
    if settings.enable_rerank:
        ranked_chunks = rerank(question, candidate_texts)
    else:
        ranked_chunks = candidate_texts[:settings.rerank_top_k]
        logger.info("QA rerank disabled using top_k=%s from search results", len(ranked_chunks))
    t2 = time.perf_counter()

    ranked_chunks = filter_meta_evaluation_chunks([
        {"doc_id": "", "filename": "", "page": 0, "content": chunk, "score": 0.0}
        for chunk in ranked_chunks
    ])
    ranked_chunks = [item["content"] for item in ranked_chunks]

    context = "\n\n---\n\n".join(ranked_chunks)
    llm = create_llm()
    chain = _PROMPT | llm | StrOutputParser()
    answer_text = clean_answer_text(chain.invoke({"context": context, "question": question}))
    unsupported_numbers, quality_issues = _inspect_guardrails(answer_text, context)
    if unsupported_numbers or quality_issues:
        logger.warning(
            "QA answer failed guardrails unsupported_numbers=%s quality_issues=%s, retrying strict generation",
            unsupported_numbers,
            quality_issues,
        )
        strict_chain = _STRICT_PROMPT | llm | StrOutputParser()
        answer_text = clean_answer_text(strict_chain.invoke({"context": context, "question": question}))
        unsupported_numbers, quality_issues = _inspect_guardrails(answer_text, context)
        if unsupported_numbers or quality_issues:
            logger.warning(
                "QA strict answer failed guardrails unsupported_numbers=%s quality_issues=%s, returning quality failure",
                unsupported_numbers,
                quality_issues,
            )
            answer_text = _QUALITY_FAILURE_MESSAGE
    t3 = time.perf_counter()
    logger.info(
        "QA timings search=%.3fs rerank=%.3fs llm=%.3fs total=%.3fs candidates=%s ranked=%s",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t3 - started,
        len(candidate_texts),
        len(ranked_chunks),
    )

    return {"answer": answer_text, "sources": ranked_chunks}
