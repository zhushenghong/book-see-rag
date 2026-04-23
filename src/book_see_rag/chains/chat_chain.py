from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import time
from book_see_rag.config import get_settings
from book_see_rag.chains.answer_cleanup import clean_answer_text
from book_see_rag.chains.answer_guardrails import find_unsupported_numbers
from book_see_rag.chains.answer_quality import inspect_answer_quality
from book_see_rag.embedding.reranker import rerank
from book_see_rag.ingestion.splitter import is_noisy_chunk
from book_see_rag.llm.factory import create_llm
from book_see_rag.memory.redis_memory import (
    append_ai_message,
    append_user_message,
    get_recent_messages,
)
from book_see_rag.vectorstore.milvus_store import SearchHit, get_doc_hits, search_hits
from book_see_rag.retrieval import filter_meta_evaluation_chunks, prefilter_hits
from book_see_rag.retrievers.llamaindex_retriever import LlamaIndexUnavailable, search_hits_with_llamaindex
from book_see_rag.retrievers.scoped_documents import load_hits_from_uploads

logger = logging.getLogger("uvicorn.error")

_QUALITY_FAILURE_MESSAGE = "检索到相关证据，但生成结果未通过质量校验。请重试或缩小问题范围。"

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的文档分析助手，擅长根据文档内容进行多轮对话。"
     "你只能依据参考内容和对话历史回答。"
     "如果参考内容不足、引用中没有直接证据、或内容疑似 OCR/解析噪声，请明确说明“依据不足”，不要猜测、不要补充未出现的技能、经历、项目或结论。"
     "回答要尽量引用能直接支撑结论的证据，不要把弱相关片段扩展成确定结论。"
     "如果用户一次问多个问题，必须逐项回答，不能遗漏子问题。"
     "涉及日期、人数、比例、文件大小、技术名词时，必须严格照抄参考内容中的原文数字和名称，不要自行换算、修正或补全。"
     "不要把“推荐测试问题”或“标准答案要点”当成事实依据，除非用户明确询问测试题或标准答案。"
     "回答必须使用简洁、干净的中文，不要输出乱码、替换字符、重复标点或无意义符号。\n\n参考内容：\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你负责把用户的追问改写成适合文档检索的独立问题。"
        "如果当前问题已经完整明确，直接原样返回。"
        "不要回答问题，只输出改写后的检索问题。",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


_STRICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个严谨的文档事实抽取助手。只能根据参考内容和对话历史回答。"
     "答案中的日期、人数、比例、文件大小、技术名词必须从参考内容中逐字复制。"
     "如果某个数字或术语在参考内容中没有原样出现，禁止写入答案。"
     "不要添加参考内容没有的背景、解释或推断。"
     "不要使用“推荐测试问题”或“标准答案要点”作为事实依据。"
     "只输出最终答案，不要输出 user、assistant、system、调试文本或无关问题。"
     "不要输出异常英文碎片、重复词、残缺列表或无意义符号。"
     "回答要简洁、完整、无乱码。\n\n参考内容：\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def _inspect_guardrails(answer: str, context: str) -> tuple[list[str], list[str]]:
    unsupported_numbers = find_unsupported_numbers(answer, context)
    quality_issues = inspect_answer_quality(answer)
    return unsupported_numbers, quality_issues


def _augment_query(question: str) -> str:
    normalized = question.strip()
    expansions: list[str] = []
    if any(term in normalized for term in ["技能", "擅长", "会什么", "能力", "技术栈"]):
        expansions.append("技能 技术栈 擅长 熟悉 掌握 能力")
    if any(term in normalized for term in ["项目", "经历", "做过", "负责", "参与"]):
        expansions.append("项目经历 项目经验 负责 参与 技术方案 工作经历")
    if any(term in normalized for term in ["简历", "候选人", "这个人", "该人", "他有哪些"]):
        expansions.append("简历 候选人 个人经历 专业技能 项目经验")
    if not expansions:
        return normalized
    return f"{normalized}\n检索关注：{'；'.join(expansions)}"


def _rerank_hits(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    if not hits:
        return []
    ranked_contents = rerank(query, [item["content"] for item in hits], top_k=top_k)
    buckets: dict[str, list[SearchHit]] = {}
    for item in hits:
        buckets.setdefault(item["content"], []).append(item)
    ranked: list[SearchHit] = []
    for content in ranked_contents:
        bucket = buckets.get(content)
        if bucket:
            ranked.append(bucket.pop(0))
    return ranked


def _filter_hits(hits: list[SearchHit]) -> list[SearchHit]:
    filtered = [item for item in hits if not is_noisy_chunk(item["content"])]
    logger.info("Chat evidence filtered total=%s kept=%s", len(hits), len(filtered))
    return filtered


def _fallback_hits(doc_ids: list[str] | None, limit: int) -> list[SearchHit]:
    if not doc_ids:
        return []
    try:
        fallback = get_doc_hits(doc_ids, limit=limit)
    except Exception:
        logger.exception("Chat doc fallback failed to load chunks from Milvus, using upload fallback")
        fallback = []
    if not fallback:
        fallback = load_hits_from_uploads(doc_ids, limit=limit)
    fallback = _filter_hits(fallback)
    logger.info("Chat fallback hits doc_scope=%s kept=%s", len(doc_ids), len(fallback))
    return fallback


def _retrieve_hits(query: str, doc_ids: list[str] | None) -> list[SearchHit]:
    settings = get_settings()
    if settings.retrieval_backend.lower() != "llamaindex":
        return search_hits(query, doc_ids=doc_ids)
    try:
        hits = search_hits_with_llamaindex(query, doc_ids=doc_ids)
        if hits:
            return hits
    except LlamaIndexUnavailable as exc:
        logger.warning("LlamaIndex unavailable, falling back to Milvus: %s", exc)
    except Exception:
        logger.exception("LlamaIndex retriever failed, falling back to Milvus")
    return search_hits(query, doc_ids=doc_ids)


def _format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for idx, item in enumerate(hits, 1):
        page = f"第 {item['page']} 页" if item["page"] else "页码未知"
        blocks.append(f"[证据{idx}] {item['filename']} | {page}\n{item['content']}")
    return "\n\n---\n\n".join(blocks)


def _rewrite_query(question: str, history: list[HumanMessage | AIMessage]) -> str:
    base_query = _augment_query(question)
    if not history:
        return base_query

    chain = _REWRITE_PROMPT | create_llm() | StrOutputParser()
    rewritten = chain.invoke({"history": history, "question": base_query}).strip()
    return rewritten or base_query


def chat(
    session_id: str,
    message: str,
    doc_ids: list[str] | None = None,
    scope: dict[str, list[str]] | None = None,
) -> dict:
    """
    多轮对话：两阶段检索 + Redis 记忆
    返回 {"answer": str, "sources": list[str]}
    """
    started = time.perf_counter()
    settings = get_settings()
    history = get_recent_messages(session_id, limit=settings.chat_history_window)
    rewrite_history = history[-settings.followup_rewrite_history:]
    t0 = time.perf_counter()
    retrieval_query = _rewrite_query(message, rewrite_history)
    t_rewrite = time.perf_counter()
    candidates = _retrieve_hits(retrieval_query, doc_ids=doc_ids)
    t1 = time.perf_counter()
    candidates = _filter_hits(candidates)
    candidates = filter_meta_evaluation_chunks(candidates)
    candidates = prefilter_hits(retrieval_query, candidates, settings.retrieval_prefilter_top_k)
    if not candidates:
        candidates = _fallback_hits(doc_ids, settings.retrieval_prefilter_top_k)
    if not candidates:
        refusal = "当前检索到的内容解析质量较差或证据不足，无法可靠回答这个问题。请尝试切换文档、重新上传更清晰的文件，或缩小问题范围。"
        append_user_message(session_id, message)
        append_ai_message(session_id, refusal, [], scope=scope)
        return {"answer": refusal, "sources": []}
    if settings.enable_rerank:
        ranked_hits = _rerank_hits(retrieval_query, candidates, top_k=settings.rerank_top_k)
    else:
        ranked_hits = candidates[:settings.rerank_top_k]
        logger.info("Chat rerank disabled using top_k=%s from search results", len(ranked_hits))
    t2 = time.perf_counter()
    if not ranked_hits:
        refusal = "当前没有足够干净且直接相关的证据，无法可靠回答这个问题。"
        append_user_message(session_id, message)
        append_ai_message(session_id, refusal, [], scope=scope)
        return {"answer": refusal, "sources": []}
    ranked_hits = filter_meta_evaluation_chunks(ranked_hits)
    context = _format_context(ranked_hits)
    llm = create_llm()
    chain = _PROMPT | llm | StrOutputParser()
    answer = clean_answer_text(chain.invoke({"question": message, "context": context, "history": history}))
    unsupported_numbers, quality_issues = _inspect_guardrails(answer, context)
    if unsupported_numbers or quality_issues:
        logger.warning(
            "Chat answer failed guardrails unsupported_numbers=%s quality_issues=%s, retrying strict generation",
            unsupported_numbers,
            quality_issues,
        )
        strict_chain = _STRICT_PROMPT | llm | StrOutputParser()
        answer = clean_answer_text(strict_chain.invoke({"question": message, "context": context, "history": history}))
        unsupported_numbers, quality_issues = _inspect_guardrails(answer, context)
        if unsupported_numbers or quality_issues:
            logger.warning(
                "Chat strict answer failed guardrails unsupported_numbers=%s quality_issues=%s, returning quality failure",
                unsupported_numbers,
                quality_issues,
            )
            answer = _QUALITY_FAILURE_MESSAGE
    append_user_message(session_id, message)
    append_ai_message(session_id, answer, ranked_hits, scope=scope)
    t3 = time.perf_counter()
    logger.info(
        "Chat timings rewrite=%.3fs search=%.3fs rerank=%.3fs llm=%.3fs total=%.3fs candidates=%s ranked=%s rewritten=%r",
        t_rewrite - t0,
        t1 - t_rewrite,
        t2 - t1,
        t3 - t2,
        t3 - started,
        len(candidates),
        len(ranked_hits),
        retrieval_query[:120],
    )
    return {"answer": answer, "sources": ranked_hits}
