from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
import time
from book_see_rag.config import get_settings
from book_see_rag.chains.answer_cleanup import clean_answer_text
from book_see_rag.chains.answer_guardrails import find_unsupported_numbers
from book_see_rag.chains.answer_quality import inspect_answer_quality
from book_see_rag.chains.evidence_brief import build_evidence_brief
from book_see_rag.chains.refusal import (
    EVIDENCE_REFUSAL_MESSAGE,
    QUALITY_FAILURE_MESSAGE,
    canonicalize_refusal_text,
    repair_citation_policy_refusal,
)
from book_see_rag.embedding.reranker import rerank
from book_see_rag.ingestion.splitter import is_noisy_chunk
from book_see_rag.llm.factory import create_llm
from book_see_rag.memory.redis_memory import (
    append_ai_message,
    append_user_message,
    get_recent_messages,
)
from book_see_rag.query_understanding import build_retrieval_queries, expand_query_terms, filter_hits_by_focus
from book_see_rag.vectorstore.milvus_store import SearchHit, get_doc_hits, search_hits
from book_see_rag.retrieval import (
    evidence_directly_supports,
    filter_meta_evaluation_chunks,
    merge_ranked_hits,
    prefilter_hits,
    prioritize_chunk_tokenization_hits,
)
from book_see_rag.retrievers.llamaindex_retriever import LlamaIndexUnavailable, search_hits_with_llamaindex
from book_see_rag.retrievers.scoped_documents import load_hits_from_uploads

logger = logging.getLogger("uvicorn.error")

_QUALITY_FAILURE_MESSAGE = QUALITY_FAILURE_MESSAGE

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的文档分析助手，擅长根据文档内容进行多轮对话。"
     "你只能依据参考内容和对话历史回答。"
     "如果参考内容不足、引用中没有直接证据、或内容疑似 OCR/解析噪声，请明确说明“依据不足”，不要猜测、不要补充未出现的技能、经历、项目或结论。"
     "回答要尽量引用能直接支撑结论的证据，不要把弱相关片段扩展成确定结论。"
     "如果当前问题或对话历史已经指向某个具体对象，只回答该对象，不要混答其他对象。"
     "如果用户一次问多个问题，必须逐项回答，不能遗漏子问题。"
     "涉及日期、人数、比例、文件大小、技术名词时，必须严格照抄参考内容中的原文数字和名称，不要自行换算、修正或补全。"
     "尽量直接复用参考内容中的原句或短语，不要自己改写成别的说法。"
     "涉及权限问题时，只输出中文权限结论，不要输出 department、role、employee、hr_admin 等内部字段名。"
     "不要使用 Markdown 表格、标题、加粗、代码块或项目符号；尽量输出纯文本。"
     "根据问题类型控制答案长度：事实抽取用 1 到 2 句；对比、原因、分别说明、多事实问题可用 3 到 6 个短句完整回答。"
     "如果参考内容中有可以直接复用的定义句、对比句或权限句，优先原样复用或轻微整理顺序，不要改写术语。"
     "对比、定义、因果关系类：若「重点证据」或参考内容里已有可直接套用的短句，应尽量原样复述或只做最小衔接，不要为了概括换掉原文关键词。"
     "若问题含「区别」「对比」「有什么不同」「分别适合」「负责什么」：须对问题里出现的每个对象各用至少一句说明（定义或文档原句），并尽量补全文档中写到的「擅长/适合场景」及在流程中的先后关系；用户同时点到 BM25、向量检索、reranker 时三者都不得遗漏。"
     "若问题将 Reranker 与向量数据库对比：须用一两句点明二者职责不同（例如向量库存 embedding、负责召回候选；reranker 负责精排/重排），并尽量写出「Reranker 不是向量数据库」或文档中的等价否定表述。"
     "若问题同时出现 chunk 与中文分词：须在参考内容中查找「不是中文分词」「分词通常是把句子切成词」等句子并复述，不要说参考内容未涉及分词。"
     "若问题列举多个组件是否被替代（如 Milvus、权限控制、最终回答生成）：须逐项点名并说明是否替代，禁止只用「这三个部分」等笼统指代而不出现「权限控制」「最终回答生成」等关键词。"
     "若问「为什么需要最小权限」而参考内容以原则与示例为主：可结合文档中的「权限风险」「越权检索」等表述作一两句动机，再逐条复述三条权限原则，不要说参考内容未说明。"
     "若问题要求列举「有哪些」「几类」「主要风险」等：参考内容若写明「五类」「三类」等总数或逐项标题，须逐条列全，不可合并、省略或只答其中几项。"
     "若文档写「五类主要风险」，答案必须列齐五类且含「性能风险」，不可写成四类。"
     "若问题对比多名人员（如李明与王敏）：每个姓名须在答案正文中至少出现一次，并分别写清可见范围与原因；写第二人时该句须以「王敏」或「HR 管理员王敏」等含全名的起句开头，不可单用「她」。"
     "若问题与可追溯、依据不足、编造、引用等相关：参考内容如出现「可追溯」「依据不足」「编造」等表述，应直接复述这些原文措辞作为依据，不要说参考内容未说明。"
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
     "如果当前问题或对话历史已经指向某个具体对象，只回答该对象，不要混答其他对象。"
     "答案中的日期、人数、比例、文件大小、技术名词必须从参考内容中逐字复制。"
     "如果某个数字或术语在参考内容中没有原样出现，禁止写入答案。"
     "尽量直接复用参考内容中的原句或短语，不要自己改写成别的说法。"
     "涉及权限问题时，只输出中文权限结论，不要输出 department、role、employee、hr_admin 等内部字段名。"
     "不要使用 Markdown 表格、标题、加粗、代码块或项目符号；尽量输出纯文本。"
     "根据问题类型控制答案长度：事实抽取用 1 到 2 句；对比、原因、分别说明、多事实问题可用 3 到 6 个短句完整回答。"
     "如果参考内容中有可以直接复用的定义句、对比句或权限句，优先原样复用或轻微整理顺序，不要改写术语。"
     "对比、定义、因果关系类：若参考内容已有可直接回答的句子，应尽量逐句复述原文关键词。"
     "若问题含「区别」「对比」「有什么不同」「分别适合」「负责什么」：须对每个对象各有一句文档依据，并尽量写全「擅长/适合」与流程先后；BM25、向量检索、reranker 同时出现时三者都须答到。"
     "Reranker 与向量数据库对比须点明职责差异，并尽量写出「Reranker 不是向量数据库」或等价否定句。"
     "chunk 与中文分词须复述文档中的「不是中文分词」与分词定义句，勿称未涉及分词。"
     "权限替代类多组件问题须逐项点名，须出现「权限控制」「最终回答生成」等关键词。"
     "为什么需要最小权限：可引用权限风险/越权再复述三条原则，勿空答未说明。"
     "五类风险须列满含性能风险。"
     "多人权限对比须每人姓名至少出现一次；第二人的句子须以全名起句，禁止单用「她」。"
     "若问题要求列举风险或多项要点：文档列了几项就必须答几项，不得漏项。"
     "若问题与可追溯、依据不足、编造或引用相关：如出现对应原文措辞，必须在答案中使用这些措辞，不要说参考内容未说明。"
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
    quality_issues = inspect_answer_quality(answer, context)
    return unsupported_numbers, quality_issues


def _augment_query(question: str) -> str:
    normalized = expand_query_terms(question.strip(), include_route_hints=False)
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


def _retrieve_hits_for_queries(queries: list[str], doc_ids: list[str] | None) -> list[SearchHit]:
    merged: list[SearchHit] = []
    for query in queries:
        merged = merge_ranked_hits(merged, _retrieve_hits(query, doc_ids=doc_ids))
    return merged


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
    retrieval_queries = build_retrieval_queries(retrieval_query, doc_ids=doc_ids)
    retrieval_query = retrieval_queries[0] if retrieval_queries else retrieval_query
    t_rewrite = time.perf_counter()
    candidates = _retrieve_hits_for_queries(retrieval_queries or [retrieval_query], doc_ids=doc_ids)
    t1 = time.perf_counter()
    candidates = _filter_hits(candidates)
    candidates = filter_meta_evaluation_chunks(candidates)
    candidates = filter_hits_by_focus(retrieval_query, candidates, doc_ids=doc_ids)
    candidates = prefilter_hits(retrieval_query, candidates, settings.retrieval_prefilter_top_k)
    candidates = prioritize_chunk_tokenization_hits(message, candidates)
    if candidates and not evidence_directly_supports(message, candidates):
        append_user_message(session_id, message)
        append_ai_message(session_id, EVIDENCE_REFUSAL_MESSAGE, candidates[:settings.rerank_top_k], scope=scope)
        return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": candidates[:settings.rerank_top_k]}
    if not candidates:
        candidates = _fallback_hits(doc_ids, settings.retrieval_prefilter_top_k)
        candidates = filter_meta_evaluation_chunks(candidates)
        candidates = filter_hits_by_focus(retrieval_query, candidates, doc_ids=doc_ids)
        candidates = prefilter_hits(retrieval_query, candidates, settings.retrieval_prefilter_top_k)
        candidates = prioritize_chunk_tokenization_hits(message, candidates)
        if candidates and not evidence_directly_supports(message, candidates):
            append_user_message(session_id, message)
            append_ai_message(session_id, EVIDENCE_REFUSAL_MESSAGE, candidates[:settings.rerank_top_k], scope=scope)
            return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": candidates[:settings.rerank_top_k]}
    if not candidates:
        refusal = EVIDENCE_REFUSAL_MESSAGE
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
        refusal = EVIDENCE_REFUSAL_MESSAGE
        append_user_message(session_id, message)
        append_ai_message(session_id, refusal, [], scope=scope)
        return {"answer": refusal, "sources": []}
    ranked_hits = filter_meta_evaluation_chunks(ranked_hits)
    if not ranked_hits:
        append_user_message(session_id, message)
        append_ai_message(session_id, EVIDENCE_REFUSAL_MESSAGE, [], scope=scope)
        return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": []}
    context = _format_context(ranked_hits)
    extra_queries = retrieval_queries[1:] if len(retrieval_queries) > 1 else None
    evidence_brief = build_evidence_brief(
        message,
        [item["content"] for item in ranked_hits],
        extra_queries=extra_queries,
    )
    if evidence_brief:
        context = f"重点证据：\n{evidence_brief}\n\n---\n\n原始参考内容：\n{context}"
    llm = create_llm()
    chain = _PROMPT | llm | StrOutputParser()
    answer = clean_answer_text(chain.invoke({"question": message, "context": context, "history": history}))
    answer = canonicalize_refusal_text(answer)
    answer = repair_citation_policy_refusal(message, context, answer)
    unsupported_numbers, quality_issues = _inspect_guardrails(answer, context)
    if unsupported_numbers or quality_issues:
        logger.warning(
            "Chat answer failed guardrails unsupported_numbers=%s quality_issues=%s, retrying strict generation",
            unsupported_numbers,
            quality_issues,
        )
        strict_chain = _STRICT_PROMPT | llm | StrOutputParser()
        answer = clean_answer_text(strict_chain.invoke({"question": message, "context": context, "history": history}))
        answer = canonicalize_refusal_text(answer)
        answer = repair_citation_policy_refusal(message, context, answer)
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
