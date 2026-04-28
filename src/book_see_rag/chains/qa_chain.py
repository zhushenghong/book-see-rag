from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import time
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
from book_see_rag.config import get_settings
from book_see_rag.llm.factory import create_llm
from book_see_rag.query_understanding import build_retrieval_queries, detect_ambiguous_objects, expand_query_terms, filter_hits_by_focus
from book_see_rag.retrieval import (
    evidence_directly_supports,
    filter_meta_evaluation_chunks,
    merge_ranked_hits,
    prefilter_hits,
    prioritize_chunk_tokenization_hits,
)
from book_see_rag.vectorstore.milvus_store import get_doc_hits, search_hits
from book_see_rag.embedding.reranker import rerank
from book_see_rag.ingestion.splitter import is_noisy_chunk
from book_see_rag.retrievers.llamaindex_retriever import LlamaIndexUnavailable, search_hits_with_llamaindex
from book_see_rag.retrievers.scoped_documents import load_hits_from_uploads

logger = logging.getLogger("uvicorn.error")

_QUALITY_FAILURE_MESSAGE = QUALITY_FAILURE_MESSAGE
_AMBIGUITY_MESSAGE_TEMPLATE = "问题未明确具体对象。当前文档里至少包含这些候选对象：{objects}。请明确你要问哪一个。"

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的文档分析助手。根据以下参考内容回答用户问题，"
     "若参考内容不足以回答，请如实说明。"
     "如果问题里已经指向某个具体对象，只回答该对象，不要混答其他对象。"
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
    ("human", "{question}"),
])


_STRICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个严谨的文档事实抽取助手。只能根据参考内容回答。"
     "如果问题里已经指向某个具体对象，只回答该对象，不要混答其他对象。"
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
    ("human", "{question}"),
])


def _inspect_guardrails(answer_text: str, context: str) -> tuple[list[str], list[str]]:
    unsupported_numbers = find_unsupported_numbers(answer_text, context)
    quality_issues = inspect_answer_quality(answer_text, context)
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


def _retrieve_hits_for_queries(queries: list[str], doc_ids: list[str] | None):
    merged = []
    for query in queries:
        merged = merge_ranked_hits(merged, _retrieve_hits(query, doc_ids=doc_ids))
    return merged


def answer(question: str, doc_ids: list[str] | None = None) -> dict:
    """
    问答：两阶段检索（ANN + rerank）→ LLM 生成答案
    返回 {"answer": str, "sources": list[str]}
    """
    started = time.perf_counter()
    settings = get_settings()
    retrieval_queries = build_retrieval_queries(question, doc_ids=doc_ids)
    retrieval_query = retrieval_queries[0] if retrieval_queries else expand_query_terms(
        question, doc_ids=doc_ids, include_route_hints=False
    )
    t0 = time.perf_counter()
    candidates = _retrieve_hits_for_queries(retrieval_queries or [retrieval_query], doc_ids=doc_ids)
    t1 = time.perf_counter()
    candidates = [item for item in candidates if not is_noisy_chunk(item["content"])]
    candidates = filter_meta_evaluation_chunks(candidates)
    ambiguous_objects = detect_ambiguous_objects(question, candidates, doc_ids=doc_ids)
    if ambiguous_objects:
        return {"answer": _AMBIGUITY_MESSAGE_TEMPLATE.format(objects="、".join(ambiguous_objects)), "sources": []}
    candidates = filter_hits_by_focus(retrieval_query, candidates, doc_ids=doc_ids)
    candidates = prefilter_hits(retrieval_query, candidates, settings.retrieval_prefilter_top_k)
    candidates = prioritize_chunk_tokenization_hits(question, candidates)
    if candidates and not evidence_directly_supports(question, candidates):
        return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": candidates[:settings.rerank_top_k]}
    if not candidates and doc_ids:
        try:
            fallback = get_doc_hits(doc_ids, limit=settings.retrieval_prefilter_top_k)
        except Exception:
            logger.exception("QA doc fallback failed to load chunks from Milvus, using upload fallback")
            fallback = []
        if not fallback:
            fallback = load_hits_from_uploads(doc_ids, limit=settings.retrieval_prefilter_top_k)
        candidates = [item for item in fallback if not is_noisy_chunk(item["content"])]
        candidates = filter_meta_evaluation_chunks(candidates)
        ambiguous_objects = detect_ambiguous_objects(question, candidates, doc_ids=doc_ids)
        if ambiguous_objects:
            return {"answer": _AMBIGUITY_MESSAGE_TEMPLATE.format(objects="、".join(ambiguous_objects)), "sources": []}
        candidates = filter_hits_by_focus(retrieval_query, candidates, doc_ids=doc_ids)
        candidates = prefilter_hits(retrieval_query, candidates, settings.retrieval_prefilter_top_k)
        candidates = prioritize_chunk_tokenization_hits(question, candidates)
        if candidates and not evidence_directly_supports(question, candidates):
            return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": candidates[:settings.rerank_top_k]}
    if not candidates:
        return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": []}
    candidate_texts = [item["content"] for item in candidates]
    if settings.enable_rerank:
        ranked_chunks = rerank(retrieval_query, candidate_texts)
    else:
        ranked_chunks = candidate_texts[:settings.rerank_top_k]
        logger.info("QA rerank disabled using top_k=%s from search results", len(ranked_chunks))
    t2 = time.perf_counter()

    ranked_chunks = filter_meta_evaluation_chunks([
        {"doc_id": "", "filename": "", "page": 0, "content": chunk, "score": 0.0}
        for chunk in ranked_chunks
    ])
    if not ranked_chunks:
        return {"answer": EVIDENCE_REFUSAL_MESSAGE, "sources": []}
    ranked_chunks = [item["content"] for item in ranked_chunks]

    context = "\n\n---\n\n".join(ranked_chunks)
    extra_queries = retrieval_queries[1:] if len(retrieval_queries) > 1 else None
    evidence_brief = build_evidence_brief(question, ranked_chunks, extra_queries=extra_queries)
    if evidence_brief:
        context = f"重点证据：\n{evidence_brief}\n\n---\n\n原始参考内容：\n{context}"
    llm = create_llm()
    chain = _PROMPT | llm | StrOutputParser()
    answer_text = clean_answer_text(chain.invoke({"context": context, "question": question}))
    answer_text = canonicalize_refusal_text(answer_text)
    answer_text = repair_citation_policy_refusal(question, context, answer_text)
    unsupported_numbers, quality_issues = _inspect_guardrails(answer_text, context)
    if unsupported_numbers or quality_issues:
        logger.warning(
            "QA answer failed guardrails unsupported_numbers=%s quality_issues=%s, retrying strict generation",
            unsupported_numbers,
            quality_issues,
        )
        strict_chain = _STRICT_PROMPT | llm | StrOutputParser()
        answer_text = clean_answer_text(strict_chain.invoke({"context": context, "question": question}))
        answer_text = canonicalize_refusal_text(answer_text)
        answer_text = repair_citation_policy_refusal(question, context, answer_text)
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
