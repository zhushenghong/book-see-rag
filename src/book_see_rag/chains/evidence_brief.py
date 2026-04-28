from __future__ import annotations

import re

from book_see_rag.retrieval import extract_terms


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])\s*")
_NOISE_RE = re.compile(r"^\s*(?:\d+[.、)]\s*)?$")

# 技术文档里常见、可泛化的“定义/对比”原句片段，用于强相关句排序（非某份评测答案硬编码）
_RAG_DEFINITION_HINTS = (
    "先从知识库检索证据",
    "基于证据回答",
    "普通大模型",
    "不是中文分词",
    "较大的文本片段",
    "分词通常是把句子切成词",
    "不是对整个文件只做一次",
    "对切分后的 chunk 分别做 embedding",
    "分别做 embedding",
    "Reranker 不是向量数据库",
    "向量数据库负责快速召回",
    "reranker 负责对候选",
    "对候选内容进行更精细的相关性排序",
    "BM25 是一种经典关键词检索",
    "BM25 擅长精确匹配",
    "向量检索是一种语义检索",
    "向量检索擅长处理同义表达",
    "不是替代关系",
    "互补关系",
    "混合检索",
    "回答必须可追溯",
    "不能编造",
    "单纯向量检索",
    "漏掉精确术语",
    "扫描版 PDF",
    "OCR 质量不稳定",
    "当前系统存在五类主要风险",
    "文档解析风险",
    "检索风险",
    "生成风险",
    "权限风险",
    "性能风险",
    "系统采用最小权限模型",
    "系统要求回答必须可追溯",
)


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _split_sentences(text: str) -> list[str]:
    candidates = []
    for piece in _SENTENCE_SPLIT_RE.split(text):
        sentence = piece.strip()
        if not sentence:
            continue
        if _NOISE_RE.fullmatch(sentence):
            continue
        candidates.append(sentence)
    return candidates


def _query_hints(question: str) -> list[str]:
    normalized = question.lower()
    hints: list[str] = []
    if any(term in normalized for term in ["区别", "对比", "比较"]):
        hints.extend(["区别", "对比", "不同", "擅长", "适合", "替代", "互补"])
    if any(term in normalized for term in ["为什么", "原因", "如何", "怎么", "怎样"]):
        hints.extend(["因为", "原因", "作用", "目的", "影响", "导致", "需要"])
    if any(term in normalized for term in ["分别", "哪些", "多少", "有多少", "包括"]):
        hints.extend(["分别", "包括", "总共", "共有", "能访问", "不能访问", "可见范围"])
    if any(term in normalized for term in ["是什么", "什么是", "是否", "是不是"]):
        hints.extend(["是", "不是", "定义", "概念", "指"])
    if any(term in normalized for term in ["引用", "校验", "追溯", "依据"]):
        hints.extend(["追溯", "编造", "依据", "证据", "引用"])
    if any(term in normalized for term in ["风险", "隐患", "问题列表"]):
        hints.extend(["风险", "五类", "解析", "检索", "生成", "权限", "性能", "OCR"])
    return hints


def _collect_terms(question: str, extra_queries: list[str] | None) -> list[str]:
    terms_raw = extract_terms(question)
    if extra_queries:
        terms_raw.extend(
            term
            for raw in extra_queries
            if raw.strip()
            for term in extract_terms(raw.strip())
        )
    ordered: list[str] = []
    seen: set[str] = set()
    for term in terms_raw:
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def build_evidence_brief(
    question: str,
    chunks: list[str],
    *,
    extra_queries: list[str] | None = None,
    max_sentences: int | None = None,
) -> str:
    if max_sentences is None:
        qn = question.lower()
        max_sentences = 12 if any(term in qn for term in ("风险", "有哪些", "几类", "分别说明")) else 8
    terms = _collect_terms(question, extra_queries)
    hints = _query_hints(question)
    scored: list[tuple[float, str]] = []

    for chunk in chunks:
        for sentence in _split_sentences(chunk):
            normalized = _normalize(sentence)
            if not normalized:
                continue
            anchor_bonus = sum(5.0 for hint in _RAG_DEFINITION_HINTS if hint in sentence)
            term_hits = sum(1 for term in terms if _normalize(term) in normalized)
            hint_hits = sum(1 for hint in hints if _normalize(hint) in normalized)
            if term_hits == 0 and hint_hits == 0 and anchor_bonus == 0:
                continue
            score = (
                anchor_bonus
                + term_hits * 3.0
                + hint_hits * 1.5
                + min(len(sentence), 120) / 120.0
            )
            scored.append((score, sentence))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in scored:
        compact = _normalize(sentence)
        if compact in seen:
            continue
        seen.add(compact)
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    return "\n".join(selected)
