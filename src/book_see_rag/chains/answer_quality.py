from __future__ import annotations

import re


_PROMPT_LEAK_RE = re.compile(r"(^|\n|\s)(user|assistant|system)\s*[:：]?", re.IGNORECASE)
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
_ALLOWED_LATIN_WORDS = {
    "rag",
    "bm25",
    "pdf",
    "docx",
    "txt",
    "markdown",
    "milvus",
    "redis",
    "fastapi",
    "streamlit",
    "celery",
    "llamaindex",
    "document",
    "vectorstoreindex",
    "retriever",
    "node",
    "searchhit",
    "embedding",
    "reranker",
    "chunk",
    "chunk_size",
    "chunk_overlap",
    "ocr",
    "hr",
    "rd",
    "api",
    "id",
    "mb",
    "gb",
    "kb",
}
_REPEATED_CJK_RE = re.compile(r"([\u4e00-\u9fff]{1,3})\1{2,}")
_BROKEN_FORMAT_RE = re.compile(r"(^|\n)\s*[-*]\s*$|\\\s*$")


def inspect_answer_quality(answer: str) -> list[str]:
    issues: list[str] = []
    compact = answer.strip()
    if not compact:
        return ["empty_answer"]

    if _PROMPT_LEAK_RE.search(compact):
        issues.append("contains_prompt_leak")

    suspicious = []
    for word in _LATIN_RE.findall(compact):
        normalized = word.lower()
        if normalized not in _ALLOWED_LATIN_WORDS:
            suspicious.append(word)
    if suspicious:
        issues.append("contains_garbled_english")

    if _REPEATED_CJK_RE.search(compact):
        issues.append("contains_repeated_terms")

    if _BROKEN_FORMAT_RE.search(compact):
        issues.append("contains_broken_format")

    useful_chars = sum(1 for ch in compact if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    symbol_ratio = 1 - useful_chars / max(len(compact), 1)
    if symbol_ratio > 0.45:
        issues.append("too_many_symbols")

    return issues
