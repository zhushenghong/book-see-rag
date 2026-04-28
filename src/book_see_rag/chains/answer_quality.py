from __future__ import annotations

import re


_PROMPT_LEAK_RE = re.compile(r"(^|\n|\s)(user|assistant|system)\s*[:：]?", re.IGNORECASE)
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")
_MODEL_RE = re.compile(
    r"(?:"
    r"[A-Za-z]{1,10}\d{1,8}[A-Za-z0-9_.+-]{0,16}"
    r"|"
    r"[A-Za-z]{1,10}[\s-]{1,3}\d{2,8}(?:\.\d{1,2})?[A-Za-z0-9_.+-]{0,16}"
    r")"
)

# Only keep stable system/protocol words here.
# Domain/professional terms should be validated against evidence instead of growing this list.
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
    "ocr",
    "api",
    "id",
    "http",
    "https",
    "json",
    "uuid",
    "chunk",
    "chunks",
    "chunk_size",
    "chunk_overlap",
    "embedding",
    "embeddings",
    "vector",
    "vectors",
    "vectorstore",
    "retrieval",
    "retriever",
    "reranker",
    "index",
    "document",
    "documents",
    "query",
    "context",
    "source",
    "sources",
    "answer",
    "prompt",
    "model",
    "token",
    "tokens",
    "semantic",
    "keyword",
    "keywords",
    "similarity",
    "search",
    "support",
    "system",
    "technical",
    "tokenization",
}
_REPEATED_CJK_RE = re.compile(r"([\u4e00-\u9fff]{1,3})\1{2,}")
_BROKEN_FORMAT_RE = re.compile(r"(^|\n)\s*[-*]\s*$|\\\s*$")
_MIXED_TECH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*\d+[A-Za-z0-9_.+-]*")
_UPPER_TECH_TOKEN_RE = re.compile(r"[A-Z]{2,}[A-Za-z0-9_.+-]*")

_ISSUE_UNSUPPORTED_TERMS = "contains_garbled_english"


def _compact_for_match(text: str) -> str:
    """
    Compact text for evidence-alignment matching.

    Keep only alphanumeric characters so that:
    - "RTX 4060" can match "RTX4060"
    - "Qwen2.5" can match "Qwen 2.5"
    """
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _extract_terms(answer: str) -> list[str]:
    """
    Extract Latin/pro tokens from answer.

    - Word-like: length>=3 Latin (including accented)
    - Model-like: letter+digits such as i7, RTX4060, RTX 4060, Qwen2.5
    """
    terms = list(_LATIN_RE.findall(answer))
    terms.extend(match.group(0) for match in _MODEL_RE.finditer(answer))
    # keep order while deduping
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def inspect_answer_quality(answer: str, evidence_text: str | None = None) -> list[str]:
    issues: list[str] = []
    compact = answer.strip()
    if not compact:
        return ["empty_answer"]

    if _PROMPT_LEAK_RE.search(compact):
        issues.append("contains_prompt_leak")

    evidence_compact = _compact_for_match(evidence_text or "")
    suspicious: list[str] = []
    for term in _extract_terms(compact):
        normalized = term.lower()
        if normalized in _ALLOWED_LATIN_WORDS:
            continue
        if _MIXED_TECH_TOKEN_RE.fullmatch(term) or _UPPER_TECH_TOKEN_RE.fullmatch(term):
            continue
        if evidence_compact:
            term_compact = _compact_for_match(term)
            if term_compact and term_compact in evidence_compact:
                continue
            # Also allow partial evidence alignment for tokens such as
            # "Windows 11", "RTX 3050", "LPDDR5", "Wi-Fi 6E".
            alpha = "".join(ch.lower() for ch in term if ch.isalpha())
            digits = "".join(ch for ch in term if ch.isdigit())
            if alpha and alpha in evidence_compact and (not digits or digits in evidence_compact):
                continue
        # Long natural-language English fragments are suspicious; short model/spec tokens are not.
        letters_only = "".join(ch for ch in term if ch.isalpha())
        if len(letters_only) >= 5 and not term.isupper():
            suspicious.append(term)
    if suspicious:
        issues.append(_ISSUE_UNSUPPORTED_TERMS)

    if _REPEATED_CJK_RE.search(compact):
        issues.append("contains_repeated_terms")

    if _BROKEN_FORMAT_RE.search(compact):
        issues.append("contains_broken_format")

    useful_chars = sum(1 for ch in compact if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    symbol_ratio = 1 - useful_chars / max(len(compact), 1)
    if symbol_ratio > 0.45:
        issues.append("too_many_symbols")

    return issues
