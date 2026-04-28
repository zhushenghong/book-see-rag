from __future__ import annotations

import re

from book_see_rag.retrieval import extract_terms


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])\s*")
_NOISE_RE = re.compile(r"^\s*(?:\d+[.、)]\s*)?$")


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
    return hints


def build_evidence_brief(question: str, chunks: list[str], max_sentences: int = 6) -> str:
    terms = [term for term in extract_terms(question) if len(term) >= 2]
    hints = _query_hints(question)
    scored: list[tuple[float, str]] = []

    for chunk in chunks:
        for sentence in _split_sentences(chunk):
            normalized = _normalize(sentence)
            if not normalized:
                continue
            term_hits = sum(1 for term in terms if _normalize(term) in normalized)
            hint_hits = sum(1 for hint in hints if _normalize(hint) in normalized)
            if term_hits == 0 and hint_hits == 0:
                continue
            score = term_hits * 3.0 + hint_hits * 1.5 + min(len(sentence), 120) / 120.0
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
