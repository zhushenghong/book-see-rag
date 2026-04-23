from __future__ import annotations

import re


_BAD_CHARS_RE = re.compile(r"[\uFFFD�]")
_SPACES_RE = re.compile(r"[ \t]{2,}")
_COMMA_RE = re.compile(r"[，,]{2,}")
_PUNCT_SPACE_RE = re.compile(r"\s+([，。；：、！？）])")
_OPEN_SPACE_RE = re.compile(r"([（])\s+")
_DUP_CLOSING_PAREN_RE = re.compile(r"([）)]){2,}")
_DUP_PERCENT_RE = re.compile(r"%{2,}")
_BROKEN_WORDS = {
    "L1mmaIndex": "LlamaIndex",
    "LmmaIndex": "LlamaIndex",
    "MIlvus": "Milvus",
    "embeddingding": "embedding",
    "BM2对于": "BM25",
    "chunksize": "chunk_size",
    "chunk size": "chunk_size",
    "chunkoverlap": "chunk_overlap",
    "chunk overlap": "chunk_overlap",
    "提提供": "提供",
    "迪过大": "过大",
    "发研知识库": "研发知识库",
}


def clean_answer_text(text: object) -> str:
    if not isinstance(text, str):
        content = getattr(text, "content", None)
        text = content if isinstance(content, str) else str(text)
    cleaned = _BAD_CHARS_RE.sub("", text)
    cleaned = cleaned.replace("`", "")
    for source, target in _BROKEN_WORDS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = _SPACES_RE.sub(" ", cleaned)
    cleaned = _COMMA_RE.sub("，", cleaned)
    cleaned = _PUNCT_SPACE_RE.sub(r"\1", cleaned)
    cleaned = _OPEN_SPACE_RE.sub(r"\1", cleaned)
    cleaned = _DUP_CLOSING_PAREN_RE.sub(lambda m: m.group(0)[0], cleaned)
    cleaned = _DUP_PERCENT_RE.sub("%", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
