import re
from typing import TypedDict


class RankedHit(TypedDict):
    doc_id: str
    filename: str
    page: int
    content: str
    score: float


_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]{2,}|[\u4e00-\u9fff]{2,}")
_META_SECTION_MARKERS = ("推荐测试问题", "标准答案要点")


def is_meta_evaluation_chunk(content: str) -> bool:
    return any(marker in content for marker in _META_SECTION_MARKERS)


def filter_meta_evaluation_chunks(hits: list[RankedHit]) -> list[RankedHit]:
    filtered = [hit for hit in hits if not is_meta_evaluation_chunk(hit["content"])]
    return filtered or hits


def _extract_terms(query: str) -> list[str]:
    base_terms = [term.lower() for term in _TOKEN_RE.findall(query)]
    expanded: list[str] = []
    for term in base_terms:
        expanded.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", term):
            expanded.extend(term[i:i + 2] for i in range(len(term) - 1))
    # keep order while deduping
    seen: set[str] = set()
    ordered: list[str] = []
    for term in expanded:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered


def prefilter_hits(query: str, hits: list[RankedHit], limit: int) -> list[RankedHit]:
    if not hits:
        return []

    terms = _extract_terms(query)
    if not terms:
        return hits[:limit]

    ranked: list[tuple[float, RankedHit]] = []
    for idx, hit in enumerate(hits):
        content = hit["content"].lower()
        filename = hit["filename"].lower()
        lexical = 0.0
        for term in terms:
            if term in content:
                lexical += min(len(term), 8) * 0.15
            if term in filename:
                lexical += 0.2
        dense = hit.get("score", 0.0)
        # reward early ANN hits while still letting lexical matches move up
        rank_bonus = max(0.0, 1.0 - idx * 0.03)
        meta_penalty = 8.0 if is_meta_evaluation_chunk(hit["content"]) else 0.0
        ranked.append((lexical + dense * 0.6 + rank_bonus - meta_penalty, hit))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]
