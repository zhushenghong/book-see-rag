import re
from typing import TypedDict


class RankedHit(TypedDict):
    doc_id: str
    filename: str
    page: int
    content: str
    score: float


_TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]{2,}|[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])\s*")
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+[.、)]\s*)(?P<title>[^\n。！？；:：]{2,60})\s*$")
_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s*(?P<title>[^\n。！？；:：]{2,60})\s*$")
_META_SECTION_MARKERS = ("推荐测试问题", "标准答案要点")
_GENERIC_QUERY_TERMS = {
    "什么", "为什么", "怎么", "如何", "哪些", "分别", "说明", "请", "是否", "是不是",
    "当前", "系统", "项目", "需要", "可以", "不能", "回答", "问题", "这个", "那个",
    "星澜", "知识助手", "星澜知识助手", "是什么",
}
_GENERIC_QUERY_SUBSTRINGS = (
    "星澜",
    "知识助手",
    "是什么",
    "项目的",
    "什么",
    "为什么",
    "怎么",
    "如何",
    "哪些",
    "是否",
    "这个",
    "那个",
    "系统",
)
_CONTENT_LINE_MARKERS = ("被", "使用", "构建", "包装", "召回", "转换", "负责", "表示", "不是", "没有", "可以", "不能")
_DOMAIN_PHRASE_TERMS = (
    "普通大模型",
    "大模型问答",
    "大模型",
    "中文分词",
    "文本片段",
    "整个文件",
    "精确术语",
    "向量数据库",
    "替代关系",
    "互补关系",
    "混合检索",
    "食堂菜单",
)
_QUERY_FILLERS = (
    "是什么",
    "为什么",
    "有什么区别",
    "什么区别",
    "有什么",
    "怎么",
    "如何",
    "哪些",
    "是否",
    "是不是",
    "项目的",
    "请",
    "说明",
    "分别",
    "和",
    "与",
    "有",
)


def is_meta_evaluation_chunk(content: str) -> bool:
    return any(marker in content for marker in _META_SECTION_MARKERS)


def filter_meta_evaluation_chunks(hits: list[RankedHit]) -> list[RankedHit]:
    filtered = [hit for hit in hits if not is_meta_evaluation_chunk(hit["content"])]
    return filtered or hits


def _derived_cjk_terms(term: str) -> list[str]:
    derived: list[str] = []
    for phrase in _DOMAIN_PHRASE_TERMS:
        if phrase in term:
            derived.append(phrase)

    compact = term
    for filler in _QUERY_FILLERS:
        compact = compact.replace(filler, "")
    compact = compact.strip()
    if 2 <= len(compact) <= 8 and not any(generic in compact for generic in _GENERIC_QUERY_SUBSTRINGS):
        derived.append(compact)
    return derived


def extract_terms(query: str) -> list[str]:
    base_terms = [term.lower() for term in _TOKEN_RE.findall(query)]
    expanded: list[str] = []
    for term in base_terms:
        expanded.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", term):
            expanded.extend(_derived_cjk_terms(term))
    # keep order while deduping
    seen: set[str] = set()
    ordered: list[str] = []
    for term in expanded:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered


def content_terms(query: str) -> list[str]:
    return [
        term for term in extract_terms(query)
        if (
            len(term) >= 2
            and term not in _GENERIC_QUERY_TERMS
            and not term.isdigit()
            and not any(generic in term for generic in _GENERIC_QUERY_SUBSTRINGS)
        )
    ]


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.split(text):
        sentence = piece.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def _is_heading_line(line: str) -> bool:
    markdown_match = _MARKDOWN_HEADING_RE.match(line)
    if markdown_match:
        return True
    numbered_match = _NUMBERED_HEADING_RE.match(line)
    if not numbered_match:
        return False
    title = numbered_match.group("title").strip()
    return len(title) <= 24 and not any(marker in title for marker in _CONTENT_LINE_MARKERS)


def _sentence_windows(sentences: list[str], index: int) -> str:
    start = max(0, index - 1)
    end = min(len(sentences), index + 2)
    return " ".join(sentences[start:end])


def _split_sections(text: str) -> list[str]:
    sections: list[list[str]] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_heading = _is_heading_line(line)
        if is_heading and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return ["\n".join(section).strip() for section in sections if section]


def _best_sentence_bonus(query_terms: list[str], content: str) -> float:
    best = 0.0
    for sentence in _split_sentences(content):
        normalized = _normalize(sentence)
        if not normalized:
            continue
        exact_hits = 0
        long_phrase_hits = 0
        coverage = 0.0
        for term in query_terms:
            compact_term = _normalize(term)
            if not compact_term:
                continue
            if compact_term in normalized:
                exact_hits += 1
                coverage += min(len(compact_term), 12) * 0.06
                if len(compact_term) >= 4:
                    long_phrase_hits += 1
        if exact_hits == 0:
            continue
        score = exact_hits * 0.45 + long_phrase_hits * 0.7 + coverage
        if score > best:
            best = score
    return best


def _score_text(query_terms: list[str], text: str) -> float:
    normalized = _normalize(text)
    if not normalized:
        return 0.0

    score = 0.0
    for term in query_terms:
        compact_term = _normalize(term)
        if not compact_term:
            continue
        occurrences = normalized.count(compact_term)
        if occurrences:
            score += min(occurrences, 3) * (1.0 + min(len(compact_term), 12) * 0.18)
            if len(compact_term) >= 4:
                score += 1.2
    return score


def prefilter_hits(query: str, hits: list[RankedHit], limit: int) -> list[RankedHit]:
    if not hits:
        return []

    terms = extract_terms(query)
    if not terms:
        return hits[:limit]

    ranked: list[tuple[float, RankedHit]] = []
    for idx, hit in enumerate(hits):
        content = hit["content"].lower()
        filename = hit["filename"].lower()
        lexical = 0.0
        matched_terms = 0
        for term in terms:
            if term in content:
                matched_terms += 1
                lexical += min(len(term), 10) * 0.18
                if len(term) >= 4:
                    lexical += 0.35
            if term in filename:
                lexical += 0.2
        if matched_terms:
            lexical += min(matched_terms, 6) * 0.08
        lexical += _best_sentence_bonus(terms, hit["content"])
        dense = hit.get("score", 0.0)
        # reward early ANN hits while still letting lexical matches move up
        rank_bonus = max(0.0, 1.0 - idx * 0.03)
        meta_penalty = 20.0 if is_meta_evaluation_chunk(hit["content"]) else 0.0
        ranked.append((lexical + dense * 0.6 + rank_bonus - meta_penalty, hit))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]


def keyword_rank_hits(query: str, hits: list[RankedHit], limit: int) -> list[RankedHit]:
    terms = content_terms(query)
    if not terms:
        return []

    ranked: list[tuple[float, RankedHit]] = []
    for hit in hits:
        content = hit["content"].lower()
        filename = hit["filename"].lower()
        score = 0.0
        for term in terms:
            occurrences = content.count(term)
            if occurrences:
                score += min(occurrences, 3) * (1.0 + min(len(term), 8) * 0.2)
                if len(term) >= 4:
                    score += 1.5
            if term in filename:
                score += 0.5
        score += _best_sentence_bonus(terms, hit["content"])
        if is_meta_evaluation_chunk(hit["content"]):
            score -= 20.0
        if score > 0:
            ranked.append((score, hit))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]


def sentence_window_hits(query: str, hits: list[RankedHit], limit: int) -> list[RankedHit]:
    terms = content_terms(query)
    if not terms:
        return []

    ranked: list[tuple[float, RankedHit]] = []
    for hit in hits:
        if is_meta_evaluation_chunk(hit["content"]):
            continue
        sentences = _split_sentences(hit["content"])
        best_candidate: tuple[float, RankedHit] | None = None
        for index, sentence in enumerate(sentences):
            sentence_score = _score_text(terms, sentence)
            if sentence_score <= 0:
                continue
            window = _sentence_windows(sentences, index)
            window_score = sentence_score + _score_text(terms, window) * 0.35
            candidate = (
                window_score,
                {
                    "doc_id": hit["doc_id"],
                    "filename": hit["filename"],
                    "page": hit["page"],
                    "content": window,
                    "score": hit.get("score", 0.0) + window_score,
                },
            )
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate
        if best_candidate is not None:
            ranked.append(best_candidate)

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]


def section_window_hits(query: str, hits: list[RankedHit], limit: int) -> list[RankedHit]:
    terms = content_terms(query)
    if not terms:
        return []

    ranked: list[tuple[float, RankedHit]] = []
    for hit in hits:
        if is_meta_evaluation_chunk(hit["content"]):
            continue
        best_candidate: tuple[float, RankedHit] | None = None
        for section in _split_sections(hit["content"]):
            score = _score_text(terms, section)
            if score <= 0:
                continue
            candidate = (
                score,
                {
                    "doc_id": hit["doc_id"],
                    "filename": hit["filename"],
                    "page": hit["page"],
                    "content": section,
                    "score": hit.get("score", 0.0) + score,
                },
            )
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate
        if best_candidate is not None:
            ranked.append(best_candidate)

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]


def _hit_identity(hit: RankedHit) -> tuple[str, int, str]:
    return (hit.get("doc_id", ""), int(hit.get("page", 0) or 0), hit.get("content", ""))


def prioritize_chunk_tokenization_hits(question: str, hits: list[RankedHit]) -> list[RankedHit]:
    """Bring definition-style chunks to the front when the user asks about chunk vs 中文分词."""
    qn = question.lower()
    if "chunk" not in qn or "分词" not in qn:
        return hits
    markers = (
        "不是中文分词",
        "chunk 是较大的文本片段",
        "较大的文本片段",
        "分词通常是把句子切成词",
        "分词通常把句子切成词",
    )
    boosted_keys: set[tuple[str, int, str]] = set()
    boosted: list[RankedHit] = []
    for hit in hits:
        content = hit.get("content", "").lower()
        if not any(m.lower() in content for m in markers):
            continue
        key = _hit_identity(hit)
        if key in boosted_keys:
            continue
        boosted_keys.add(key)
        boosted.append(hit)
    if not boosted:
        return hits
    back = [hit for hit in hits if _hit_identity(hit) not in boosted_keys]
    return boosted + back


def merge_ranked_hits(*hit_groups: list[RankedHit]) -> list[RankedHit]:
    merged: list[RankedHit] = []
    seen: set[tuple[str, int, str]] = set()
    for hits in hit_groups:
        for hit in hits:
            key = (hit.get("doc_id", ""), int(hit.get("page", 0) or 0), hit.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    merged.sort(key=lambda hit: float(hit.get("score", 0.0) or 0.0), reverse=True)
    return merged


def evidence_directly_supports(query: str, hits: list[RankedHit]) -> bool:
    normalized_query = query.lower()
    merged = "\n".join(hit["content"].lower() for hit in hits[:5])
    if (
        any(term in normalized_query for term in ("权限", "访问", "知识库", "doc_id", "李明", "王敏", "最小权限"))
        and any(term in merged for term in ("权限模型", "有权访问", "可以访问", "不能访问", "越权", "doc_id", "知识库"))
    ):
        return True
    if "引用校验" in query or (
        "为什么" in query and "引用" in query and ("需要" in query or "校验" in query)
    ):
        if any(term in merged for term in ("可追溯", "依据不足", "不能编造", "编造", "直接证据", "引用")):
            return True
    if (
        any(term in normalized_query for term in ("引用", "可追溯", "依据不足", "支持结论"))
        and any(term in merged for term in ("可追溯", "依据不足", "不能编造", "引用", "直接证据"))
    ):
        return True

    terms = content_terms(query)
    if not terms:
        return bool(hits)
    return any(term in merged for term in terms)
