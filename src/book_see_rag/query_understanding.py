from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from book_see_rag.config import get_settings
from book_see_rag.metadata_store import get_knowledge_base, list_documents
from book_see_rag.retrieval import RankedHit, extract_terms


_OBJECT_RE = re.compile(r"[\u4e00-\u9fff]{2,12}[A-Za-z]{0,10}[A-Za-z0-9 ]{0,12}\d{1,5}[A-Za-z0-9]*\s*(?:20\d{2}款)?")
_HEADING_LINE_RE = re.compile(r"^\s*(?:\d+[.、)]\s*)?([^\n:：]{4,60})\s*$")
_GENERIC_FOCUS_STOPWORDS = {
    "什么", "多少", "哪些", "这个", "那个", "这款", "那款", "参数", "配置", "请问", "是啥",
}
_GENERIC_AMBIGUOUS_QUERY_TERMS = {
    "他", "它", "其", "这个", "那个", "这款", "那款", "这个人", "该人", "上述", "前者", "后者",
}
_GENERIC_OBJECT_NOISE_TERMS = {
    "基本参数", "主要特点", "对比总结", "参数", "配置", "概述", "简介", "总结",
}
_EXPLICIT_QUERY_ANCHORS = {
    "llamaindex",
    "milvus",
    "bm25",
    "rag",
    "reranker",
    "embedding",
    "embeddings",
    "vectorstore",
    "fastapi",
    "redis",
    "celery",
    "streamlit",
    "ocr",
}


@dataclass(frozen=True)
class DomainProfile:
    name: str = "default"
    kb_ids: tuple[str, ...] = ()
    filename_keywords: tuple[str, ...] = ()
    query_synonyms: dict[str, str] = field(default_factory=dict)
    focus_stopwords: set[str] = field(default_factory=set)
    object_noise_terms: set[str] = field(default_factory=set)
    ambiguous_query_terms: set[str] = field(default_factory=set)


DEFAULT_PROFILE = DomainProfile(
    query_synonyms={
        "检索增强生成": "RAG 不是单纯的大模型问答 先检索再回答 先从知识库检索证据 基于证据回答",
        "文档切分": "chunk 不是中文分词 文本片段 长文档 片段",
        "向量化": "embedding 通常对 chunk 分别做 计算向量",
        "重排序器": "reranker 负责对候选证据排序 相关性排序",
        "关键词检索": "BM25 精确匹配",
        "语义检索": "向量检索 语义相似",
        "最小权限": "用户只能看到自己有权访问的知识库 用户只能检索自己有权访问的文档 无权限 doc_id 必须过滤 越权过滤",
        "引用校验": "回答必须可追溯 如果检索不到直接证据 不能编造答案 依据不足",
        "可追溯": "回答必须可追溯 不能编造答案",
        "rag": "RAG 不是单纯的大模型问答 先检索再回答 先从知识库检索证据 基于证据回答",
        "chunk": "chunk 不是中文分词 文本片段 长文档 片段",
        "embedding": "embedding 通常对 chunk 分别做 计算向量",
        "reranker": "reranker 负责对候选证据排序 相关性排序",
        "bm25": "关键词检索 精确匹配",
        "向量检索": "语义检索 语义相似",
        "最小权限模型": "用户只能看到自己有权访问的知识库 用户只能检索自己有权访问的文档 无权限 doc_id 必须过滤 越权过滤",
        "llamaindex": "不是大模型 不是向量数据库 编排层 Document VectorStoreIndex retriever SearchHit",
    },
    focus_stopwords=_GENERIC_FOCUS_STOPWORDS,
    object_noise_terms=_GENERIC_OBJECT_NOISE_TERMS,
    ambiguous_query_terms=_GENERIC_AMBIGUOUS_QUERY_TERMS,
)


_BUILTIN_PROFILES = [
    DEFAULT_PROFILE,
    DomainProfile(
        name="laptop",
        filename_keywords=("联想", "电脑", "laptop"),
        query_synonyms={
            "gpu": "显卡 图形处理器 核显 独显",
            "显卡": "GPU 图形处理器 核显 独显",
            "核显": "集成显卡 显卡 GPU",
            "独显": "独立显卡 显卡 GPU",
            "音响": "扬声器 音效 杜比 Nahimic",
            "音箱": "扬声器 音效 杜比 Nahimic",
            "音效": "扬声器 音响 杜比 Nahimic",
            "键盘": "背光键盘 RGB键盘 键程 全键无冲",
            "重量": "多重 kg 重量",
            "多重": "重量 kg",
            "接口": "USB-C USB-A HDMI 耳机孔 读卡器 RJ45 网口",
        },
        focus_stopwords=_GENERIC_FOCUS_STOPWORDS | {"联想", "电脑", "笔记本", "2024", "2024款"},
        object_noise_terms=_GENERIC_OBJECT_NOISE_TERMS | {
            "16gb", "32gb", "512gb", "1tb", "75wh", "80wh", "100w", "140w",
            "120hz", "240hz", "3050", "4060", "4070", "1080p", "2.5k",
        },
        ambiguous_query_terms=_GENERIC_AMBIGUOUS_QUERY_TERMS | {
            "多重", "重量", "gpu", "显卡", "键盘", "音响", "音效", "接口", "电池", "屏幕", "内存", "存储", "处理器",
        },
    ),
]


def _profile_from_dict(payload: dict) -> DomainProfile:
    return DomainProfile(
        name=str(payload.get("name", "default")),
        kb_ids=tuple(payload.get("kb_ids", []) or []),
        filename_keywords=tuple(payload.get("filename_keywords", []) or []),
        query_synonyms={str(k): str(v) for k, v in (payload.get("query_synonyms") or {}).items()},
        focus_stopwords={str(item).lower() for item in (payload.get("focus_stopwords") or [])},
        object_noise_terms={str(item).lower() for item in (payload.get("object_noise_terms") or [])},
        ambiguous_query_terms={str(item).lower() for item in (payload.get("ambiguous_query_terms") or [])},
    )


@lru_cache(maxsize=1)
def load_profiles() -> tuple[DomainProfile, ...]:
    path = Path(get_settings().query_profiles_path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", payload if isinstance(payload, list) else [])
        if profiles:
            return tuple(_profile_from_dict(item) for item in profiles)
    return tuple(_BUILTIN_PROFILES)


def _resolve_doc_records(doc_ids: list[str] | None) -> list[dict]:
    if not doc_ids:
        return []
    allowed = set(doc_ids)
    return [item for item in list_documents() if item["doc_id"] in allowed]


def _profile_by_name(name: str | None, profiles: tuple[DomainProfile, ...]) -> DomainProfile | None:
    if not name:
        return None
    normalized = name.strip().lower()
    for profile in profiles:
        if profile.name.lower() == normalized:
            return profile
    return None


def _matches_profile(profile: DomainProfile, query: str, hits: list[RankedHit] | None, doc_ids: list[str] | None) -> bool:
    docs = _resolve_doc_records(doc_ids)
    if profile.kb_ids and any(doc.get("kb_id") in profile.kb_ids for doc in docs):
        return True

    filenames = [doc.get("filename", "") for doc in docs]
    if hits:
        filenames.extend(hit.get("filename", "") for hit in hits[:3])
    lowered_filenames = "\n".join(filenames).lower()
    if profile.filename_keywords and any(keyword.lower() in lowered_filenames for keyword in profile.filename_keywords):
        return True

    merged = "\n".join([query, *(hit["content"] for hit in hits[:2])] if hits else [query]).lower()
    if any(term in merged for term in profile.query_synonyms):
        return True
    if any(term in merged for term in profile.ambiguous_query_terms):
        return True
    return False


def choose_profile(query: str, hits: list[RankedHit] | None = None, doc_ids: list[str] | None = None) -> DomainProfile:
    profiles = load_profiles()
    docs = _resolve_doc_records(doc_ids)
    kb_profile_names = []
    for doc in docs:
        kb = get_knowledge_base(doc.get("kb_id", ""))
        if kb and kb.get("query_profile"):
            kb_profile_names.append(str(kb["query_profile"]))
    if kb_profile_names:
        resolved = _profile_by_name(kb_profile_names[0], profiles)
        if resolved:
            return resolved
    for profile in profiles:
        if profile.name == "default":
            continue
        if _matches_profile(profile, query, hits, doc_ids):
            return profile
    return profiles[0] if profiles else DEFAULT_PROFILE


def expand_query_terms(
    query: str,
    profile: DomainProfile | None = None,
    doc_ids: list[str] | None = None,
    include_route_hints: bool = True,
) -> str:
    profile = profile or choose_profile(query, doc_ids=doc_ids)
    expansions: list[str] = []
    lowered = query.lower()
    for term, synonyms in profile.query_synonyms.items():
        if term in lowered:
            expansions.append(synonyms)
    if include_route_hints:
        expansions.extend(_route_hints(query))
    if not expansions:
        return query.strip()

    seen: set[str] = set()
    ordered: list[str] = []
    for item in expansions:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return f"{query.strip()}\n检索关注：{'；'.join(ordered)}"


def build_retrieval_queries(
    query: str,
    profile: DomainProfile | None = None,
    doc_ids: list[str] | None = None,
) -> list[str]:
    profile = profile or choose_profile(query, doc_ids=doc_ids)
    queries: list[str] = [
        expand_query_terms(query, profile=profile, doc_ids=doc_ids, include_route_hints=False)
    ]
    queries.extend(_route_hints(query))

    seen: set[str] = set()
    ordered: list[str] = []
    for item in queries:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _route_hints(query: str) -> list[str]:
    normalized = query.lower()
    hints: list[str] = []

    if "rag" in normalized and any(term in normalized for term in ("区别", "对比", "问答", "大模型")):
        hints.extend([
            "RAG 的关键是先从知识库检索证据",
            "再让大模型基于证据回答",
            "普通大模型只依赖模型内部知识",
        ])
    if "chunk" in normalized and "分词" in normalized:
        hints.extend([
            "Chunk 不是中文分词",
            "chunk 是较大的文本片段",
            "分词通常是把句子切成词或 token",
        ])
    if "embedding" in normalized and any(term in normalized for term in ("整个文件", "chunk", "切分")):
        hints.extend([
            "Embedding 不是对整个文件只做一次",
            "对切分后的 chunk 分别做 embedding",
        ])
    if ("reranker" in normalized or "重排序" in normalized) and any(term in normalized for term in ("向量数据库", "区别", "分别")):
        hints.extend([
            "Reranker 不是向量数据库",
            "向量数据库负责快速召回候选内容",
            "reranker 负责对候选内容进行更精细的相关性排序",
        ])
    if "最小权限" in normalized or "无权限" in normalized or "doc_id" in normalized:
        hints.extend([
            "系统采用最小权限模型",
            "用户只能看到自己有权访问的知识库",
            "用户只能检索自己有权访问的文档",
        ])
    if any(term in normalized for term in ("引用校验", "可追溯", "依据不足")):
        hints.extend([
            "系统要求回答必须可追溯",
            "而不是编造答案",
        ])
    if "llamaindex" in normalized and any(term in normalized for term in ("做了", "定位", "替代", "不是大模型", "没有替代")):
        hints.extend([
            "LlamaIndex 在星澜知识助手中不是大模型",
            "将系统已有的 chunk 包装成 LlamaIndex Document",
            "构建临时 VectorStoreIndex",
            "使用 LlamaIndex retriever 从指定文档范围内召回相关 node",
            "将 node 转回系统统一的 SearchHit",
            "LlamaIndex 没有替代文档解析",
        ])
    if "bm25" in normalized and any(term in normalized for term in ("向量检索", "reranker", "比较", "区别")):
        hints.extend([
            "BM25 是一种经典关键词检索算法",
            "向量检索是一种语义检索方式",
            "BM25 和向量检索通常不是替代关系",
            "互补关系",
            "推荐做法是混合检索",
            "reranker 负责对候选内容进行更精细的相关性排序",
        ])
    return hints


def _focus_terms(query: str, profile: DomainProfile) -> list[str]:
    candidates = [
        term for term in extract_terms(query)
        if len(term) >= 3 and term not in profile.focus_stopwords and not term.isdigit()
    ]
    candidates.sort(key=len, reverse=True)
    return candidates[:4]


def _is_abstract_explanation_query(query: str) -> bool:
    normalized = query.lower()
    technical_terms = ("rag", "chunk", "embedding", "reranker", "bm25", "向量检索", "llamaindex")
    abstract_markers = ("为什么", "区别", "对比", "比较", "是什么", "什么是", "是否", "分别", "作用")
    return any(term in normalized for term in technical_terms) and any(marker in normalized for marker in abstract_markers)


def filter_hits_by_focus(
    query: str,
    hits: list[RankedHit],
    profile: DomainProfile | None = None,
    doc_ids: list[str] | None = None,
) -> list[RankedHit]:
    if not hits:
        return []

    profile = profile or choose_profile(query, hits, doc_ids=doc_ids)
    focus_terms = _focus_terms(query, profile)
    if not focus_terms:
        return hits

    matched = [
        hit for hit in hits
        if any(term.lower() in hit["content"].lower() for term in focus_terms)
    ]
    if _is_abstract_explanation_query(query) and matched:
        matched_ids = {id(hit) for hit in matched}
        unmatched = [hit for hit in hits if id(hit) not in matched_ids]
        return matched + unmatched
    return matched or hits


def _normalize_candidate(text: str) -> str:
    return " ".join(text.split()).strip("：:()（）[]【】,.，。;；")


def _looks_like_subject(candidate: str, profile: DomainProfile) -> bool:
    lowered = candidate.lower()
    if not candidate or len(candidate) < 4:
        return False
    if lowered in profile.object_noise_terms:
        return False
    if any(unit in lowered for unit in ("gb", "tb", "wh", "hz", "nit", "mm", "kg", "w")):
        return False
    has_ascii_letter = any(ch.isascii() and ch.isalpha() for ch in candidate)
    has_digit = any(ch.isdigit() for ch in candidate)
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in candidate)
    return candidate.endswith("款") or (has_cjk and has_digit) or (has_ascii_letter and has_digit)


def _extract_heading_candidates(text: str, profile: DomainProfile) -> list[str]:
    candidates: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) > 60 or "：" in line or ":" in line:
            continue
        matched = _HEADING_LINE_RE.match(line)
        if not matched:
            continue
        candidate = _normalize_candidate(matched.group(1))
        if _looks_like_subject(candidate, profile):
            candidates.append(candidate)
    return candidates


def extract_explicit_objects(text: str, profile: DomainProfile | None = None, doc_ids: list[str] | None = None) -> list[str]:
    if not text:
        return []

    profile = profile or choose_profile(text, doc_ids=doc_ids)
    objects: list[str] = []
    seen: set[str] = set()

    for match in _OBJECT_RE.finditer(text):
        raw = match.group(0)
        end = match.end()
        if end < len(text) and text[end] == "款" and not raw.endswith("款"):
            raw = f"{raw}款"
        candidate = _normalize_candidate(raw)
        if _looks_like_subject(candidate, profile) and candidate not in seen:
            seen.add(candidate)
            objects.append(candidate)

    for candidate in _extract_heading_candidates(text, profile):
        if candidate not in seen:
            seen.add(candidate)
            objects.append(candidate)

    return objects


def is_underspecified_query(query: str, profile: DomainProfile | None = None, doc_ids: list[str] | None = None) -> bool:
    profile = profile or choose_profile(query, doc_ids=doc_ids)
    normalized = query.strip().lower()
    if extract_explicit_objects(query, profile=profile, doc_ids=doc_ids):
        return False
    if any(term in normalized for term in _EXPLICIT_QUERY_ANCHORS):
        return False
    return any(term in normalized for term in profile.ambiguous_query_terms)


def detect_ambiguous_objects(
    query: str,
    hits: list[RankedHit],
    limit: int = 5,
    profile: DomainProfile | None = None,
    doc_ids: list[str] | None = None,
) -> list[str]:
    profile = profile or choose_profile(query, hits, doc_ids=doc_ids)
    if not is_underspecified_query(query, profile=profile, doc_ids=doc_ids):
        return []

    objects: list[str] = []
    seen: set[str] = set()
    for hit in hits[:limit]:
        for obj in extract_explicit_objects(hit["content"], profile=profile, doc_ids=doc_ids):
            if obj not in seen:
                seen.add(obj)
                objects.append(obj)
            if len(objects) >= 3:
                return objects
    return objects if len(objects) >= 2 else []
