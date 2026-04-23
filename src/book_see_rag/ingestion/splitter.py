from langchain_core.documents import Document
import logging
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from book_see_rag.config import get_settings

logger = logging.getLogger(__name__)


_REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}")
_ALNUM_CJK_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_NOISE_RUN_RE = re.compile(r"([A-Za-z])(?:\s*\1){5,}")


def _is_noisy_line(text: str) -> bool:
    stripped = " ".join(text.split()).strip()
    if not stripped:
        return True
    if _REPEATED_CHAR_RE.search(stripped):
        return True
    if _NOISE_RUN_RE.search(stripped):
        return True
    useful_chars = sum(1 for ch in stripped if _ALNUM_CJK_RE.match(ch))
    return useful_chars / max(len(stripped), 1) < 0.4


def normalize_chunk_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        cleaned = " ".join(raw.split()).strip()
        if cleaned and not _is_noisy_line(cleaned):
            lines.append(cleaned)
    return "\n".join(lines).strip()


def is_noisy_chunk(text: str) -> bool:
    settings = get_settings()
    normalized = normalize_chunk_text(text)
    if not normalized:
        return True
    useful_chars = sum(1 for ch in normalized if _ALNUM_CJK_RE.match(ch))
    if useful_chars < max(8, settings.min_clean_chunk_chars // 2):
        return True
    if _REPEATED_CHAR_RE.search(normalized):
        return True
    if _NOISE_RUN_RE.search(normalized):
        return True

    noise_ratio = 1 - (useful_chars / max(len(normalized), 1))
    if noise_ratio > settings.noisy_char_ratio_threshold:
        return True

    return False


def clean_chunks(chunks: list[Document]) -> list[Document]:
    cleaned: list[Document] = []
    dropped = 0
    for chunk in chunks:
        content = normalize_chunk_text(chunk.page_content)
        if is_noisy_chunk(content):
            dropped += 1
            continue
        cleaned.append(Document(page_content=content, metadata=chunk.metadata))
    logger.info("Chunk cleanup finished kept=%s dropped=%s", len(cleaned), dropped)
    return cleaned


def split_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    logger.info(
        "Splitting documents count=%s chunk_size=%s overlap=%s",
        len(documents),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    cleaned = clean_chunks(chunks)
    logger.info("Split complete raw_chunks=%s clean_chunks=%s", len(chunks), len(cleaned))
    return cleaned
