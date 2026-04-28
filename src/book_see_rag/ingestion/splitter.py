from langchain_core.documents import Document
import logging
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from book_see_rag.config import get_settings

logger = logging.getLogger(__name__)


_REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}")
_ALNUM_CJK_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_NOISE_RUN_RE = re.compile(r"([A-Za-z])(?:\s*\1){5,}")

# 报价单行检测：产品名关键词 + 配置关键词 + 价格分隔符 + 价格
_PRODUCT_KEYWORDS = (
    "刃|拯救者|Y7000|Y9000|GeekPro|天逸|Yoga|Legion|小新|Lecoo|酷310|联想|拯救者刃"
)
_CONFIG_KEYWORDS = "RTX|RX|Ultra|酷睿|i[3579]|[A-Z]\\d{4}|U\\d{4,5}"
# 报价单行特征：产品名 + 配置 + 分隔符(----或----) + 价格
_PRICING_LINE_RE = re.compile(
    rf"^[\s]*(?:{_PRODUCT_KEYWORDS}).*(?:{_CONFIG_KEYWORDS}).*?----.*$",
    re.IGNORECASE,
)
_PRICE_NUM_RE = re.compile(r"\d{4,5}含税|\d{4,5}元|\d+\.\d+万|\d+万")


def is_pricing_line(text: str) -> bool:
    """检测报价单风格的行，应作为原子单元不被切开。

    特征：
    - 包含产品名关键词（刃/拯救者/Y7000/GeekPro/天逸/Yoga/小新/联想等）
    - 包含配置关键词（RTX/RX/Ultra/酷睿/i7/U9200等）
    - 包含价格信息（4-5位数字+含税/元/万）
    - 包含 ---- 分隔符
    """
    stripped = text.strip()
    if not stripped or len(stripped) < 15 or len(stripped) > 600:
        return False
    if "----" not in stripped:
        return False
    # 产品名检查：至少包含一个产品关键词
    has_product_keyword = any(
        kw in stripped for kw in [
            "刃", "拯救者", "Y7000", "Y9000", "GeekPro", "天逸",
            "Yoga", "Legion", "小新", "Lecoo", "酷310", "联想"
        ]
    )
    # 配置关键词检查
    has_config_keyword = bool(_CONFIG_KEYWORDS and re.search(_CONFIG_KEYWORDS, stripped, re.IGNORECASE))
    # 价格检查
    has_price = bool(_PRICE_NUM_RE.search(stripped))
    return has_product_keyword and has_config_keyword and has_price


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
    """清理噪声 chunk。报价单行使用原始内容不过滤，非报价单行走 normalize + 噪声过滤。"""
    cleaned: list[Document] = []
    dropped = 0
    for chunk in chunks:
        is_pricing = is_pricing_line(chunk.page_content)
        if is_pricing:
            # 报价单行：直接保留原始内容，不过 normalize（避免 ---- 被误判为噪声）
            cleaned.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
        else:
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

    # 第一步：检测报价单行，保留为原子 chunk
    all_chunks: list[Document] = []
    non_pricing_docs: list[Document] = []

    for doc in documents:
        lines = doc.page_content.split("\n")
        pricing_lines: list[str] = []
        for line in lines:
            if is_pricing_line(line):
                # 报价单行作为原子 chunk 保留
                all_chunks.append(Document(page_content=line.strip(), metadata=doc.metadata))
            else:
                pricing_lines.append(line)

        # 非报价单内容合并后走普通切分
        if pricing_lines:
            non_pricing_text = "\n".join(pricing_lines)
            non_pricing_docs.append(Document(page_content=non_pricing_text, metadata=doc.metadata))

    # 非报价单内容使用 RecursiveCharacterTextSplitter 切分
    if non_pricing_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        regular_chunks = splitter.split_documents(non_pricing_docs)
        all_chunks.extend(regular_chunks)

    cleaned = clean_chunks(all_chunks)
    logger.info(
        "Split complete raw_chunks=%s clean_chunks=%s (pricing_lines=%s)",
        len(all_chunks),
        len(cleaned),
        sum(1 for c in all_chunks if is_pricing_line(c.page_content)),
    )
    return cleaned
