import pdfplumber
from pathlib import Path
from langchain_core.documents import Document
from book_see_rag.config import get_settings


def _extract_text_pages(path: str) -> list[tuple[int, str]]:
    """用 pdfplumber 提取每页文字，返回 [(page_num, text), ...]"""
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text.strip()))
    return pages


def _needs_ocr(pages: list[tuple[int, str]]) -> bool:
    """若超过半数页面文字量不足阈值，判定为扫描版"""
    settings = get_settings()
    threshold = settings.ocr_min_chars_per_page
    sparse = sum(1 for _, text in pages if len(text) < threshold)
    return sparse > len(pages) / 2


def _ocr_with_marker(path: str) -> list[Document]:
    """用 marker-pdf 对扫描版 PDF 做 OCR，返回按页的 Document 列表"""
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    settings = get_settings()
    models = load_all_models()
    full_text, images, metadata = convert_single_pdf(
        path, models, langs=["Chinese", "English"]
    )
    # marker 输出整篇 Markdown，按分页符拆分
    page_texts = full_text.split("\n\n---\n\n")
    return [
        Document(
            page_content=text.strip(),
            metadata={"source": path, "page": i + 1, "is_ocr": True},
        )
        for i, text in enumerate(page_texts)
        if text.strip()
    ]


def load_pdf(path: str) -> list[Document]:
    """
    自动判断 PDF 类型：
    - 文字版 → pdfplumber
    - 扫描版 → marker-pdf OCR
    """
    pages = _extract_text_pages(path)

    if _needs_ocr(pages):
        return _ocr_with_marker(path)

    return [
        Document(
            page_content=text,
            metadata={"source": path, "page": page_num, "is_ocr": False},
        )
        for page_num, text in pages
        if text
    ]
