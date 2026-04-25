from __future__ import annotations

import re


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:年|月|日|人|名|%|MB|GB|KB|MHz|GHz|kg|mm|Wh|mah|W|Hz|nit)?", re.IGNORECASE)
_UNIT_SUFFIXES = ("年", "月", "日", "人", "名", "%", "mb", "gb", "kb", "mhz", "ghz", "kg", "mm", "wh", "mah", "w", "hz", "nit")


def extract_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        value = re.sub(r"\s+", "", match)
        if value and value not in seen:
            seen.add(value)
            numbers.append(value)
    return numbers


def _normalize_for_match(text: str) -> str:
    """Remove punctuation and Chinese characters, keep digits, letters for number matching."""
    import unicodedata
    # Keep only alphanumeric characters (removes punctuation and Chinese)
    return "".join(ch.lower() for ch in text if ch.isalnum())


def find_unsupported_numbers(answer: str, evidence_text: str) -> list[str]:
    evidence_normalized = _normalize_for_match(evidence_text)
    unsupported: list[str] = []
    for number in extract_numbers(answer):
        number_normalized = _normalize_for_match(number)
        if not number_normalized:
            continue
        if _has_number_support(number_normalized, evidence_normalized):
            continue
        unsupported.append(number)
    return unsupported


def _candidate_normalized_numbers(value: str) -> list[str]:
    lowered = value.lower()
    candidates = [lowered]
    for suffix in _UNIT_SUFFIXES:
        if lowered.endswith(suffix):
            candidates.append(lowered[: -len(suffix)])
            break
    # allow "windows11" style evidence when the extracted number is just "11"
    digits_only = "".join(ch for ch in lowered if ch.isdigit())
    if digits_only and digits_only not in candidates:
        candidates.append(digits_only)
    return [item for item in candidates if item]


def _has_number_support(number_normalized: str, evidence_normalized: str) -> bool:
    return any(candidate in evidence_normalized for candidate in _candidate_normalized_numbers(number_normalized))
