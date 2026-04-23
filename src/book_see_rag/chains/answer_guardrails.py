from __future__ import annotations

import re


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:年|月|日|人|名|%|MB|GB|KB)?", re.IGNORECASE)


def extract_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        value = re.sub(r"\s+", "", match)
        if value and value not in seen:
            seen.add(value)
            numbers.append(value)
    return numbers


def find_unsupported_numbers(answer: str, evidence_text: str) -> list[str]:
    evidence = re.sub(r"\s+", "", evidence_text)
    unsupported: list[str] = []
    for number in extract_numbers(answer):
        if number not in evidence:
            unsupported.append(number)
    return unsupported
