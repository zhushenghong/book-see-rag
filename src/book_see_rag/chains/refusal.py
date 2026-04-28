from __future__ import annotations

import re


EVIDENCE_REFUSAL_MESSAGE = "依据不足，无法可靠回答：参考内容中没有直接证据支持该问题。"
QUALITY_FAILURE_MESSAGE = "依据不足，无法可靠回答：检索到的证据不足以生成通过质量校验的答案。"

_REFUSAL_TEXT_RE = re.compile(
    r"^\s*(?:"
    r"依据不足|无法可靠回答|无法回答|不能回答|"
    r"参考内容中(?:没有|未找到).*(?:直接证据|相关信息|关于)|"
    r"没有足够.*证据|检索到的证据不足|未通过质量校验"
    r")"
)


def canonicalize_refusal_text(answer: str) -> str:
    if _REFUSAL_TEXT_RE.search(answer):
        return EVIDENCE_REFUSAL_MESSAGE
    return answer
