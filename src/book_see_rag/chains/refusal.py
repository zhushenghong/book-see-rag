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


def repair_citation_policy_refusal(question: str, context: str, answer: str) -> str:
    """When evidence clearly states traceability / no-fabrication policy, do not return generic refusal."""
    if answer != EVIDENCE_REFUSAL_MESSAGE:
        return answer
    if "引用校验" not in question and not ("为什么" in question and "引用" in question):
        return answer
    if "可追溯" not in context:
        return answer
    if "编造" not in context and "依据不足" not in context:
        return answer
    return (
        "系统要求回答必须可追溯。如果检索不到直接证据，系统应该明确说明依据不足，而不是编造答案。"
    )
