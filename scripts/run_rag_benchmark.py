from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

from book_see_rag.chains.answer_guardrails import find_unsupported_numbers
from book_see_rag.chains.answer_quality import inspect_answer_quality
from book_see_rag.chains.chat_chain import chat
from book_see_rag.chains.qa_chain import answer
from book_see_rag.config import get_settings
from book_see_rag.metadata_store import list_documents


DEFAULT_EVAL_SET = Path("docs/rag_quality_eval_set.json")

QUESTIONS = [
    "星澜知识助手项目什么时候启动，第一阶段计划什么时候完成？",
    "项目负责人、技术负责人、产品负责人分别是谁？",
    "第一阶段试点总共多少人？研发、人事、综合管理分别多少人？",
    "当前系统用了哪些技术栈？",
    "当前系统支持哪些文件格式？",
    "RAG 和普通大模型问答有什么区别？",
    "chunk 和中文分词是一回事吗？为什么？",
    "embedding 是对整个文件做一次，还是对 chunk 分别做？",
    "chunk_size 和 chunk_overlap 分别是什么意思？",
    "为什么 chunk 太大或太小都会影响 RAG 效果？",
    "BM25 和向量检索有什么区别？分别适合什么场景？",
    "BM25 和向量检索是替代关系吗？",
    "Reranker 和向量数据库有什么区别？",
    "LlamaIndex 在当前系统里做了哪些事？",
    "LlamaIndex 是否替代了 Milvus、权限控制和最终回答生成？",
    "研发员工李明能访问哪些知识库？不能访问哪个知识库？",
    "HR 管理员王敏可以做哪些普通员工不能做的事？",
    "如果用户手动传入无权限 doc_id，系统应该怎么处理？",
    "公共知识库、研发知识库、人事知识库的可见范围分别是什么？",
    "为什么企业 RAG 系统需要最小权限模型？",
    "当前系统有哪些主要风险？",
    "第一阶段的三个关键验收指标是什么？",
    "为什么扫描版 PDF 会带来解析风险？",
    "为什么单纯向量检索可能漏掉精确术语？",
    "系统为什么需要引用校验？",
    "请同时说明：项目负责人是谁、试点总人数是多少、支持哪些文件格式、第一阶段验收指标有哪些。",
    "请对比 BM25、向量检索和 reranker，并说明它们在 RAG 中分别负责什么。",
    "请说明 LlamaIndex 当前做了什么、没有替代什么、为什么它不是大模型。",
    "请判断李明和王敏分别能访问哪些知识库，并说明原因。",
    "如果系统回答没有引用或引用不支持结论，应该如何处理？",
]

_REFUSAL_RE = re.compile(r"无法可靠回答|依据不足|未通过质量校验|没有足够.*证据|检索不到直接证据")


def _source_preview(item: object) -> dict:
    if isinstance(item, dict):
        return {
            "filename": item.get("filename", ""),
            "page": item.get("page", 0),
            "content": item.get("content", "")[:300],
            "score": item.get("score", 0.0),
        }
    return {
        "filename": "",
        "page": 0,
        "content": str(item)[:300],
        "score": 0.0,
    }


def _normalize_text(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _term_group_matched(text: str, group: list[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(term) in normalized for term in group)


def _matched_group_count(text: str, groups: list[list[str]]) -> int:
    return sum(1 for group in groups if _term_group_matched(text, group))


def _coerce_groups(value: object) -> list[list[str]]:
    if not value:
        return []
    groups: list[list[str]] = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, list):
            group = [str(term) for term in item if str(term)]
        else:
            group = [str(item)]
        if group:
            groups.append(group)
    return groups


def _source_group_hits(sources: list[dict], source_groups: list[list[str]], k: int) -> list[int | None]:
    hits: list[int | None] = []
    top_sources = sources[:k]
    for group in source_groups:
        rank = None
        for idx, source in enumerate(top_sources, 1):
            if _term_group_matched(source.get("content", ""), group):
                rank = idx
                break
        hits.append(rank)
    return hits


def _dcg(ranks: list[int]) -> float:
    return sum(1 / math.log2(rank + 1) for rank in ranks)


def evaluate_row(row: dict, case: dict, k: int = 5) -> dict:
    answer_text = row.get("answer", "")
    sources = row.get("sources", [])
    source_text = "\n".join(source.get("content", "") for source in sources if isinstance(source, dict))
    required_groups = _coerce_groups(case.get("required_answer_terms"))
    source_groups = _coerce_groups(case.get("relevant_source_terms"))
    forbidden_groups = _coerce_groups(case.get("forbidden_answer_terms"))
    expected_refusal = bool(case.get("expected_refusal"))

    answer_matched = _matched_group_count(answer_text, required_groups)
    answer_total = len(required_groups)
    answer_score = answer_matched / answer_total if answer_total else 0.0
    forbidden_matched = _matched_group_count(answer_text, forbidden_groups)

    source_ranks = _source_group_hits(sources, source_groups, k)
    retrieved_groups = sum(1 for rank in source_ranks if rank is not None)
    source_total = len(source_groups)
    recall_at_k = retrieved_groups / source_total if source_total else 0.0
    first_rank = min((rank for rank in source_ranks if rank is not None), default=None)
    mrr = 1 / first_rank if first_rank else 0.0
    actual_dcg = _dcg([rank for rank in source_ranks if rank is not None])
    ideal_dcg = _dcg(list(range(1, min(source_total, k) + 1))) if source_total else 0.0
    ndcg_at_k = actual_dcg / ideal_dcg if ideal_dcg else 0.0

    unsupported_numbers = find_unsupported_numbers(answer_text, source_text)
    quality_issues = inspect_answer_quality(answer_text)
    refused = bool(_REFUSAL_RE.search(answer_text))
    hallucinated = bool(unsupported_numbers or quality_issues or forbidden_matched)
    answer_correct = answer_total > 0 and answer_matched == answer_total and not refused and not hallucinated
    if expected_refusal:
        answer_score = 1.0 if refused else 0.0
        answer_correct = refused and not hallucinated

    return {
        "answer_score": round(answer_score, 4),
        "answer_correct": answer_correct,
        "answer_matched_terms": answer_matched,
        "answer_required_terms": answer_total,
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg_at_k, 4),
        "source_matched_terms": retrieved_groups,
        "source_required_terms": source_total,
        "refused": refused,
        "expected_refusal": expected_refusal,
        "hallucinated": hallucinated,
        "unsupported_numbers": unsupported_numbers,
        "quality_issues": quality_issues,
        "forbidden_matched_terms": forbidden_matched,
    }


def summarize_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    evaluations = [row["evaluation"] for row in rows if row.get("evaluation")]
    if not evaluations:
        return {}

    total = len(evaluations)

    def avg(key: str) -> float:
        return round(sum(float(item.get(key, 0.0)) for item in evaluations) / total, 4)

    return {
        "case_count": total,
        "answer_accuracy": round(sum(1 for item in evaluations if item.get("answer_correct")) / total, 4),
        "avg_answer_score": avg("answer_score"),
        "avg_recall_at_k": avg("recall_at_k"),
        "avg_mrr": avg("mrr"),
        "avg_ndcg_at_k": avg("ndcg_at_k"),
        "refusal_rate": round(sum(1 for item in evaluations if item.get("refused")) / total, 4),
        "hallucination_rate": round(sum(1 for item in evaluations if item.get("hallucinated")) / total, 4),
    }


def _find_doc_id(filename: str) -> str:
    matches = [doc for doc in list_documents() if doc["filename"] == filename]
    if not matches:
        raise SystemExit(f"未找到已摄入文档：{filename}")
    return matches[-1]["doc_id"]


def _load_cases(eval_set: Path, limit: int) -> list[dict]:
    if eval_set.exists():
        payload = json.loads(eval_set.read_text(encoding="utf-8"))
        cases = payload.get("cases", payload if isinstance(payload, list) else [])
        if not isinstance(cases, list) or not cases:
            raise SystemExit(f"评测集为空或格式错误：{eval_set}")
    else:
        cases = [{"id": f"q{idx}", "question": question} for idx, question in enumerate(QUESTIONS, 1)]
    return cases[:limit] if limit else cases


def _run_question(index: int, case: dict, doc_id: str, mode: str, api_base: str | None, k: int) -> dict:
    question = case["question"]
    session_id = f"benchmark-{int(time.time())}-{index}"
    started = time.perf_counter()
    if api_base:
        result = _run_question_via_api(api_base, session_id, question, doc_id, mode)
    elif mode == "chat":
        result = chat(
            session_id=session_id,
            message=question,
            doc_ids=[doc_id],
            scope={"doc_ids": [doc_id], "kb_ids": ["kb_public"]},
        )
    else:
        result = answer(question, doc_ids=[doc_id])
    elapsed = time.perf_counter() - started
    sources = result.get("sources", [])
    row = {
        "index": index,
        "case_id": case.get("id", f"q{index}"),
        "question": question,
        "answer": result.get("answer", ""),
        "source_count": len(sources),
        "sources": [_source_preview(item) for item in sources[: max(k, 5)]],
        "elapsed_seconds": round(elapsed, 3),
    }
    if case.get("required_answer_terms") or case.get("relevant_source_terms"):
        row["evaluation"] = evaluate_row(row, case, k=k)
    return row


def _run_question_via_api(api_base: str, session_id: str, question: str, doc_id: str, mode: str) -> dict:
    headers = {
        "X-User-Id": "benchmark",
        "X-Role": "hr_admin",
        "X-Department": "rd",
    }
    if mode == "chat":
        response = httpx.post(
            f"{api_base.rstrip('/')}/chat",
            json={"session_id": session_id, "message": question, "doc_ids": [doc_id]},
            headers=headers,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    response = httpx.post(
        f"{api_base.rstrip('/')}/query",
        json={"question": question, "mode": "qa", "doc_ids": [doc_id]},
        headers=headers,
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return {"answer": payload.get("result", ""), "sources": payload.get("sources") or []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed RAG quality benchmark questions with retrieval and answer metrics.")
    parser.add_argument("--filename", default="rag_quality_test.docx")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="Evaluation set JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N questions.")
    parser.add_argument("--mode", choices=["qa", "chat"], default="qa")
    parser.add_argument("--api-base", default="", help="Optional API base, e.g. http://127.0.0.1:8000/api")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--k", type=int, default=5, help="K used for Recall@K / nDCG@K.")
    args = parser.parse_args()

    settings = get_settings()
    doc_id = _find_doc_id(args.filename)
    cases = _load_cases(Path(args.eval_set), args.limit)
    rows = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['question']}", flush=True)
        row = _run_question(idx, case, doc_id, args.mode, args.api_base or None, args.k)
        rows.append(row)
        eval_text = ""
        if row.get("evaluation"):
            evaluation = row["evaluation"]
            eval_text = (
                f" recall@{args.k}={evaluation['recall_at_k']}"
                f" answer_score={evaluation['answer_score']}"
                f" hallucinated={evaluation['hallucinated']}"
            )
        print(f"  sources={row['source_count']} elapsed={row['elapsed_seconds']}s{eval_text}", flush=True)

    summary = summarize_metrics(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "filename": args.filename,
        "doc_id": doc_id,
        "eval_set": args.eval_set,
        "retrieval_backend": settings.retrieval_backend,
        "mode": args.mode,
        "api_base": args.api_base,
        "k": args.k,
        "question_count": len(rows),
        "summary": summary,
        "results": rows,
    }
    json_path = output_dir / f"rag_benchmark_{timestamp}.json"
    md_path = output_dir / f"rag_benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


def _to_markdown(payload: dict) -> str:
    lines = [
        "# RAG Benchmark Result",
        "",
        f"- 文件：{payload['filename']}",
        f"- doc_id：{payload['doc_id']}",
        f"- eval_set：{payload['eval_set']}",
        f"- retrieval_backend：{payload['retrieval_backend']}",
        f"- mode：{payload['mode']}",
        f"- api_base：{payload['api_base'] or 'local-chain'}",
        f"- K：{payload['k']}",
        f"- 问题数：{payload['question_count']}",
        "",
    ]
    if payload.get("summary"):
        lines.extend(["## Summary", ""])
        for key, value in payload["summary"].items():
            lines.append(f"- {key}：{value}")
        lines.append("")

    for row in payload["results"]:
        lines.extend(
            [
                f"## {row['index']}. {row['question']}",
                "",
                f"- case_id：{row['case_id']}",
                f"- 耗时：{row['elapsed_seconds']}s",
                f"- 引用数量：{row['source_count']}",
            ]
        )
        if row.get("evaluation"):
            evaluation = row["evaluation"]
            lines.extend(
                [
                    f"- answer_score：{evaluation['answer_score']}",
                    f"- answer_correct：{evaluation['answer_correct']}",
                    f"- recall@{payload['k']}：{evaluation['recall_at_k']}",
                    f"- mrr：{evaluation['mrr']}",
                    f"- ndcg@{payload['k']}：{evaluation['ndcg_at_k']}",
                    f"- refused：{evaluation['refused']}",
                    f"- hallucinated：{evaluation['hallucinated']}",
                    f"- unsupported_numbers：{evaluation['unsupported_numbers']}",
                    f"- quality_issues：{evaluation['quality_issues']}",
                ]
            )
        lines.extend(["", row["answer"], "", "引用预览："])
        for idx, source in enumerate(row["sources"], 1):
            lines.append(f"{idx}. {source['filename']} 第 {source['page']} 页：{source['content']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
