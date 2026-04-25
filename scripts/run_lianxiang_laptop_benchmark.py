#!/usr/bin/env python3
"""
联想电脑基本参数对比.docx 专项 Benchmark

用法:
    python scripts/run_lianxiang_laptop_benchmark.py                          # 本地模式
    python scripts/run_lianxiang_laptop_benchmark.py --api-base http://127.0.0.1:8000/api  # API模式
"""
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


DEFAULT_EVAL_SET = Path("docs/lianxiang_laptop_eval_set.json")
DEFAULT_FILENAME = "联想电脑基本参数对比.docx"

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
    quality_issues = inspect_answer_quality(answer_text, source_text)
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
        raise SystemExit(f"评测集文件不存在：{eval_set}")
    return cases[:limit] if limit else cases


def _run_question(index: int, case: dict, doc_id: str, mode: str, api_base: str | None, k: int) -> dict:
    question = case["question"]
    session_id = f"lianxiang-benchmark-{int(time.time())}-{index}"
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
    parser = argparse.ArgumentParser(
        description="联想电脑基本参数对比.docx 专项 RAG 评测"
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"要评测的文档名 (默认: {DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--eval-set",
        default=str(DEFAULT_EVAL_SET),
        help="评测集 JSON 文件路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只运行前 N 个问题 (默认全部)",
    )
    parser.add_argument(
        "--mode",
        choices=["qa", "chat"],
        default="qa",
        help="问答模式",
    )
    parser.add_argument(
        "--api-base",
        default="",
        help="API 服务地址，如 http://127.0.0.1:8000/api",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Recall@K / nDCG@K 的 K 值",
    )
    args = parser.parse_args()

    settings = get_settings()
    doc_id = _find_doc_id(args.filename)
    cases = _load_cases(Path(args.eval_set), args.limit)
    rows = []

    print(f"开始评测: {args.filename}")
    print(f"文档ID: {doc_id}")
    print(f"评测问题数: {len(cases)}")
    print(f"模式: {args.mode}")
    print(f"API: {args.api_base or 'local-chain'}")
    print("-" * 60)

    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['question']}", flush=True)
        row = _run_question(idx, case, doc_id, args.mode, args.api_base or None, args.k)
        rows.append(row)
        eval_text = ""
        if row.get("evaluation"):
            evaluation = row["evaluation"]
            eval_text = (
                f" | recall@{args.k}={evaluation['recall_at_k']}"
                f" answer_score={evaluation['answer_score']}"
                f" hallucinated={evaluation['hallucinated']}"
            )
        print(f"  sources={row['source_count']} elapsed={row['elapsed_seconds']}s{eval_text}", flush=True)

    summary = summarize_metrics(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.filename.replace(".docx", "").replace(" ", "_")
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
    json_path = output_dir / f"lianxiang_laptop_benchmark_{timestamp}.json"
    md_path = output_dir / f"lianxiang_laptop_benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    print("-" * 60)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    _print_summary(summary, args.k)


def _print_summary(summary: dict, k: int) -> None:
    if not summary:
        return
    print("\n========== 评测汇总 ==========")
    print(f"评测用例数: {summary.get('case_count')}")
    print(f"答案准确率: {summary.get('answer_accuracy'):.2%}")
    print(f"平均答案得分: {summary.get('avg_answer_score'):.4f}")
    print(f"平均 Recall@{k}: {summary.get('avg_recall_at_k'):.4f}")
    print(f"平均 MRR: {summary.get('avg_mrr'):.4f}")
    print(f"平均 nDCG@{k}: {summary.get('avg_ndcg_at_k'):.4f}")
    print(f"拒答率: {summary.get('refusal_rate'):.2%}")
    print(f"幻觉率: {summary.get('hallucination_rate'):.2%}")
    print("=" * 30)


def _to_markdown(payload: dict) -> str:
    lines = [
        "# 联想电脑 RAG Benchmark 结果",
        "",
        f"- **文件**: {payload['filename']}",
        f"- **doc_id**: `{payload['doc_id']}`",
        f"- **评测集**: {payload['eval_set']}",
        f"- **检索后端**: {payload['retrieval_backend']}",
        f"- **模式**: {payload['mode']}",
        f"- **API地址**: {payload['api_base'] or 'local-chain'}",
        f"- **K值**: {payload['k']}",
        f"- **问题数**: {payload['question_count']}",
        "",
    ]
    if payload.get("summary"):
        s = payload["summary"]
        lines.extend(["## 评测汇总", ""])
        lines.append(f"- 评测用例数: {s.get('case_count', 0)}")
        lines.append(f"- 答案准确率: {s.get('answer_accuracy', 0):.2%}")
        lines.append(f"- 平均答案得分: {s.get('avg_answer_score', 0):.4f}")
        lines.append(f"- 平均 Recall@{payload['k']}: {s.get('avg_recall_at_k', 0):.4f}")
        lines.append(f"- 平均 MRR: {s.get('avg_mrr', 0):.4f}")
        lines.append(f"- 平均 nDCG@{payload['k']}: {s.get('avg_ndcg_at_k', 0):.4f}")
        lines.append(f"- 拒答率: {s.get('refusal_rate', 0):.2%}")
        lines.append(f"- 幻觉率: {s.get('hallucination_rate', 0):.2%}")
        lines.append("")

    for row in payload["results"]:
        lines.extend([
            f"## {row['index']}. {row['question']}",
            "",
            f"- **case_id**: `{row['case_id']}`",
            f"- **耗时**: {row['elapsed_seconds']}s",
            f"- **引用数量**: {row['source_count']}",
        ])
        if row.get("evaluation"):
            e = row["evaluation"]
            lines.extend([
                f"- **answer_score**: {e['answer_score']}",
                f"- **answer_correct**: {e['answer_correct']}",
                f"- **recall@{payload['k']}**: {e['recall_at_k']}",
                f"- **mrr**: {e['mrr']}",
                f"- **ndcg@{payload['k']}**: {e['ndcg_at_k']}",
                f"- **refused**: {e['refused']}",
                f"- **hallucinated**: {e['hallucinated']}",
                f"- **unsupported_numbers**: {e['unsupported_numbers']}",
                f"- **quality_issues**: {e['quality_issues']}",
            ])
        lines.extend(["", row["answer"], "", "**引用预览:**", ""])
        for idx, source in enumerate(row["sources"], 1):
            lines.append(f"{idx}. {source['filename']} 第 {source['page']} 页：{source['content'][:200]}...")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
