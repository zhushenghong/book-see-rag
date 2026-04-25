#!/usr/bin/env python3
"""
联想消费显示器/配件/笔记本/台式机报价单 专项 RAG 评测
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
from book_see_rag.config import get_settings


DEFAULT_FILENAME = "消费 显示器 配件  报价2026年.txt"
DEFAULT_EVAL_SET = Path(__file__).parent.parent / "docs" / "display_pricing_eval_set.json"

_REFUSAL_RE = re.compile(r"检索到相关证据.*质量校验", re.IGNORECASE)


def _source_preview(source) -> dict:
    if isinstance(source, dict):
        return {
            "filename": source.get("filename", ""),
            "page": source.get("page", 0),
            "content": source.get("content", "")[:300],
            "score": source.get("score", 0.0),
        }
    # String source (from API)
    return {
        "filename": "",
        "page": 0,
        "content": str(source)[:300] if source else "",
        "score": 0.0,
    }


def _term_group_matched(text: str, group: str) -> bool:
    text_lower = text.lower()
    keywords = [k.strip() for k in group.split() if k.strip()]
    return all(k.lower() in text_lower for k in keywords if len(k) >= 2)


def _get_source_content(source) -> str:
    if isinstance(source, dict):
        return source.get("content", "")
    return str(source)


def _source_group_hits(sources: list, source_groups: list[str], k: int) -> list[int | None]:
    hits: list[int | None] = []
    top_sources = sources[:k]
    for group in source_groups:
        rank = None
        for idx, source in enumerate(top_sources, 1):
            if _term_group_matched(_get_source_content(source), group):
                rank = idx
                break
        hits.append(rank)
    return hits


def _dcg(scores: list[int]) -> float:
    return sum((2**s - 1) / math.log2(idx + 2) for idx, s in enumerate(scores))


def evaluate_row(row: dict, case: dict, k: int) -> dict:
    answer_text = row.get("answer", "")
    source_text = "\n\n".join(s.get("content", "") if isinstance(s, dict) else str(s) for s in row.get("sources", []))
    required_terms = case.get("required_answer_terms", [])
    source_terms = case.get("relevant_source_terms", [])
    forbidden_terms = case.get("forbidden_answer_terms", [])
    expected_refusal = case.get("expected_refusal", False)

    source_groups = source_terms or []
    source_ranks = _source_group_hits(row.get("sources", []), source_groups, k)
    retrieved_groups = sum(1 for rank in source_ranks if rank is not None)
    source_total = len(source_groups)
    recall_at_k = retrieved_groups / source_total if source_total else 0.0
    first_rank = min((rank for rank in source_ranks if rank is not None), default=None)
    mrr = 1 / first_rank if first_rank else 0.0
    actual_dcg = _dcg([rank for rank in source_ranks if rank is not None])
    ideal_dcg = _dcg(list(range(1, min(source_total, k) + 1))) if source_total else 0.0
    ndcg_at_k = actual_dcg / ideal_dcg if ideal_dcg else 0.0

    answer_matched = sum(1 for term in required_terms if term.lower() in answer_text.lower())
    answer_total = len(required_terms)
    answer_score = answer_matched / answer_total if answer_total else 0.0

    forbidden_matched = [t for t in forbidden_terms if t.lower() in answer_text.lower()]

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


def summarize_metrics(evaluations: list[dict]) -> dict:
    total = len(evaluations)
    if total == 0:
        return {}

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


def _list_documents() -> list[dict]:
    try:
        resp = httpx.get("http://127.0.0.1:8000/api/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _find_doc_id(filename: str) -> str:
    matches = [doc for doc in _list_documents() if doc["filename"] == filename]
    return matches[0]["doc_id"] if matches else ""


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


def _run_question(index: int, case: dict, doc_id: str, mode: str, api_base: str | None, k: int) -> dict:
    question = case["question"]
    session_id = f"display-benchmark-{int(time.time())}-{index}"
    started = time.perf_counter()
    if api_base:
        result = _run_question_via_api(api_base, session_id, question, doc_id, mode)
    else:
        from book_see_rag.chains.qa_chain import answer
        from book_see_rag.chains.chat_chain import chat
        if mode == "chat":
            result = chat(session_id=session_id, message=question, doc_ids=[doc_id])
        else:
            result = answer(question, doc_ids=[doc_id])
    elapsed = time.perf_counter() - started

    sources = result.get("sources", [])
    row = {
        "index": index,
        "case_id": case.get("id", f"q{index}"),
        "categories": case.get("categories", []),
        "question": question,
        "answer": result.get("answer", ""),
        "source_count": len(sources),
        "sources": [_source_preview(item) for item in sources[: max(k, 5)]],
        "elapsed_seconds": round(elapsed, 3),
    }
    if case.get("required_answer_terms") or case.get("relevant_source_terms"):
        row["evaluation"] = evaluate_row(row, case, k=k)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="联想消费显示器/配件/笔记本/台式机报价单 专项 RAG 评测")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help=f"要评测的文档名 (默认: {DEFAULT_FILENAME})")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET), help="评测集 JSON 文件路径")
    parser.add_argument("--limit", type=int, default=0, help="只运行前 N 个问题 (默认全部)")
    parser.add_argument("--mode", choices=["qa", "chat"], default="qa", help="问答模式")
    parser.add_argument("--api-base", default="", help="API 服务地址，如 http://127.0.0.1:8000/api")
    parser.add_argument("--output-dir", default="benchmark_results", help="结果输出目录")
    parser.add_argument("--k", type=int, default=5, help="Recall@K 的 K 值 (默认: 5)")
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    if not eval_set_path.exists():
        print(f"评测集文件不存在: {eval_set_path}")
        return

    with open(eval_set_path) as f:
        eval_set = json.load(f)
    cases = eval_set.get("cases", [])
    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"加载评测集: {eval_set.get('name', 'unknown')} ({len(cases)} 个用例)")
    print(f"文档: {args.filename}")
    print(f"API: {args.api_base or '直接调用'}")
    print("-" * 60)

    doc_id = _find_doc_id(args.filename)
    if not doc_id:
        print(f"警告: 未找到文档 '{args.filename}'，将使用空 doc_id")
    else:
        print(f"文档ID: {doc_id}")

    rows = []
    for idx, case in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {case['question']}", flush=True)
        row = _run_question(idx, case, doc_id, args.mode, args.api_base or None, args.k)
        rows.append(row)
        eval_text = ""
        if row.get("evaluation"):
            e = row["evaluation"]
            eval_text = (
                f" | recall@{args.k}={e['recall_at_k']}"
                f" answer_score={e['answer_score']}"
                f" hallucinated={e['hallucinated']}"
            )
        print(f"  sources={row['source_count']} elapsed={row['elapsed_seconds']}s{eval_text}", flush=True)

    summary = summarize_metrics([r["evaluation"] for r in rows if "evaluation" in r])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.filename.replace(".txt", "").replace(" ", "_")
    payload = {
        "filename": args.filename,
        "doc_id": doc_id,
        "eval_set": str(eval_set_path),
        "retrieval_backend": get_settings().retrieval_backend,
        "mode": args.mode,
        "api_base": args.api_base,
        "k": args.k,
        "question_count": len(rows),
        "summary": summary,
        "results": rows,
    }

    json_path = output_dir / f"display_pricing_benchmark_{timestamp}.json"
    md_path = output_dir / f"display_pricing_benchmark_{timestamp}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 联想消费报价单 RAG 评测报告\n\n")
        f.write(f"**文档**: {args.filename}\n\n")
        f.write(f"**评测集**: {eval_set.get('name', 'unknown')}\n\n")
        f.write(f"**时间**: {timestamp}\n\n")
        f.write(f"**模式**: {args.mode}\n\n")
        f.write("---\n\n")
        for row in rows:
            f.write(f"## {row['case_id']}: {row['question']}\n\n")
            f.write(f"**耗时**: {row['elapsed_seconds']}s\n\n")
            if row.get("evaluation"):
                e = row["evaluation"]
                f.write(f"- **answer_score**: {e['answer_score']}\n")
                f.write(f"- **answer_correct**: {e['answer_correct']}\n")
                f.write(f"- **recall@{args.k}**: {e['recall_at_k']}\n")
                f.write(f"- **mrr**: {e['mrr']}\n")
                f.write(f"- **refused**: {e['refused']}\n")
                f.write(f"- **hallucinated**: {e['hallucinated']}\n")
            f.write(f"\n**答案**:\n\n{row['answer']}\n\n")
            f.write("**引用预览:**\n\n")
            for idx, source in enumerate(row["sources"][:3], 1):
                content = source.get("content", "")[:200]
                f.write(f"{idx}. {content}...\n\n")

    print("-" * 60)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    _print_summary(summary, args.k)


def _print_summary(summary: dict, k: int) -> None:
    if not summary:
        return
    print("\n========== 评测汇总 ==========")
    print(f"评测用例数: {summary['case_count']}")
    print(f"答案准确率: {summary['answer_accuracy']:.2%}")
    print(f"平均答案得分: {summary['avg_answer_score']:.4f}")
    print(f"平均 Recall@{k}: {summary['avg_recall_at_k']:.4f}")
    print(f"平均 MRR: {summary['avg_mrr']:.4f}")
    print(f"平均 nDCG@{k}: {summary['avg_ndcg_at_k']:.4f}")
    print(f"拒答率: {summary['refusal_rate']:.2%}")
    print(f"幻觉率: {summary['hallucination_rate']:.2%}")
    print("==============================")


if __name__ == "__main__":
    main()
