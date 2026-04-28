import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_rag_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("run_rag_benchmark", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

evaluate_row = _MODULE.evaluate_row
infer_case_categories = _MODULE.infer_case_categories
summarize_metrics = _MODULE.summarize_metrics
summarize_by_category = _MODULE.summarize_by_category
to_markdown = _MODULE._to_markdown


def test_profile_default_eval_set_resolution(monkeypatch):
    monkeypatch.setattr(_MODULE, "resolve_doc_profile", lambda _filename: "laptop")
    monkeypatch.setattr(_MODULE, "resolve_eval_set_for_profile", lambda _profile: "docs/lianxiang_laptop_eval_set.json")

    profile_name = "laptop"
    resolved_eval_set = "docs/rag_quality_eval_set.json"
    if profile_name:
        resolved_eval_set = _MODULE.resolve_eval_set_for_profile(profile_name) or resolved_eval_set

    assert resolved_eval_set == "docs/lianxiang_laptop_eval_set.json"


def test_evaluate_row_scores_answer_and_retrieval_metrics():
    case = {
        "required_answer_terms": ["陈舟", "30 名员工"],
        "relevant_source_terms": ["项目负责人是陈舟", "第一阶段试点范围包括 30 名员工"],
    }
    row = {
        "answer": "项目负责人是陈舟，第一阶段试点范围包括 30 名员工。",
        "sources": [
            {"content": "无关内容"},
            {"content": "项目负责人是陈舟。第一阶段试点范围包括 30 名员工。"},
        ],
    }

    evaluation = evaluate_row(row, case, k=2)

    assert evaluation["answer_correct"] is True
    assert evaluation["answer_correct_lenient"] is True
    assert evaluation["answer_score"] == 1.0
    assert evaluation["recall_at_k"] == 1.0
    assert evaluation["recall_in_retrieved"] == 1.0
    assert evaluation["mrr"] == 0.5
    assert evaluation["ndcg_at_k"] < 1.0
    assert evaluation["hallucinated"] is False


def test_evaluate_row_detects_hallucinated_numbers_and_forbidden_terms():
    case = {
        "required_answer_terms": ["准确率不低于 85%"],
        "relevant_source_terms": ["准确率不低于 85%"],
        "forbidden_answer_terms": ["80%"],
    }
    row = {
        "answer": "第一阶段要求准确率不低于 80%。",
        "sources": [{"content": "常见制度问题回答准确率不低于 85%。"}],
    }

    evaluation = evaluate_row(row, case, k=1)

    assert evaluation["answer_correct"] is False
    assert evaluation["hallucinated"] is True
    assert evaluation["forbidden_matched_terms"] == 1


def test_evaluate_row_respects_min_answer_score_for_lenient_correctness():
    case = {
        "required_answer_terms": ["项甲", "项乙", "项丙", "项丁"],
        "relevant_source_terms": ["依据"],
        "min_answer_score": 0.75,
    }
    row = {
        "answer": "项甲项乙项丙",
        "sources": [{"content": "依据"}],
    }
    ev = evaluate_row(row, case, k=5)
    assert ev["answer_correct"] is False
    assert ev["answer_correct_lenient"] is True


def test_evaluate_row_supports_expected_refusal():
    case = {
        "expected_refusal": True,
        "required_answer_terms": ["依据不足"],
    }
    row = {
        "answer": "依据不足，无法可靠回答这个问题。",
        "sources": [],
    }

    evaluation = evaluate_row(row, case, k=5)

    assert evaluation["answer_correct"] is True
    assert evaluation["answer_score"] == 1.0
    assert evaluation["refused"] is True


def test_evaluate_row_does_not_treat_policy_mentions_as_refusal():
    case = {
        "required_answer_terms": ["回答必须可追溯", "不能编造答案"],
        "relevant_source_terms": ["系统要求回答必须可追溯", "而不是编造答案"],
    }
    row = {
        "answer": "系统要求回答必须可追溯；如果检索不到直接证据，应该说明依据不足，不能编造答案。",
        "sources": [{"content": "系统要求回答必须可追溯。如果检索不到直接证据，系统应该明确说明依据不足，而不是编造答案。"}],
    }

    evaluation = evaluate_row(row, case, k=5)

    assert evaluation["answer_correct"] is True
    assert evaluation["refused"] is False


def test_evaluate_row_recognizes_no_related_info_refusal():
    case = {"expected_refusal": True}
    row = {
        "answer": "参考内容中没有涉及食堂菜单的相关信息。",
        "sources": [],
    }

    evaluation = evaluate_row(row, case, k=5)

    assert evaluation["answer_correct"] is True
    assert evaluation["refused"] is True


def test_evaluate_row_does_not_treat_chunk_explanation_as_refusal():
    case = {
        "required_answer_terms": ["不是中文分词"],
        "relevant_source_terms": ["Chunk 不是中文分词"],
    }
    row = {
        "answer": "参考内容中没有涉及中文分词的定义或概念，因此无法判断。",
        "sources": [{"content": "Chunk 不是中文分词"}],
    }
    evaluation = evaluate_row(row, case, k=5)
    assert evaluation["refused"] is False


def test_summarize_metrics_aggregates_rates():
    rows = [
        {
            "evaluation": {
                "answer_correct": True,
                "answer_correct_lenient": True,
                "answer_score": 1.0,
                "recall_at_k": 1.0,
                "recall_in_retrieved": 1.0,
                "mrr": 1.0,
                "ndcg_at_k": 1.0,
                "refused": False,
                "hallucinated": False,
            }
        },
        {
            "evaluation": {
                "answer_correct": False,
                "answer_correct_lenient": False,
                "answer_score": 0.5,
                "recall_at_k": 0.0,
                "recall_in_retrieved": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
                "refused": True,
                "hallucinated": True,
            }
        },
    ]

    summary = summarize_metrics(rows)

    assert summary["answer_accuracy"] == 0.5
    assert summary["answer_accuracy_lenient"] == 0.5
    assert summary["avg_answer_score"] == 0.75
    assert summary["avg_recall_at_k"] == 0.5
    assert summary["avg_recall_in_retrieved"] == 0.5
    assert summary["refusal_rate"] == 0.5
    assert summary["hallucination_rate"] == 0.5


def test_infer_case_categories_prefers_explicit_categories():
    case = {"question": "示例", "categories": ["短追问", "多对象消歧"]}
    assert infer_case_categories(case) == ["短追问", "多对象消歧"]


def test_infer_case_categories_uses_question_heuristics():
    case = {"question": "请对比 BM25 和向量检索有什么区别？"}
    categories = infer_case_categories(case)
    assert "对比问答" in categories
    assert "解释说明" in categories


def test_summarize_by_category_groups_rows():
    _ev_ok = {
        "answer_correct": True,
        "answer_correct_lenient": True,
        "answer_score": 1.0,
        "recall_at_k": 1.0,
        "recall_in_retrieved": 1.0,
        "mrr": 1.0,
        "ndcg_at_k": 1.0,
        "refused": False,
        "hallucinated": False,
    }
    _ev_bad = {
        "answer_correct": False,
        "answer_correct_lenient": False,
        "answer_score": 0.5,
        "recall_at_k": 0.0,
        "recall_in_retrieved": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
        "refused": False,
        "hallucinated": True,
    }
    rows = [
        {"categories": ["参数抽取"], "evaluation": _ev_ok},
        {"categories": ["参数抽取", "多事实"], "evaluation": _ev_bad},
    ]

    summary = summarize_by_category(rows)

    assert summary["参数抽取"]["case_count"] == 2
    assert summary["参数抽取"]["answer_accuracy"] == 0.5
    assert summary["多事实"]["case_count"] == 1


def test_to_markdown_uses_chinese_labels():
    payload = {
        "filename": "rag_quality_test.docx",
        "doc_id": "d1",
        "eval_set": "docs/rag_quality_eval_set.json",
        "retrieval_backend": "llamaindex",
        "mode": "qa",
        "api_base": "",
        "k": 5,
        "question_count": 1,
        "summary": {
            "case_count": 1,
            "answer_accuracy": 1.0,
            "answer_accuracy_lenient": 1.0,
            "avg_answer_score": 1.0,
            "avg_recall_at_k": 1.0,
            "avg_recall_in_retrieved": 1.0,
            "avg_mrr": 1.0,
            "avg_ndcg_at_k": 1.0,
            "refusal_rate": 0.0,
            "hallucination_rate": 0.0,
        },
        "category_summary": {},
        "results": [
            {
                "index": 1,
                "question": "示例问题",
                "case_id": "case_1",
                "categories": ["参数抽取"],
                "elapsed_seconds": 1.23,
                "source_count": 1,
                "answer": "示例答案",
                "sources": [{"filename": "a.pdf", "page": 1, "content": "示例引用"}],
                "evaluation": {
                    "answer_score": 1.0,
                    "answer_correct": True,
                    "answer_correct_lenient": True,
                    "recall_at_k": 1.0,
                    "recall_in_retrieved": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_k": 1.0,
                    "refused": False,
                    "hallucinated": False,
                    "unsupported_numbers": [],
                    "quality_issues": [],
                },
            }
        ],
    }

    md = to_markdown(payload)

    assert "# RAG Benchmark 结果" in md
    assert "- 文档 ID：" in md
    assert "- 总体结果" not in md  # ensure heading formatting is separate
    assert "答案准确率（严格）" in md
    assert "召回率@5（Top-K）" in md
    assert "答案正确" in md
    assert "质量问题" in md
