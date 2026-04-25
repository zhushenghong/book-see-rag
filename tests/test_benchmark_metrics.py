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
    assert evaluation["answer_score"] == 1.0
    assert evaluation["recall_at_k"] == 1.0
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


def test_summarize_metrics_aggregates_rates():
    rows = [
        {"evaluation": {"answer_correct": True, "answer_score": 1.0, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, "refused": False, "hallucinated": False}},
        {"evaluation": {"answer_correct": False, "answer_score": 0.5, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "refused": True, "hallucinated": True}},
    ]

    summary = summarize_metrics(rows)

    assert summary["answer_accuracy"] == 0.5
    assert summary["avg_answer_score"] == 0.75
    assert summary["avg_recall_at_k"] == 0.5
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
    rows = [
        {"categories": ["参数抽取"], "evaluation": {"answer_correct": True, "answer_score": 1.0, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, "refused": False, "hallucinated": False}},
        {"categories": ["参数抽取", "多事实"], "evaluation": {"answer_correct": False, "answer_score": 0.5, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "refused": False, "hallucinated": True}},
    ]

    summary = summarize_by_category(rows)

    assert summary["参数抽取"]["case_count"] == 2
    assert summary["参数抽取"]["answer_accuracy"] == 0.5
    assert summary["多事实"]["case_count"] == 1
