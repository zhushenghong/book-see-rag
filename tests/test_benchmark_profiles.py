from book_see_rag.benchmark_profiles import resolve_doc_profile, resolve_eval_set_for_profile


def test_resolve_eval_set_for_profile_uses_mapping():
    assert resolve_eval_set_for_profile("laptop") == "docs/lianxiang_laptop_eval_set.json"


def test_resolve_doc_profile_reads_kb_binding(monkeypatch):
    monkeypatch.setattr(
        "book_see_rag.benchmark_profiles.list_documents",
        lambda: [{"doc_id": "d1", "filename": "联想电脑基本参数对比.docx", "kb_id": "kb_laptop"}],
    )
    monkeypatch.setattr(
        "book_see_rag.benchmark_profiles.get_knowledge_base",
        lambda _kb_id: {"kb_id": "kb_laptop", "query_profile": "laptop"},
    )

    assert resolve_doc_profile("联想电脑基本参数对比.docx") == "laptop"
