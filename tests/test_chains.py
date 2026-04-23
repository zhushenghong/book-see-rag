import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from book_see_rag.chains.answer_cleanup import clean_answer_text
from book_see_rag.chains.answer_guardrails import find_unsupported_numbers
from book_see_rag.chains.answer_quality import inspect_answer_quality
from book_see_rag.retrieval import filter_meta_evaluation_chunks, prefilter_hits


class _FakePrompt:
    def __init__(self, output: str):
        self.output = output

    def __or__(self, _other):
        return self

    def invoke(self, _payload):
        return self.output


@pytest.fixture
def mock_retrieval():
    """Mock 检索和精排，返回固定 chunks / hits"""
    chunks = ["文档片段一：主题是人工智能。", "文档片段二：深度学习是核心技术。"]
    hits = [
        {"doc_id": "d1", "filename": "简历.pdf", "page": 1, "content": chunks[0], "score": 0.9},
        {"doc_id": "d1", "filename": "简历.pdf", "page": 2, "content": chunks[1], "score": 0.8},
    ]
    with patch("book_see_rag.chains.qa_chain.search_hits", return_value=hits), \
         patch("book_see_rag.chains.qa_chain.get_doc_hits", return_value=hits), \
         patch("book_see_rag.chains.qa_chain.rerank", return_value=chunks), \
         patch("book_see_rag.chains.extraction_chain.search", return_value=chunks), \
         patch("book_see_rag.chains.chat_chain.search_hits", return_value=hits), \
         patch("book_see_rag.chains.chat_chain.get_doc_hits", return_value=hits), \
         patch("book_see_rag.chains.chat_chain.rerank", return_value=chunks):
        yield {"chunks": chunks, "hits": hits}


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="这是模拟的 LLM 回答")
    llm.__or__ = lambda self, other: other  # 支持 | 链式调用
    with patch("book_see_rag.llm.factory.create_llm", return_value=llm):
        yield llm


# ── QA Chain ─────────────────────────────────────────────────

def test_qa_answer_returns_dict(mock_retrieval, mock_llm):
    with patch("book_see_rag.chains.qa_chain._PROMPT") as mock_prompt, \
         patch("book_see_rag.chains.qa_chain.find_unsupported_numbers", return_value=[]), \
         patch("book_see_rag.chains.qa_chain.inspect_answer_quality", return_value=[]):
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = "这是答案"
        mock_prompt.__or__ = MagicMock(return_value=chain_mock)

        from book_see_rag.chains.qa_chain import answer
        result = answer("什么是人工智能？")
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)


def test_qa_returns_quality_failure_when_strict_retry_still_bad():
    hits = [
        {"doc_id": "d1", "filename": "rag.md", "page": 1, "content": "BM25 是一种关键词检索算法。", "score": 0.9},
    ]
    settings = SimpleNamespace(
        retrieval_prefilter_top_k=4,
        enable_rerank=False,
        rerank_top_k=1,
    )

    from book_see_rag.chains import qa_chain

    with patch("book_see_rag.chains.qa_chain.get_settings", return_value=settings), \
         patch("book_see_rag.chains.qa_chain._retrieve_hits", return_value=hits), \
         patch("book_see_rag.chains.qa_chain.create_llm", return_value=MagicMock()), \
         patch("book_see_rag.chains.qa_chain._PROMPT", _FakePrompt("chunk far太小，chunk 太nard。")), \
         patch("book_see_rag.chains.qa_chain._STRICT_PROMPT", _FakePrompt("答案如下：\n- ")):
        result = qa_chain.answer("BM25 是什么？")

    assert result["answer"] == qa_chain._QUALITY_FAILURE_MESSAGE


# ── Extraction Chain ─────────────────────────────────────────

def test_extraction_returns_knowledge_item(mock_retrieval):
    from book_see_rag.chains.extraction_chain import KnowledgeItem
    mock_item = KnowledgeItem(
        entities=["人工智能", "深度学习"],
        key_facts=["深度学习是人工智能的核心技术"],
        relationships=["人工智能 → 包含 → 深度学习"],
    )
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = mock_item
    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = structured_llm

    with patch("book_see_rag.llm.factory.create_llm", return_value=mock_llm_instance), \
         patch("book_see_rag.chains.extraction_chain._PROMPT") as mock_prompt:
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_item
        mock_prompt.__or__ = MagicMock(return_value=chain_mock)

        from book_see_rag.chains.extraction_chain import extract
        result = extract("主要知识点")
        assert isinstance(result, KnowledgeItem)
        assert len(result.entities) > 0


def test_chat_returns_answer_and_persists_sources(mock_retrieval):
    llm = MagicMock()
    llm.__or__ = lambda self, other: other

    with patch("book_see_rag.chains.chat_chain.create_llm", return_value=llm), \
         patch("book_see_rag.chains.chat_chain._PROMPT") as mock_prompt, \
         patch("book_see_rag.chains.chat_chain._REWRITE_PROMPT") as rewrite_prompt, \
         patch("book_see_rag.chains.chat_chain.get_recent_messages", return_value=[]), \
         patch("book_see_rag.chains.chat_chain.append_user_message") as append_user, \
         patch("book_see_rag.chains.chat_chain.append_ai_message") as append_ai:
        rewrite_chain = MagicMock()
        rewrite_chain.invoke.return_value = "独立检索问题"
        rewrite_chain.__or__ = MagicMock(return_value=rewrite_chain)
        rewrite_prompt.__or__ = MagicMock(return_value=rewrite_chain)

        answer_chain = MagicMock()
        answer_chain.invoke.return_value = "这是多轮回答"
        answer_chain.__or__ = MagicMock(return_value=answer_chain)
        mock_prompt.__or__ = MagicMock(return_value=answer_chain)

        from book_see_rag.chains.chat_chain import chat

        result = chat("session-001", "那研发岗呢？")
        assert result["answer"] == "这是多轮回答"
        assert result["sources"] == mock_retrieval["hits"]
        append_user.assert_called_once_with("session-001", "那研发岗呢？")
        append_ai.assert_called_once_with("session-001", "这是多轮回答", mock_retrieval["hits"], scope=None)


def test_chat_returns_quality_failure_when_strict_retry_still_bad():
    hits = [
        {"doc_id": "d1", "filename": "rag.md", "page": 1, "content": "BM25 是一种关键词检索算法。", "score": 0.9},
    ]
    settings = SimpleNamespace(
        chat_history_window=6,
        followup_rewrite_history=2,
        retrieval_prefilter_top_k=4,
        enable_rerank=False,
        rerank_top_k=1,
    )

    from book_see_rag.chains import chat_chain

    with patch("book_see_rag.chains.chat_chain.get_settings", return_value=settings), \
         patch("book_see_rag.chains.chat_chain.get_recent_messages", return_value=[]), \
         patch("book_see_rag.chains.chat_chain._rewrite_query", return_value="BM25 是什么？"), \
         patch("book_see_rag.chains.chat_chain._retrieve_hits", return_value=hits), \
         patch("book_see_rag.chains.chat_chain.create_llm", return_value=MagicMock()), \
         patch("book_see_rag.chains.chat_chain.append_user_message") as append_user, \
         patch("book_see_rag.chains.chat_chain.append_ai_message") as append_ai, \
         patch("book_see_rag.chains.chat_chain._PROMPT", _FakePrompt("chunk far太小，chunk 太nard。")), \
         patch("book_see_rag.chains.chat_chain._STRICT_PROMPT", _FakePrompt("答案如下：\n- ")):
        result = chat_chain.chat("session-001", "BM25 是什么？")

    assert result["answer"] == chat_chain._QUALITY_FAILURE_MESSAGE
    append_user.assert_called_once_with("session-001", "BM25 是什么？")
    append_ai.assert_called_once_with("session-001", chat_chain._QUALITY_FAILURE_MESSAGE, hits, scope=None)


def test_chat_refuses_when_all_hits_are_noisy():
    noisy_hits = [
        {"doc_id": "d1", "filename": "简历.pdf", "page": 1, "content": "V V V V V V V V V"},
    ]
    llm = MagicMock()

    with patch("book_see_rag.chains.chat_chain.search_hits", return_value=noisy_hits), \
         patch("book_see_rag.chains.chat_chain.create_llm", return_value=llm), \
         patch("book_see_rag.chains.chat_chain.get_recent_messages", return_value=[]), \
         patch("book_see_rag.chains.chat_chain.append_user_message") as append_user, \
         patch("book_see_rag.chains.chat_chain.append_ai_message") as append_ai:
        from book_see_rag.chains.chat_chain import chat

        result = chat("session-001", "这个人会什么")
        assert "无法可靠回答" in result["answer"]
        assert result["sources"] == []
        append_user.assert_called_once()
        append_ai.assert_called_once()


def test_chat_falls_back_to_doc_scope_hits_when_ann_returns_empty():
    llm = MagicMock()
    llm.__or__ = lambda self, other: other
    fallback_hits = [
        {"doc_id": "d1", "filename": "bm25.docx", "page": 1, "content": "BM25 是一种信息检索中的排名函数。", "score": 0.0},
    ]

    with patch("book_see_rag.chains.chat_chain.search_hits", return_value=[]), \
         patch("book_see_rag.chains.chat_chain.get_doc_hits", return_value=fallback_hits), \
         patch("book_see_rag.chains.chat_chain.create_llm", return_value=llm), \
         patch("book_see_rag.chains.chat_chain.get_recent_messages", return_value=[]), \
         patch("book_see_rag.chains.chat_chain.append_user_message"), \
         patch("book_see_rag.chains.chat_chain.append_ai_message"), \
         patch("book_see_rag.chains.chat_chain._PROMPT") as mock_prompt:
        answer_chain = MagicMock()
        answer_chain.invoke.return_value = "BM25 是一种信息检索中的排名函数。"
        answer_chain.__or__ = MagicMock(return_value=answer_chain)
        mock_prompt.__or__ = MagicMock(return_value=answer_chain)

        from book_see_rag.chains.chat_chain import chat

        result = chat("session-001", "bm25 是什么", doc_ids=["d1"])
        assert "排名函数" in result["answer"]
        assert result["sources"] == fallback_hits


def test_qa_falls_back_to_doc_scope_hits_when_ann_returns_empty():
    fallback_hits = [
        {"doc_id": "d1", "filename": "bm25.docx", "page": 1, "content": "BM25 是一种信息检索中的排名函数。", "score": 0.0},
    ]
    with patch("book_see_rag.chains.qa_chain.search_hits", return_value=[]), \
         patch("book_see_rag.chains.qa_chain.get_doc_hits", return_value=fallback_hits), \
         patch("book_see_rag.chains.qa_chain.rerank", return_value=["BM25 是一种信息检索中的排名函数。"]), \
         patch("book_see_rag.chains.qa_chain._PROMPT") as mock_prompt:
        answer_chain = MagicMock()
        answer_chain.invoke.return_value = "BM25 是一种信息检索中的排名函数。"
        answer_chain.__or__ = MagicMock(return_value=answer_chain)
        mock_prompt.__or__ = MagicMock(return_value=answer_chain)

        from book_see_rag.chains.qa_chain import answer

        result = answer("bm25 是什么", doc_ids=["d1"])
        assert "排名函数" in result["answer"]


def test_chat_uses_llamaindex_backend_when_enabled():
    hits = [
        {"doc_id": "d1", "filename": "bm25.docx", "page": 1, "content": "BM25 是一种排名函数。", "score": 0.7},
    ]
    settings = SimpleNamespace(retrieval_backend="llamaindex")
    with patch("book_see_rag.chains.chat_chain.get_settings", return_value=settings), \
         patch("book_see_rag.chains.chat_chain.search_hits_with_llamaindex", return_value=hits) as llama_search:
        from book_see_rag.chains.chat_chain import _retrieve_hits

        result = _retrieve_hits("bm25 是什么", ["d1"])
        assert result == hits
        llama_search.assert_called_once_with("bm25 是什么", doc_ids=["d1"])


def test_chat_llamaindex_backend_falls_back_to_milvus():
    hits = [
        {"doc_id": "d1", "filename": "bm25.docx", "page": 1, "content": "BM25 是一种排名函数。", "score": 0.7},
    ]
    settings = SimpleNamespace(retrieval_backend="llamaindex")
    with patch("book_see_rag.chains.chat_chain.get_settings", return_value=settings), \
         patch("book_see_rag.chains.chat_chain.search_hits_with_llamaindex", side_effect=RuntimeError("boom")), \
         patch("book_see_rag.chains.chat_chain.search_hits", return_value=hits) as milvus_search:
        from book_see_rag.chains.chat_chain import _retrieve_hits

        result = _retrieve_hits("bm25 是什么", ["d1"])
        assert result == hits
        milvus_search.assert_called_once_with("bm25 是什么", doc_ids=["d1"])


def test_prefilter_hits_prioritizes_lexical_match():
    hits = [
        {"doc_id": "d1", "filename": "doc.pdf", "page": 1, "content": "这段只谈公司制度", "score": 0.95},
        {"doc_id": "d1", "filename": "resume.pdf", "page": 2, "content": "熟悉 Python、Java、RAG 与模型部署", "score": 0.60},
    ]
    ranked = prefilter_hits("技术栈 Python", hits, limit=2)
    assert ranked[0]["content"] == "熟悉 Python、Java、RAG 与模型部署"


def test_prefilter_hits_downranks_meta_evaluation_chunks():
    hits = [
        {"doc_id": "d1", "filename": "doc.pdf", "page": 1, "content": "推荐测试问题：当前系统支持哪些文件格式？", "score": 0.95},
        {"doc_id": "d1", "filename": "doc.pdf", "page": 2, "content": "系统目前支持四种文件格式：PDF、DOCX、TXT、Markdown。", "score": 0.60},
    ]
    ranked = prefilter_hits("当前系统支持哪些文件格式？", hits, limit=2)
    assert ranked[0]["content"].startswith("系统目前支持四种文件格式")


def test_filter_meta_evaluation_chunks_removes_meta_when_possible():
    hits = [
        {"doc_id": "d1", "filename": "doc.pdf", "page": 1, "content": "推荐测试问题：当前系统支持哪些文件格式？", "score": 0.95},
        {"doc_id": "d1", "filename": "doc.pdf", "page": 2, "content": "系统目前支持四种文件格式：PDF、DOCX、TXT、Markdown。", "score": 0.60},
    ]
    filtered = filter_meta_evaluation_chunks(hits)
    assert len(filtered) == 1
    assert filtered[0]["page"] == 2


def test_find_unsupported_numbers_detects_answer_numbers_not_in_evidence():
    evidence = "第一阶段试点范围包括 30 名员工，其中研发部门 18 人，人事部门 5 人。"
    unsupported = find_unsupported_numbers("试点 35 人，研发 18 人。", evidence)
    assert unsupported == ["35人"]


def test_inspect_answer_quality_detects_benchmark_bad_outputs():
    assert "contains_garbled_english" in inspect_answer_quality("chunk far太小，chunk 太nard。")
    assert "contains_garbled_english" in inspect_answer_quality("Milvus 作为向 kuku，pérdida。")
    assert "contains_prompt_leak" in inspect_answer_quality("user 请问一下 LlamaIndex 的 reranker 是什么？")
    assert "contains_repeated_terms" in inspect_answer_quality("Redis 保存多轮会会会话记忆。")
    assert "contains_broken_format" in inspect_answer_quality("答案如下：\n- ")


def test_inspect_answer_quality_allows_common_rag_terms():
    answer = "RAG 使用 BM25、embedding、reranker、Milvus、FastAPI 和 LlamaIndex。"
    assert inspect_answer_quality(answer) == []


def test_clean_answer_text_removes_broken_chars_and_common_noise():
    raw = "根据提供的信息，，L1mmaIndex 和 MIlvus\n`chunksize 表示 单�个文本片段。chunk overlap �表示 重复保留。发研知识库 85%% token)) embeddingding"
    cleaned = clean_answer_text(raw)
    assert "�" not in cleaned
    assert "，，" not in cleaned
    assert "`" not in cleaned
    assert "LlamaIndex" in cleaned
    assert "Milvus" in cleaned
    assert "85%" in cleaned
    assert "token)" in cleaned
    assert "embeddingding" not in cleaned
    assert "chunk_size" in cleaned
    assert "chunk_overlap" in cleaned
    assert "研发知识库" in cleaned
