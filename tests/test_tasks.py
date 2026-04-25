from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_ingest_task_runs_loader_splitter_and_insert():
    from book_see_rag.tasks.ingest_task import ingest_document

    task = ingest_document._get_current_object()
    task.update_state = MagicMock()
    task.push_request(id="task-001")

    docs = [Document(page_content="hello", metadata={"page": 1, "is_ocr": False})]
    chunks = [
        Document(page_content="chunk1", metadata={"page": 1, "is_ocr": False}),
        Document(page_content="chunk2", metadata={"page": 1, "is_ocr": False}),
    ]

    with patch("book_see_rag.ingestion.loader.load_document", return_value=docs) as mock_load, \
         patch("book_see_rag.ingestion.splitter.split_documents", return_value=chunks) as mock_split, \
         patch("book_see_rag.vectorstore.milvus_store.insert_chunks", return_value=len(chunks)) as mock_insert:
        result = task.run("doc-1", "/tmp/fake.txt", "fake.txt")
    task.pop_request()

    assert result["status"] == "done"
    assert result["doc_id"] == "doc-1"
    assert result["chunks"] == 2
    mock_load.assert_called_once_with("/tmp/fake.txt")
    mock_split.assert_called_once_with(docs)
    mock_insert.assert_called_once()
    assert task.update_state.call_args_list[0].kwargs["meta"]["step"] == "loading"
    assert task.update_state.call_args_list[1].kwargs["meta"]["step"] == "splitting"
    assert task.update_state.call_args_list[2].kwargs["meta"]["step"] == "embedding"
