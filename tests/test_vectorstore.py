from unittest.mock import patch


class _FakeHit:
    def __init__(self, entity: dict, distance: float):
        self.entity = entity
        self.distance = distance


class _FakeCollection:
    def __init__(self, search_hits=None, query_rows=None):
        self._search_hits = search_hits or []
        self._query_rows = query_rows or []

    def search(self, **_kwargs):
        return [self._search_hits]

    def query(self, **_kwargs):
        return list(self._query_rows)


def test_search_hits_skips_when_doc_scope_empty():
    from book_see_rag.vectorstore.milvus_store import search_hits

    with patch("book_see_rag.vectorstore.milvus_store.embed_query") as mock_embed, \
         patch("book_see_rag.vectorstore.milvus_store._get_collection", return_value=_FakeCollection()):
        hits = search_hits("test", doc_ids=[])

    assert hits == []
    mock_embed.assert_not_called()


def test_search_hits_dedupes_and_normalizes_content():
    from book_see_rag.vectorstore.milvus_store import search_hits

    fake_hits = [
        _FakeHit(
            entity={"doc_id": "d1", "filename": "a.pdf", "page": 1, "content": "  Hello   world \n"},
            distance=0.12,
        ),
        _FakeHit(
            entity={"doc_id": "d1", "filename": "a.pdf", "page": 1, "content": "Hello world"},
            distance=0.11,
        ),
        _FakeHit(
            entity={"doc_id": "d2", "filename": "b.pdf", "page": 2, "content": "Second   hit"},
            distance=0.08,
        ),
    ]
    col = _FakeCollection(search_hits=fake_hits)

    with patch("book_see_rag.vectorstore.milvus_store.embed_query", return_value=[0.0, 0.0]), \
         patch("book_see_rag.vectorstore.milvus_store._get_collection", return_value=col):
        hits = search_hits("hello", doc_ids=["d1", "d2"])

    assert len(hits) == 2
    assert hits[0]["content"] == "Hello world"
    assert hits[0]["doc_id"] == "d1"
    assert hits[1]["doc_id"] == "d2"


def test_get_doc_hits_dedupes_by_doc_page_content():
    from book_see_rag.vectorstore.milvus_store import get_doc_hits

    rows = [
        {"doc_id": "d1", "filename": "a.pdf", "page": 1, "content": "  Same   content "},
        {"doc_id": "d1", "filename": "a.pdf", "page": 1, "content": "Same content"},
        {"doc_id": "d2", "filename": "b.pdf", "page": 2, "content": "Other"},
    ]
    col = _FakeCollection(query_rows=rows)

    with patch("book_see_rag.vectorstore.milvus_store._get_collection", return_value=col):
        hits = get_doc_hits(["d1", "d2"], limit=10)

    assert len(hits) == 2
    assert hits[0]["content"] == "Same content"
    assert hits[1]["doc_id"] == "d2"

