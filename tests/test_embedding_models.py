from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


class _FakeArray:
    def __init__(self, payload):
        self._payload = payload

    def tolist(self):
        return self._payload


class _FakeFlagModel:
    def __init__(self, _model_name, use_fp16: bool, device: str):
        self.use_fp16 = use_fp16
        self.device = device

    def encode(self, texts, batch_size=32):
        return _FakeArray([[0.0] * 3 for _ in texts])

    def encode_queries(self, queries):
        return [_FakeArray([1.0, 2.0, 3.0]) for _ in queries]


class _FakeReranker:
    def __init__(self, _model_name, use_fp16: bool, device: str):
        self.use_fp16 = use_fp16
        self.device = device

    def compute_score(self, pairs, normalize=True):
        # score is based on chunk length to make ranking deterministic
        return [len(chunk) for _query, chunk in pairs]


def test_embedder_falls_back_to_cpu_when_preferred_device_fails():
    from book_see_rag.embedding import embedder

    embedder._load_model.cache_clear()

    settings = SimpleNamespace(embed_model="fake/bge", bge_device="cuda")
    calls = []

    def _factory(model_name, use_fp16: bool, device: str):
        calls.append((model_name, use_fp16, device))
        if device == "cuda":
            raise RuntimeError("cuda unavailable")
        return _FakeFlagModel(model_name, use_fp16=use_fp16, device=device)

    with patch("book_see_rag.embedding.embedder.get_settings", return_value=settings), \
         patch("book_see_rag.embedding.embedder.FlagModel", side_effect=_factory):
        vectors = embedder.embed_documents(["a", "b"])

    assert vectors == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert calls == [
        ("fake/bge", True, "cuda"),
        ("fake/bge", False, "cpu"),
    ]


def test_embed_query_uses_cached_embedding():
    from book_see_rag.embedding import embedder

    embedder._load_model.cache_clear()
    embedder._embed_query_cached.cache_clear()

    settings = SimpleNamespace(embed_model="fake/bge", bge_device="cpu")
    model = _FakeFlagModel(settings.embed_model, use_fp16=False, device="cpu")

    with patch("book_see_rag.embedding.embedder.get_settings", return_value=settings), \
         patch("book_see_rag.embedding.embedder.FlagModel", return_value=model) as mock_flag:
        v1 = embedder.embed_query("hello")
        v2 = embedder.embed_query("hello")

    assert v1 == v2 == [1.0, 2.0, 3.0]
    mock_flag.assert_called_once()


def test_reranker_ranks_chunks_by_score_and_falls_back_to_cpu():
    from book_see_rag.embedding import reranker

    reranker._load_reranker.cache_clear()

    settings = SimpleNamespace(reranker_model="fake/reranker", bge_device="cuda", rerank_top_k=2)

    def _factory(model_name, use_fp16: bool, device: str):
        if device == "cuda":
            raise RuntimeError("cuda unavailable")
        return _FakeReranker(model_name, use_fp16=use_fp16, device=device)

    with patch("book_see_rag.embedding.reranker.get_settings", return_value=settings), \
         patch("book_see_rag.embedding.reranker.FlagReranker", side_effect=_factory):
        ranked = reranker.rerank("q", ["a", "bbbb", "cc"], top_k=2)

    assert ranked == ["bbbb", "cc"]

