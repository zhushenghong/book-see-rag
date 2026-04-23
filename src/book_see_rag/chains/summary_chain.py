from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import time
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_core.documents import Document
from book_see_rag.llm.factory import create_llm
from book_see_rag.vectorstore.milvus_store import search
from book_see_rag.embedding.reranker import rerank
from book_see_rag.config import get_settings

logger = logging.getLogger("uvicorn.error")

_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "请对以下文档片段进行简洁摘要，提取核心要点：\n\n{context}"),
    ("human", "请生成摘要"),
])

_REDUCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "以下是多个文档片段的摘要，请将它们整合成一份连贯、完整的总结：\n\n{context}"),
    ("human", "请生成最终总结"),
])


def summarize(doc_ids: list[str] | None = None, topic: str = "文档的主要内容") -> str:
    """
    摘要：检索相关 chunks → MapReduce 生成摘要
    """
    settings = get_settings()
    llm = create_llm()

    started = time.perf_counter()
    t0 = time.perf_counter()
    candidates = search(topic, doc_ids=doc_ids)
    t1 = time.perf_counter()
    if settings.enable_rerank:
        ranked_chunks = rerank(topic, candidates, top_k=settings.rerank_top_k)
    else:
        ranked_chunks = candidates[:settings.rerank_top_k]
        logger.info("Summary rerank disabled using top_k=%s from search results", len(ranked_chunks))
    t2 = time.perf_counter()
    documents = [Document(page_content=c) for c in ranked_chunks]

    map_chain = LLMChain(llm=llm, prompt=_MAP_PROMPT)
    reduce_chain = LLMChain(llm=llm, prompt=_REDUCE_PROMPT)

    reduce_docs_chain = ReduceDocumentsChain(
        combine_documents_chain=create_stuff_documents_chain(llm, _REDUCE_PROMPT),
        collapse_documents_chain=create_stuff_documents_chain(llm, _REDUCE_PROMPT),
        token_max=3000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_docs_chain,
        document_variable_name="context",
        return_intermediate_steps=False,
    )

    result = map_reduce_chain.invoke({"input_documents": documents})
    t3 = time.perf_counter()
    logger.info(
        "Summary timings search=%.3fs rerank=%.3fs llm=%.3fs total=%.3fs candidates=%s ranked=%s",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t3 - started,
        len(candidates),
        len(ranked_chunks),
    )
    return result["output_text"]
