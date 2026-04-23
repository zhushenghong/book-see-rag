from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from book_see_rag.api.routes import ingest, documents, query, chat, knowledge_bases

app = FastAPI(
    title="book-see-rag",
    description="生产级通用 RAG 文档分析系统",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router,    prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(query.router,     prefix="/api")
app.include_router(chat.router,      prefix="/api")
app.include_router(knowledge_bases.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
