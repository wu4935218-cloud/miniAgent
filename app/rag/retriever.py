from functools import lru_cache

from app.rag import bm25_retriever
from app.rag.hybrid_retriever import reciprocal_rank_fusion
from app.rag.models import SearchResult
from app.rag.chroma_store import ChromaVectorStore
from pathlib import Path
from app.rag.bm25_retriever import BM25Retriever
from app.rag.loader import load_all_documents
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

@lru_cache(maxsize=1)
def get_bm25_retriever():
    chunks = load_all_documents(DATA_DIR)
    return BM25Retriever(chunks)


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()

def retrieve(
        query: str,
        top_k: int = 3,
        min_score: float = 0.4
) -> list[SearchResult]:
    store = get_vector_store()
    return store.search(query, top_k, min_score)

def hybrid_retrieve(
        query: str,
        top_k: int = 3
) -> list[SearchResult]:
    vector_results = retrieve(query,top_k=10,min_score=0.0)
    bm25_results = (
        get_bm25_retriever().search(query,top_k=10)
    )
    return reciprocal_rank_fusion(
        [
            vector_results,
            bm25_results
        ],
        top_k=top_k
    )

if __name__ == "__main__":
    queries = [
        "asyncio.gather 有什么作用？",
        "怎样同时等待多个协程完成？",
        "RAG 全称是什么？"
    ]

    for query in queries:
        print("\n======", query, "======")
        result = hybrid_retrieve(query)
        print(result)
    # results = retrieve( "为什么 Agent 适合使用异步编程？")
    # results = retrieve("姚明多高")
    # for result in results:
    #     print(
    #         result.score,
    #         result.chunk.content,
    #         result.chunk.metadata
    #     )
    # questions = [
    #     "Agent 怎么使用外部工具？",
    #     "Embedding 是什么？",
    #     "北京今天气温是多少？"
    # ]
    # for question in questions:
    #     print("\nquestion:",question)
    #     results = retrieve(question,3,0.0)
    #     for document,score in results:
    #         print(f"{score:.4f} {document} ")
