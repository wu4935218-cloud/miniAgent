from functools import lru_cache

from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.models import SearchResult
from app.rag.reranker import Reranker


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
def rerank_retrieve(
        query: str,
        top_k: int = 3
) -> list[SearchResult]:
    candidates = hybrid_retrieve(query,10)
    reranker = get_reranker()
    return reranker.rerank(
        query=query,
        results=candidates,
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
        results = rerank_retrieve(query, top_k=3)
        for rank,result in enumerate(results,start=1):
            print(f"\nRank:{rank}")
            print("score:",result.score)
            print("source:",result.chunk.metadata["source"])
            print("content:",result.chunk.content)