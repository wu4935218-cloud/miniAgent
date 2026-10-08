from functools import lru_cache
from app.rag.models import SearchResult
from app.rag.chroma_store import ChromaVectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()

def vector_retrieve(
        query: str,
        top_k: int = 3,
        min_score: float = 0.4
) -> list[SearchResult]:
    store = get_vector_store()
    return store.search(query, top_k, min_score)

if __name__ == "__main__":
    queries = [
        "asyncio.gather 有什么作用？",
        "怎样同时等待多个协程完成？",
        "RAG 全称是什么？"
    ]

    for query in queries:
        print("\n======", query, "======")
        results = vector_retrieve(query, top_k=3, min_score=0.0)
        for rank, result in enumerate(results, start=1):
            print(f"\nRank:{rank}")
            print("score:", result.score)
            print("source:", result.chunk.metadata["source"])
            print("content:", result.chunk.content)