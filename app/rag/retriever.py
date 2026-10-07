from functools import lru_cache
from app.rag.indexer import build_index
from app.rag.models import SearchResult
from app.rag.vector_store import SimpleVectorStore
from app.rag.chroma_store import ChromaVectorStore


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

if __name__ == "__main__":
    queries = [
        "asyncio.gather 有什么作用？",
        "RAG 是什么？",
        "姚明多高？"
    ]

    for query in queries:
        print("\n======", query, "======")
        result = retrieve(query)
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
