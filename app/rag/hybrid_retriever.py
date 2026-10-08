from collections import defaultdict
from app.rag.models import SearchResult
from app.rag.bm25_retriever import bm25_retrieve
from app.rag.vector_retriever import vector_retrieve


def make_chunk_key(result: SearchResult) -> str:
    source = result.chunk.metadata["source"]
    chunk_index = result.chunk.metadata["chunk_index"]
    return f"{source}-{chunk_index}"
def reciprocal_rank_fusion(
        result_lists: list[list[SearchResult]],
        top_k: int = 3,
        rrf_k: int = 60
) -> list[SearchResult]:
    scores = defaultdict(float)
    result_map = {}
    for results in result_lists:
        for rank,result in enumerate(results,start=1):
            key = make_chunk_key(result)
            scores[key] += (1/(rrf_k + rank))
            result_map[key] = result
    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )[:top_k]

    return [
        SearchResult(
            chunk=result_map[key].chunk,
            score=scores[key]
        )
        for key in ranked_keys
    ]

def hybrid_retrieve(
        query: str,
        top_k: int = 3
) -> list[SearchResult]:
    vector_results = vector_retrieve(query,top_k=10,min_score=0.0)
    bm25_results = bm25_retrieve(query,top_k=10)
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
        results = hybrid_retrieve(query, top_k=10)
        for rank,result in enumerate(results,start=1):
            print(f"\nRank:{rank}")
            print("score:",result.score)
            print("source:",result.chunk.metadata["source"])
            print("content:",result.chunk.content)