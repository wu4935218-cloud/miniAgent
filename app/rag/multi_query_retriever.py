from app.rag.hybrid_retriever import hybrid_retrieve,reciprocal_rank_fusion
from app.rag.models import SearchResult
from app.rag.query_rewriter import rewrite_query
import asyncio

from app.rag.rerank_retrieve import get_reranker


async def multi_query_retrieve(
        query: str,
        top_k: int = 10,
        queries: list[str] | None = None
) -> list[SearchResult]:
    if queries is None:
        queries = await rewrite_query(query)
    tasks = [
        asyncio.to_thread(
            hybrid_retrieve,
            rewritten_query,
            10
        )
        for rewritten_query in queries
    ]
    result_lists = await asyncio.gather(*tasks)
    return reciprocal_rank_fusion(result_lists=result_lists,top_k=top_k)

async def multi_query_rerank_retrieve(
        query: str,
        top_k: int = 3
) -> list[SearchResult]:
    candidates = await multi_query_retrieve(query=query,top_k=10)
    reranker = get_reranker()
    return await asyncio.to_thread(
        reranker.rerank,
        query,
        candidates,
        top_k
    )

def print_results(title:str,results:list[SearchResult]) -> None:
    print(f"\n======{title}======")
    if not results:
        print("没有检索结果")
        return
    for rank,result in enumerate(results,start=1):
        print(f"\nrank: {rank}")
        print(f"score: {result.score:.6f}")
        print("source:",result.chunk.metadata["source"])
        print("chunk:",result.chunk.metadata["chunk_index"])
        print("content:",result.chunk.content)

if __name__ == "__main__":
    queries = [
        "模型不知道外部资料的时候怎么办？"
    ]
    async def main():
        for query in queries:
            print("\n======query======")
            print(query)
            generates_queries = await rewrite_query(query)
            print("\n======generates_queries======")
            for index,rewritten_query in enumerate(generates_queries,start=1):
                print(f"{index},{rewritten_query}")
            hybrid_results = hybrid_retrieve(query=query,top_k=3)
            print_results("hybrid original top3",hybrid_results)
            multi_query_results = (
                await multi_query_retrieve(
                    query=query,
                    top_k=3,
                    queries=generates_queries
                )
            )
            print_results("multi-query top3",multi_query_results)
            candidates = await multi_query_retrieve(
                query=query,
                top_k=10,
                queries=generates_queries
            )
            reranker = get_reranker()
            reranker_results = await asyncio.to_thread(
                reranker.rerank,
                query,
                candidates,
                3
            )
            print_results("reranker top3",reranker_results)
    asyncio.run(main())

