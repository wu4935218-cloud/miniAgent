from app.rag.models import SearchResult
from app.rag.vector_retriever import vector_retrieve
from app.rag.rerank_retrieve import rerank_retrieve
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.rag_pipeline import has_sufficient_evidence
test_cases = [
    {
        "query": "asyncio.gather 有什么作用？",
        "expected_source": "python_knowledge.txt",
        "expected_chunk_index": 0
    },
    {
        "query": "RAG 如何利用外部知识？",
        "expected_source": "agent_knowledge.txt",
        "expected_chunk_index": 0
    },
    {
        "query": "Agent 为什么适合处理 I/O 请求？",
        "expected_source": "agent_knowledge.txt",
        "expected_chunk_index": 2
    }
]
def is_expected(
        result: SearchResult,
        expected_source: str,
        expected_chunk_index: int
) -> bool:
    metadata = result.chunk.metadata
    return (
        metadata["source"] == expected_source
        and metadata["chunk_index"] == expected_chunk_index
    )

def hit_at_k(
        results:list[SearchResult],
        expected_source: str,
        expected_chunk_index: int,
        k:int
) -> bool:
    top_k_results = results[:k]
    return any(
        is_expected(
            result,
            expected_source,
            expected_chunk_index
        )
        for result in top_k_results
    )

def find_rank(
        results:list[SearchResult],
        expected_source:str,
        expected_chunk_index:int
) -> int | None:
    for rank,result in enumerate(results,start=1):
        if is_expected(
            result,expected_source,expected_chunk_index
        ):
            return rank
    return None

def reciprocal_rank(
        rank: int | None
) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank

def evaluate() -> None:
    hybrid_rr_sum = 0.0
    rerank_rr_sum = 0.0
    hybrid_hit_1 = 0
    hybrid_hit_3 = 0
    rerank_hit_1 = 0
    rerank_hit_3 = 0
    total = len(test_cases)
    for case in test_cases:
        query = case["query"]
        expected_source = case["expected_source"]
        expected_chunk_index = case["expected_chunk_index"]
        hybrid_results = hybrid_retrieve(query, top_k=10)
        rerank_results = rerank_retrieve(query, top_k=3)
        hybrid_rank = find_rank(
            hybrid_results,
            expected_source,
            expected_chunk_index
        )
        rerank_rank = find_rank(
            rerank_results,
            expected_source,
            expected_chunk_index
        )
        hybrid_rr_sum += reciprocal_rank(hybrid_rank)
        rerank_rr_sum += reciprocal_rank(rerank_rank)
        if hit_at_k(
            hybrid_results,
            expected_source,
            expected_chunk_index,
            1
        ):
            hybrid_hit_1 += 1
        if hit_at_k(
            hybrid_results,
            expected_source,
            expected_chunk_index,
            3
        ):
            hybrid_hit_3 += 1
        if hit_at_k(
            rerank_results,
            expected_source,
            expected_chunk_index,
            1
        ):
            rerank_hit_1 += 1
        if hit_at_k(
            rerank_results,
            expected_source,
            expected_chunk_index,
            3
        ):
            rerank_hit_3 += 1
        print(f"\n====={query}=====")
        print("Expected:",expected_source,expected_chunk_index)
        print("Hybrid Rank:",hybrid_rank)
        print("Rerank Rank:",rerank_rank)
        print(f"Hybrid MRR:{hybrid_rr_sum / total:.4f}")
        print(f"Rerank MRR:{rerank_rr_sum / total:.4f}")
    print("\n======Eval Result=====")
    print(f"Hybrid Hit@1:{hybrid_hit_1/total:.2%}")
    print(f"Hybrid Hit@3:{hybrid_hit_3/total:.2%}")
    print(f"Rerank Hit@1:{rerank_hit_1/total:.2%}")
    print(f"Rerank Hit@3:{rerank_hit_3/total:.2%}")

def evaluate_no_answer() -> None:
    negative_cases = [
        "姚明多高？",
        "法国首都是哪里？",
        "Java HashMap 的底层结构是什么？",
        "什么是混合专家模型？"
    ]
    rejected = 0
    for query in negative_cases:
        has_evidence = has_sufficient_evidence(query)
        print(f"\nquery:{query}")
        print(f"has_evidence:{has_evidence}")
        if not has_evidence:
            rejected += 1
    total = len(negative_cases)
    print("\n====== No-answer Eval ======")
    print(f"Rejected:{rejected/total:.2%}")

if __name__ == "__main__":
    evaluate()
    evaluate_no_answer()