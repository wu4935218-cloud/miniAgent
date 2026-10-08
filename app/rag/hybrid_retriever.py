from collections import defaultdict
from app.rag.models import SearchResult

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