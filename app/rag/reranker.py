from sentence_transformers import CrossEncoder

from app.rag.models import SearchResult


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")
    def rerank(
            self,
            query: str,
            results: list[SearchResult],
            top_k: int = 3
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [
            [query,result.chunk.content]
            for result in results
        ]
        scores = self.model.predict(pairs)
        reranked = [
            SearchResult(
                chunk=result.chunk,
                score=float(score)
            )
            for result,score in zip(results,scores)
        ]
        reranked.sort(
            key=lambda result:result.score,
            reverse=True
        )
        return reranked[:top_k]