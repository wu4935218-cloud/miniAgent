from rank_bm25 import BM25Okapi
from app.rag.models import DocumentChunk,SearchResult
def tokenize(text: str) -> list[str]:
    import jieba
    return [
        token.strip().lower()
        for token in jieba.cut(text)
        if token.strip()
    ]

class BM25Retriever:
    def __init__(self,chunks: list[DocumentChunk]):
        self.chunks = chunks
        tokenized_corpus = [
            tokenize(chunk.content)
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
    def search(self,query:str,top_k:int = 3) -> list[SearchResult]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i:scores[i],
            reverse=True
        )[:top_k]
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(scores[index])
            )
            for index in ranked_indices
            if scores[index] > 0
        ]