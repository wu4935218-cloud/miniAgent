import numpy as np
from app.rag.embedding import embed_texts,embed_query
from app.rag.models import DocumentChunk,SearchResult

class SimpleVectorStore:
    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embedding = None
    def add(self,chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        texts = [chunk.content for chunk in chunks]
        new_embeddings = embed_texts(texts)
        if self.embedding is None:
            self.embedding = new_embeddings
        else:
            self.embedding = np.vstack([
                self.embedding,
                new_embeddings
            ])
        self.chunks.extend(chunks)

    def search(
            self,
            query: str,
            top_k: int = 3,
            min_score: float = 0.55
    ) -> list[SearchResult]:
        if self.embedding is None:
            return []
        query_embedding = embed_query(query)
        scores = self.embedding @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in indices:
            score = float(scores[index])
            if score < min_score:
                continue
            results.append(SearchResult(chunk=self.chunks[index],score=score))
        return results