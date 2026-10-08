from functools import lru_cache
from pathlib import Path
from rank_bm25 import BM25Okapi
from app.rag.loader import load_all_documents
from app.rag.models import DocumentChunk,SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

@lru_cache(maxsize=1)
def get_bm25_retriever():
    chunks = load_all_documents(DATA_DIR)
    return BM25Retriever(chunks)

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
        self.bm25 = BM25Okapi(tokenized_corpus)# 整个知识库建立一次关键词索引
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

def bm25_retrieve(
        query: str,
        top_k: int = 10
) -> list[SearchResult]:
    return get_bm25_retriever().search(query,top_k=top_k)

if __name__ == "__main__":
    queries = [
        "asyncio.gather 有什么作用？",
        "怎样同时等待多个协程完成？",
        "RAG 全称是什么？"
    ]

    for query in queries:
        print("\n======", query, "======")
        results = bm25_retrieve(query, top_k=3)
        for rank,result in enumerate(results,start=1):
            print(f"\nRank:{rank}")
            print("score:",result.score)
            print("source:",result.chunk.metadata["source"])
            print("content:",result.chunk.content)