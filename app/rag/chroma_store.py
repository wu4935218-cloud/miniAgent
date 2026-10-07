from pathlib import Path

import chromadb
from app.rag.embedding import embed_query,embed_texts
from app.rag.loader import load_all_documents
from app.rag.models import DocumentChunk, SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)
# collection = client.get_or_create_collection(
#     name="mini_agent_knowledge"
# )
collection = client.get_or_create_collection(
    name="mini_agent_knowledge_cosine",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

class ChromaVectorStore:
    def __init__(self):
        pass
    def add(self,chunks:list[DocumentChunk]) -> None:
        if not chunks:
            return
        texts = [chunk.content for chunk in chunks]
        embedding = embed_texts(texts)
        ids = []
        metadatas = []
        for chunk in chunks:
            source = chunk.metadata["source"]
            chunk_index = chunk.metadata["chunk_index"]
            ids.append(f"{source}-{chunk_index}")
            metadatas.append(chunk.metadata)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embedding.tolist(),
            metadatas=metadatas
        )

    def search(self,query:str,top_k:int = 3,min_score:float = 0.4) -> list[SearchResult]:
        query_embedding = embed_query(query)
        raw_results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )
        # print(results)#返回的是distances越小越好
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]
        results = []
        for document,metadata,distance in zip(documents,metadatas,distances):
            score = 1 - float(distance)
            if score < min_score:
                continue
            chunk = DocumentChunk(content=document,metadata=metadata)
            results.append(
                SearchResult(chunk=chunk,score=score)
            )
        return results

if __name__ == "__main__":
    chromaStore = ChromaVectorStore()
    queries = [
        "asyncio.gather 有什么作用？",
        "RAG 是什么？",
        "姚明多高？"
    ]

    for query in queries:
        print("\n======", query, "======")
        result = chromaStore.search(query)
        print(result)