from pathlib import Path

import chromadb
from app.rag.embedding import embed_query,embed_texts
from app.rag.loader import load_all_documents
from app.rag.models import DocumentChunk, SearchResult

client = chromadb.PersistentClient(
    path="./chroma_db"
)
collection = client.get_or_create_collection(
    name="mini_agent_knowledge"
)

class ChromaVectorStore:
    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embedding = None
    def add(self,chunks:list[DocumentChunk]) -> None:
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

    def search(self,query:str,top_k:int = 3,min_score:float = 0.55) -> list[SearchResult]:
        query_embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )
        print(results)#返回的是distances越小越好

if __name__ == "__main__":
    chromaStore = ChromaVectorStore()
    path = Path("../../data")
    chunks = load_all_documents(path)
    chromaStore.add(chunks)
    query = "asyncio.gather 有什么作用？"
    chromaStore.search(query)