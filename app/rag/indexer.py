from app.rag.loader import load_and_split
from app.rag.vector_store import SimpleVectorStore

def build_index() -> SimpleVectorStore:
    chunks = load_and_split(
        file_path="../../data/agent_knowledge.txt",
        chunk_size=200,
        chunk_overlap=40,
    )
    store = SimpleVectorStore()
    store.add(chunks)
    return store