from app.rag.loader import load_and_split
from app.rag.vector_store import SimpleVectorStore
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_FILE = (
    PROJECT_ROOT
    / "data"
    / "agent_knowledge.txt"
)

def build_index() -> SimpleVectorStore:
    chunks = load_and_split(
        file_path=KNOWLEDGE_FILE,
        chunk_size=200,
        chunk_overlap=40,
    )
    store = SimpleVectorStore()
    store.add(chunks)
    return store