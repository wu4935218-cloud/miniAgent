from pathlib import Path

from app.rag.chroma_store import ChromaVectorStore
from app.rag.loader import load_all_documents


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def ingest() -> None:
    chunks = load_all_documents(DATA_DIR)

    store = ChromaVectorStore()
    store.add(chunks)

    print(f"成功写入 {len(chunks)} 个 chunks")


if __name__ == "__main__":
    ingest()