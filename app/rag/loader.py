from pathlib import Path
from app.rag.chunker import split_text
from app.rag.models import DocumentChunk

def load_text_file(file_path: str | Path) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")

def load_and_split(
        file_path: str | Path,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
) -> list[DocumentChunk]:
    path = Path(file_path)
    text = load_text_file(path)
    texts = split_text(text, chunk_size, chunk_overlap)
    return [
        DocumentChunk(
            content=chunk,
            metadata={
                "source":path.name,
                "chunk_index":index
            }
        )
        for index,chunk in enumerate(texts)
    ]

if __name__ == "__main__":
    load_text = load_and_split("../../data/agent_knowledge.txt")
    print(load_text)