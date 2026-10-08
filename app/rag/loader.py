from pathlib import Path
from app.rag.chunker import split_text, recursive_split_text
from app.rag.models import DocumentChunk

def load_text_file(file_path: str | Path) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")

def load_all_documents(data_dir: Path):
    all_chunks = []
    for file_path in data_dir.glob("*.txt"):
        if file_path.is_file():
            chunks = load_and_split(file_path,chunk_size=200,chunk_overlap=40)
            all_chunks.extend(chunks)
    return all_chunks

def load_and_split(
        file_path: str | Path,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
) -> list[DocumentChunk]:
    path = Path(file_path)
    text = load_text_file(path)
    # texts = split_text(text, chunk_size, chunk_overlap)# 按字符切分
    texts = recursive_split_text(text, chunk_size, None)# 按段落切分
    return [
        DocumentChunk(
            content=chunk,
            metadata={
                "source":path.name,
                "chunk_index":index,
                "char_count":len(chunk)
            }
        )
        for index,chunk in enumerate(texts)
    ]

if __name__ == "__main__":
    load_text = load_and_split("../../data/agent_knowledge.txt")
    print(load_text)