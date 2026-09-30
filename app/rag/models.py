from typing import Any

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    content: str
    metadata: dict[str,Any]

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float