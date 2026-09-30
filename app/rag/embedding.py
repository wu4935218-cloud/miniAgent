from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5"
)

def embed_texts(
        texts: list[str]
):
    return model.encode(texts, normalize_embeddings=True)

def embed_query(
        query: str
):
    return model.encode(query, normalize_embeddings=True)

