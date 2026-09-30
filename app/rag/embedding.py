from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5"
)
texts = [
    "AI Agent 可以调用工具。",
    "RAG 可以检索外部知识。",
    "北京今天很晴朗。"
]
embeddings = model.encode(texts,normalize_embeddings=True)
query = "智能体怎样使用外部工具？"
query_embedding = model.encode(query,normalize_embeddings=True)
scores = embeddings @ query_embedding
for text,score in zip(texts,scores):
    print(f"{text}: {score:.4f}")

# print(embeddings.shape)

