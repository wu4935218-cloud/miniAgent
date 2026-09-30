import numpy as np
from app.rag.embedding import embed_query,embed_texts

documents = [
    "AI Agent 是能够自主完成任务的人工智能系统。",
    "Agent 可以通过 Tool Calling 使用外部工具。",
    "RAG 通过检索外部知识增强大模型回答。",
    "Embedding 可以将文本转换为向量。",
    "Agent Memory 用来保存与任务相关的历史信息。"
]
document_embedding = embed_texts(documents)

def retrieve(
        query: str,
        top_k: int = 3
) -> list[str]:
    query_embeddings = embed_query(query)
    scores = document_embedding @ query_embeddings
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        (
            documents[index],
            float(scores[index])
        )
        for index in indices
    ]

# print(retrieve( "Agent 怎么调用外部功能？"))