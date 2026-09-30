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
        top_k: int = 3,
        min_score: float = 0.5
) -> list[tuple[str, float]]:
    query_embeddings = embed_query(query)
    scores = document_embedding @ query_embeddings
    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in indices:
        score = float(scores[index])
        if score >= min_score:
            results.append((documents[index], score))
    return results

if __name__ == "__main__":
    # print(retrieve( "Agent 怎么调用外部功能？"))
    questions = [
        "Agent 怎么使用外部工具？",
        "Embedding 是什么？",
        "北京今天气温是多少？"
    ]
    for question in questions:
        print("\nquestion:",question)
        results = retrieve(question,3,0.0)
        for document,score in results:
            print(f"{score:.4f} {document} ")
