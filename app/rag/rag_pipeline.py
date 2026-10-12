from pydantic import BaseModel
from app.rag.models import SearchResult
from app.rag.vector_retriever import vector_retrieve
from app.rag.contextual_query_rewriter import ChatMessage
from app.rag.conversation_retriever import conversation_retrieve
from app.core.config import client
import asyncio
class RAGPipelineResult(BaseModel):
    original_question: str
    standalone_query: str
    rewritten_queries: list[str]
    results: list[SearchResult]
    context: str
    has_evidence: bool

def has_sufficient_evidence(
        query: str,
        min_score: float = 0.4
) -> bool:
    results = vector_retrieve(
        query=query,
        top_k=1,
        min_score=0.0
    )
    if not results:
        return False
    return results[0].score >= min_score
def build_context(
        results: list[SearchResult]
) -> str:
    return "\n\n".join(
        (
            f"[来源:{result.chunk.metadata['source']}"
            f"Chunk:{result.chunk.metadata['chunk_index']}]"
            f"{result.chunk.content}"
        )
        for result in results
    )
async def run_rag(
        question: str,
        history: list[ChatMessage],
        top_k: int = 3
) -> RAGPipelineResult:
    retrieval_result = await conversation_retrieve(
        question=question,
        history=history,
        top_k=top_k
    )
    has_evidence = await asyncio.to_thread(has_sufficient_evidence, retrieval_result.standalone_query)
    if not has_evidence:
        return RAGPipelineResult(
            original_question=question,
            standalone_query=retrieval_result.standalone_query,
            rewritten_queries=retrieval_result.rewritten_queries,
            results=[],
            context="",
            has_evidence=False
        )
    context = build_context(retrieval_result.final_results)
    return RAGPipelineResult(
        original_question=question,
        standalone_query=retrieval_result.standalone_query,
        rewritten_queries=retrieval_result.rewritten_queries,
        results=retrieval_result.final_results,
        context=context,
        has_evidence=True
    )

async def answer_with_rag(
        question: str,
        history: list[ChatMessage]
) -> str:
    rag_result = await run_rag(question, history)
    if not rag_result.has_evidence:
        return "当前知识库中没有足够信息。"
    response = await client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个知识库问答助手。"
                    "必须严格根据提供的知识库上下文回答。"
                    "不要使用上下文之外的事实。"
                    "如果上下文无法支持某个结论，"
                    "不要自行编造。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"知识库上下文：\n"
                    f"{rag_result.context}\n\n"
                    f"用户问题：\n"
                    f"{question}"
                )
            }
        ]
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    import asyncio
    async def main():
        history1 = [
            ChatMessage(role="user",content="RAG是什么？"),
            ChatMessage(role="assistant",content="RAG是检索增强生成。")
        ]
        question1 = "它是怎么获得外部知识的？"
        history = []
        question = "什么是混合专家？"
        result = await run_rag(question, history)
        print("\n====== Original Question ======")
        print(result.original_question)
        print("\n====== Standalone Query ======")
        print(result.standalone_query)
        print("\n====== Rewritten Queries ======")
        for index,query in enumerate(result.rewritten_queries,start=1):
            print(f"{index}:{query}")
        print("\n====== Has Evidence ======")
        print(result.has_evidence)
        print("\n====== Context ======")
        print(result.context)
    asyncio.run(main())

