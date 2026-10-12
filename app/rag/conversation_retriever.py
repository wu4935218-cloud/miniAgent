import asyncio
from pydantic import BaseModel
from app.rag.contextual_query_rewriter import ChatMessage,contextualize_query
from app.rag.query_rewriter import rewrite_query
from app.rag.multi_query_retriever import multi_query_retrieve
from app.rag.rerank_retrieve import get_reranker
from app.rag.models import SearchResult
class ConversationRetrievalResult(BaseModel):
    current_question: str
    standalone_query: str
    rewritten_queries: list[str]
    candidates: list[SearchResult]
    final_results: list[SearchResult]
async def conversation_retrieve(
        question: str,
        history: list[ChatMessage],
        top_k: int = 3
) -> ConversationRetrievalResult:
    standalone_query = await contextualize_query(question, history)
    rewritten_queries = await rewrite_query(standalone_query)
    candidates = await multi_query_retrieve(standalone_query,10,rewritten_queries)
    reranker = get_reranker()
    final_results = await asyncio.to_thread(
        reranker.rerank,
        standalone_query,
        candidates,
        top_k
    )
    return ConversationRetrievalResult(
        current_question=question,
        standalone_query=standalone_query,
        rewritten_queries=rewritten_queries,
        candidates=candidates,
        final_results=final_results
    )

def print_results(
        title: str,
        results: list[SearchResult]
) -> None:
    print(f"\n======{title}======")
    if not results:
        print("没有结果")
        return
    for rank,result in enumerate(results,start=1):
        print(f"\nrank:{rank}")
        print(f"score:{result.score:.6f}")
        print("source:",result.chunk.metadata["source"])
        print("chunk:",result.chunk.metadata["chunk_index"])
        print("content:",result.chunk.content)
if __name__ == "__main__":
    import asyncio
    async def main():
        history = [
            ChatMessage(role="user", content="RAG是什么？"),
            ChatMessage(role="assistant", content="RAG是检索增强生成。")
        ]
        question = "它是怎么获得外部知识的？"
        result = await conversation_retrieve(
            question=question,
            history=history,
            top_k=3
        )
        print("\n====== Current Question ======")
        print(result.current_question)
        print("\n====== Standalone Query ======")
        print(result.standalone_query)
        print("\n====== Rewritten Queries ======")
        for index,query in enumerate(result.rewritten_queries,start=1):
            print(f"{index}:{query}")
        print_results("Candidates Top10",result.candidates)
        print_results("Final Reranked Top3",result.final_results)
    asyncio.run(main())