import asyncio
from app.core.config import client
from app.rag.retriever import retrieve

async def rag_answer(
        question: str
) -> str:
    results = await asyncio.to_thread(retrieve, question)
    if not results:
        return "当前知识库中没有足够信息。"
    context = "\n\n".join(
        f"[来源:{result.chunk.metadata['source']}"
        f"Chunk:{result.chunk.metadata['chunk_index']}]"
        f"{result.chunk.content}"
        for result in results
    )
    messages = [
        {
            "role": "system",
            "content":(
                "你是一个知识库问答助手。"
                "请优先根据提供的资料回答。"
                "如果资料无法回答，请明确说明资料不足。"
            )
        },
        {
            "role":"user",
            "content":f"""
                参考资料：
                {context}
                用户问题：
                {question}
            """
        }
    ]
    response = await client.chat.completions.create(
        model="deepseek-flash",
        messages=messages
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    # print(asyncio.run(rag_answer("Agent 怎么使用外部工具？")))
    # print(asyncio.run(rag_answer("什么是 Embedding？")))
    print(asyncio.run(rag_answer("北京天气怎么样？")))
