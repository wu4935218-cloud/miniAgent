import asyncio
from app.core.config import client
from app.rag.retriever import retrieve

async def rag_answer(
        question: str
) -> str:
    chunks = await asyncio.to_thread(retrieve, question)
    context = "\n\n".join(chunks)
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
print(asyncio.run(rag_answer("Agent 怎么使用外部工具？")))
# print(asyncio.run(rag_answer("什么是 Embedding？")))
# print(asyncio.run(rag_answer("北京天气怎么样？")))
