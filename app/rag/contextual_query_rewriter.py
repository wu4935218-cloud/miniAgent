from typing import Literal
from pydantic import BaseModel,ValidationError
from app.core.config import client
class ChatMessage(BaseModel):
    role: Literal["user","assistant"]
    content: str

class StandaloneQueryResult(BaseModel):
    query: str

async def contextualize_query(
        question: str,
        history: list[ChatMessage]
) -> str:
    if not history:
        return question
    recent_history = history[-6:]
    history_text = "\n".join(
        f"{message.role}:{message.content}"
        for message in recent_history
    )
    response = await client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role":"system",
                "content":(
                    "你是 RAG 系统中的上下文查询改写器。"
                    "你的任务不是回答用户问题。"
                    "请根据聊天历史，把当前问题改写成一个"
                    "不依赖聊天历史也能够独立理解的问题。"
                    "需要解决代词、省略和上下文指代。"
                    "如果当前问题本身已经完整，就保持原意。"
                    "不要加入聊天历史中不存在的新事实。"
                    '只输出 JSON，例如：'
                    '{"query":"改写后的独立问题"}'
                )
            },
            {
                "role":"user",
                "content":(
                    f"聊天历史：\n{history_text}\n"
                    f"当前问题：{question}\n"
                    "请把当前问题改写成一个不依赖聊天历史也能够独立理解的问题。"
                )
            }
        ]
    )
    content = response.choices[0].message.content
    if not content:
        return question
    try:
        result = StandaloneQueryResult.model_validate_json(content)
        standalone_query = result.query.strip()
        return standalone_query or question
    except ValidationError:
        return question

if __name__ == "__main__":
    import asyncio
    async def main():
        history1 = [
            ChatMessage(role="user",content="RAG是什么？"),
            ChatMessage(role="assistant",content="RAG是检索增强生成。")
        ]
        question1 = "它有什么优势？"
        history2 = [
            ChatMessage(
                role="user",
                content="Tool Calling 是什么？"
            ),
            ChatMessage(
                role="assistant",
                content="Tool Calling 可以让模型调用外部工具。"
            )
        ]

        question2 = "这个机制是怎么执行 Python 函数的？"
        history3 = [
            ChatMessage(
                role="user",
                content="asyncio.gather 有什么作用？"
            ),
            ChatMessage(
                role="assistant",
                content="它可以并发等待多个协程完成。"
            )
        ]

        question3 = "和 await 有什么区别？"
        history = [
            ChatMessage(
                role="user",
                content="RAG 是什么？"
            ),
            ChatMessage(
                role="assistant",
                content="RAG 是检索增强生成。"
            )
        ]
        question = "asyncio.gather 有什么作用？"
        standalone_query = await contextualize_query(question,history)
        print(f"Current Question:{question}")
        print(f"\nStandalone Query: {standalone_query}")
    asyncio.run(main())


