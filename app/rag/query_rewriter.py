from app.core.config import client
from pydantic import BaseModel, Field, ValidationError


class QueryRewriteResult(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=5
    )
async def rewrite_query(
        query: str,
        num_queries: int = 3
) -> list[str]:
    response = await client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role":"system",
                "content":(
                    "你是 RAG 查询改写器。"
                    "你的任务不是回答用户问题，"
                    "而是生成更适合知识库检索的查询。"
                    f"请生成 {num_queries} 个不同表达的查询。"
                    "保留原问题核心意图，不要加入未知事实。"
                    '只输出 JSON，例如：'
                    '{"queries":["查询1","查询2","查询3"]}'
                )
            },
            {
                "role":"user",
                "content":query
            }
        ]
    )
    content = response.choices[0].message.content
    queries = [query]
    if not content:
        return queries
    try:
        result = QueryRewriteResult.model_validate_json(content)#解析json字符串，生成QueryRewriteResult实例
    except ValidationError:
        return queries

    for rewritten_query in result.queries:
        rewritten_query = rewritten_query.strip()
        if rewritten_query and rewritten_query not in queries:
            queries.append(rewritten_query)
    return queries

if __name__ == "__main__":
    import asyncio
    queries = [
        "Agent 用异步到底好在哪？",
        "RAG 为什么能减少幻觉？",
        "这个 Tool Calling 怎么工作的？"
    ]
    async def main():
        for query in queries:
            print(f"\n======{query}======")
            rewritten = await rewrite_query(query)
            for index,item in enumerate(rewritten,start=1):
                print(index,item)
    asyncio.run(main())