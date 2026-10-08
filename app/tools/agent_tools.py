from pydantic import BaseModel, Field
import asyncio

from app.rag.vector_retriever import retrieve

tools = []
tool_registry = {}
def tool(
        name: str,
        description: str,
        args_model: type[BaseModel]
):
    def decorator(func):
        tool_registry[name] = {
            "function": func,
            "args_model": args_model
        }
        tools.append(
            {
                "type":"function",
                "function":{
                    "name": name,
                    "description": description,
                    "parameters": args_model.model_json_schema()
                }
            }
        )
        return func
    return decorator
class CalculatorArgs(BaseModel):
    a:int = Field(
        ge=-10000,
        le=10000,
        description="第一个整数参数"
    )
    b:int = Field(
        ge=-10000,
        le=10000,
        description="第二个整数参数"
    )
class SearchArgs(BaseModel):
    query:str = Field(
        min_length=1,
        max_length=1000,
        description="搜索关键词"
    )
class GetWeatherArgs(BaseModel):
    city:str = Field(
        min_length=1,
        description="要查询天气的城市名称"
    )
class KnowledgeSearchArgs(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="要在内部知识库中检索的问题或关键词"
    )
@tool(
    name="search_knowledge_base",
    description=(
        "搜索内部知识库。"
        "当用户询问 Agent、RAG、Embedding、"
        "Tool Calling、Memory 等内部知识时使用。"
    ),
    args_model=KnowledgeSearchArgs
)
async def search_knowledge_base(query: str) -> str:
    results = await asyncio.to_thread(retrieve,query)
    if not results:
        return "知识库中没有找到足够相关的信息。"
    return "\n\n".join(
        (
            f"[来源：{result.chunk.metadata['source']},"
            f"Chunk:{result.chunk.metadata['chunk_index']}]\n"
            f"{result.chunk.content}"
        )
        for result in results
    )
@tool(
    name="calculator",
    description="计算两个整数的加法",
    args_model=CalculatorArgs
)
def calculator(a:int,b:int) -> int:
    return a+b
@tool(
    name="search",
    description="搜索关键词",
    args_model=SearchArgs
)
async def search(query:str) -> str:
    await asyncio.sleep(1)
    return f"搜到了关于{query}的资料"
@tool(
    name="get_weather",
    description="查询天气",
    args_model=GetWeatherArgs
)
async def get_weather(city:str) -> str:
    await asyncio.sleep(2)
    return f"{city}当前天气：晴，25℃"