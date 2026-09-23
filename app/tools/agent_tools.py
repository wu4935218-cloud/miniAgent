from pydantic import BaseModel, Field

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
def search(query:str) -> str:
    return f"搜到了关于{query}的资料"
@tool(
    name="get_weather",
    description="查询天气",
    args_model=GetWeatherArgs
)
def get_weather(city:str) -> str:
    return f"{city}当前天气：晴，25℃"