from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="用户发送给Agent的消息"
    )

class ChatResponse(BaseModel):
    answer: str