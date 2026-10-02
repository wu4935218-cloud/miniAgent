from app.schemas.chat import ChatResponse,ChatRequest
from fastapi import APIRouter
from app.services.agent_service import run_agent


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)

@router.post(
    "/chat",
    response_model=ChatResponse# 定义模型返回给用户的格式
)
async def chat(request: ChatRequest) -> ChatResponse:
    answer = await run_agent(request.message)
    return ChatResponse(
        answer=answer
    )