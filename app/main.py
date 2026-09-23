from fastapi import FastAPI
from app.api.agent import router as agent_router

app = FastAPI(
    title="mini agent",
    version="0.1.0"
)

app.include_router(agent_router)

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

