# app/main.py
from fastapi import FastAPI
from app.db.database import engine
from app.db.base import Base
from app.api.trace_api import router as trace_router
from app.api.agent_api import router as agent_router

app = FastAPI(title="AgentScope API", version="1.0.0")

# Include routers
app.include_router(trace_router, prefix="/api", tags=["trace"])
app.include_router(agent_router, prefix="/api", tags=["agent"])

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created successfully")

@app.get("/")
async def root():
    return {"message": "AgentScope API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}