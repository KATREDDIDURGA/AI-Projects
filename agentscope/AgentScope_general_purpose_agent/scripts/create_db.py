from app.db.base import Base
from app.db.session import engine

# ✅ Make sure ALL models are imported here!
from app.db.models.agent_run import AgentRun
from app.db.models.agent_step import AgentStep

def create_tables():
    import asyncio
    async def run():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)  # Optional: resets old tables
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database and tables created successfully.")

    asyncio.run(run())

if __name__ == "__main__":
    create_tables()
