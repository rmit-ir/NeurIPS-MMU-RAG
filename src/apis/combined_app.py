"""
Combined API app implementation.
Combines both OpenAI-compatible and MMU-RAG Challenge APIs into a single application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from both modules
from apis.openai_router import router as openai_router
from apis.mmu_rag_router import router as mmu_router


# Create the combined app
app = FastAPI(
    title="MMU RAG Combined API",
    version="1.0.0",
    description="Combined OpenAI-compatible and MMU-RAG Challenge API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include both routers
app.include_router(openai_router)
app.include_router(mmu_router)

if __name__ == "__main__":
    print("Run\nuv run fastapi run src/apis/combined_app.py")
    pass
