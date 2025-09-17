"""
DEPRECATED: Combined API app implementation.

⚠️  WARNING: This file is deprecated and will be removed in a future version.
    Please use the individual API servers instead:
    
    - For MMU-RAG Challenge API: src/apis/mmu_rag_router.py
    - For OpenAI-compatible API: src/apis/openai_router.py
    
    Run them separately:
    uv run fastapi run src/apis/mmu_rag_router.py
    uv run fastapi run src/apis/openai_router.py

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
app.include_router(openai_router, prefix="/openai")
app.include_router(mmu_router)

if __name__ == "__main__":
    print("Deprecated: Please run the individual API servers instead.")
    print("Run\nuv run fastapi run src/apis/combined_app.py")
    pass
