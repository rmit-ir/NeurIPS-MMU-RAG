"""
MMU-RAG Challenge API implementation with /evaluate and /run endpoints.
"""

import asyncio
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from systems.rag_interface import (
    EvaluateResponse,
    RAGInterface,
    EvaluateRequest,
    RunRequest,
)
from systems.rag_router.rag_router_query_complexity import RAGRouterQueryComplexity
from tools.llm_servers.vllm_server import get_llm_mgr
from tools.logging_utils import get_logger
from tools.reranker_vllm import get_reranker
from tools.responses.mmu_rag_stream import to_mmu_rag_stream


# Create router for MMU-RAG endpoints
router = APIRouter(prefix="", tags=["MMU-RAG Challenge"])

logger = get_logger('mmu_rag_router')


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Launch llm
    logger.info("Starting reranker_vllm reranker and LLM manager in parallel...")
    default_llm_mgr = get_llm_mgr(
        model_id="Qwen/Qwen3-4B",
        reasoning_parser="qwen3",
        gpu_memory_utilization=0.8,
        max_model_len=20000)

    # Run both initialization tasks concurrently
    await asyncio.gather(
        get_reranker(),
        default_llm_mgr.get_server()
    )

    logger.info("Reranker and LLM manager ready.")
    yield

# Create app for standalone usage
app = FastAPI(title="MMU-RAG Challenge API",
              version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instances (to be set by the application)
rag_system: RAGInterface = RAGRouterQueryComplexity()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "systems": [rag_system.name]}


@router.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest) -> EvaluateResponse:
    try:
        # Process the request using Pydantic model directly
        response = await rag_system.evaluate(request)
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/run")
async def run_endpoint(request: RunRequest):
    """
    Streaming /run endpoint for RAG-Arena live evaluation.

    See https://agi-lti.github.io/MMU-RAGent/text-to-texct for API details.
    """
    try:
        # Use the streaming utility to convert RAG responses to MMU-RAG format
        run = await rag_system.run_streaming(request)
        return StreamingResponse(
            to_mmu_rag_stream(run),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error setting up stream: {str(e)}")

app.include_router(router)

if __name__ == "__main__":
    print("Run\nuv run fastapi run src/apis/mmu-rag-router.py")
    pass
