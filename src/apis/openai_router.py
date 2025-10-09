"""
OpenAI-compatible API router implementation.
Implements /v1/models and /v1/chat/completions endpoints.
"""

import os
import hashlib
from typing import List, Dict, Optional, Union
import uuid
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from apis.warmup import warmup_models
from systems.commercial.azure_o3_research import AzureO3ResearchRAG
from systems.commercial.perplexity_research import PerplexityResearchRAG
from systems.decomposition_rag.decomposition_rag import DecompositionRAG
from systems.rag_interface import RAGInterface, RunRequest
from systems.rag_router.rag_router_llm import RAGRouterLLM
# from systems.rag_router.rag_router_query_complexity import RAGRouterQueryComplexity
from systems.vanilla_agent.vanilla_agent import VanillaAgent
from systems.vanilla_agent.vanilla_rag import VanillaRAG
from tools.logging_utils import get_logger
from tools.responses.openai_stream import to_openai_stream

logger = get_logger('openai_router')


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await warmup_models()
    yield

# Create app for standalone usage
app = FastAPI(title="OpenAI-Compatible RAG API",
              version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router for OpenAI-compatible endpoints
router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])

# Authentication setup
API_KEY = os.getenv("API_KEY")
security = HTTPBearer(auto_error=False)


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> bool:
    """
    Verify API key if one is set in environment variables.
    If no API_KEY is set, skip authentication.
    """
    if API_KEY is None:
        # No API key configured, skip authentication
        return True

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide Authorization header with Bearer token."
        )

    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return True


def generate_chat_hash(question: str, model: str) -> str:
    cache_input = f"{model}:{question}"
    return hashlib.sha256(cache_input.encode('utf-8')).hexdigest()


# Global RAG system instance
rag_systems: Dict[str, RAGInterface] = {
    "vanilla-rag": VanillaRAG(),
    "vanilla-agent": VanillaAgent(),
    "decomposition-rag": DecompositionRAG(),
    # "rag-router-qc": RAGRouterQueryComplexity(),
    "rag-router-llm": RAGRouterLLM(),
    "perplexity-sonar": PerplexityResearchRAG(model="sonar"),
    "perplexity-sonar-pro": PerplexityResearchRAG(model="sonar-pro"),
    "perplexity-sonar-reasoning": PerplexityResearchRAG(model="sonar-reasoning"),
    "perplexity-sonar-reasoning-pro": PerplexityResearchRAG(model="sonar-reasoning-pro"),
    "perplexity-deep-research": PerplexityResearchRAG(model="sonar-deep-research"),
    "azure-o3-deep-research": AzureO3ResearchRAG(),
}


# OpenAI API Models
class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None


class ModelInfo(BaseModel):
    """OpenAI model information format."""
    id: str
    object: str = "model"
    created: int = 0  # we don't have a creation timestamp
    owned_by: str = "rmit-ir"


class ModelsResponse(BaseModel):
    """OpenAI models list response format."""
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    """OpenAI chat completion choice format."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """OpenAI chat completion usage format."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response format."""
    id: str
    object: str = "chat.completion"
    created: int = 1234567890
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


@router.get("/models")
async def list_models(authenticated: bool = Depends(verify_api_key)) -> ModelsResponse:
    """List available models (OpenAI-compatible endpoint)."""
    return ModelsResponse(
        data=[ModelInfo(id=f, owned_by='rmit-ir')
              for f in rag_systems.keys()]
    )


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authenticated: bool = Depends(verify_api_key)):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    """
    try:
        # Extract the user's question from the last message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400, detail="No user message found")
        model = rag_systems.get(request.model)
        if model is None:
            raise HTTPException(
                status_code=404, detail=f"Model {request.model} not found")

        question = user_messages[-1].content
        run_request = RunRequest(question=question)

        if request.stream:
            # Generate chat hash for caching
            chat_hash = generate_chat_hash(question, request.model)

            # Streaming response
            run = await model.run_streaming(run_request)
            return StreamingResponse(
                to_openai_stream(run, model=request.model,
                                 chat_hash=chat_hash),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                }
            )
        else:
            # Non-streaming response - use evaluate method
            from systems.rag_interface import EvaluateRequest

            eval_request = EvaluateRequest(query=question, iid="openai-chat")
            eval_response = await model.evaluate(eval_request)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=eval_response.generated_response
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=len(question.split()),
                    completion_tokens=len(
                        eval_response.generated_response.split()),
                    total_tokens=len(question.split()) +
                    len(eval_response.generated_response.split())
                )
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat completion: {str(e)}")


@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "openai-compatible-api",
        "systems": [ModelInfo(id=f, owned_by='rmit-ir')
                    for f in rag_systems.keys()]
    }

app.include_router(router)

if __name__ == "__main__":
    print("Run\nuv run fastapi run src/apis/openai-router.py")
    pass
