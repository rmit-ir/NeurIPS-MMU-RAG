"""
OpenAI-compatible API router implementation.
Implements /v1/models and /v1/chat/completions endpoints.
"""

import os
from typing import List, Dict, Optional, Union
import uuid
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from systems.decomposition_rag.decomposition_rag import DecompositionRAG
from systems.rag_interface import RAGInterface, RunRequest
from systems.vanilla_agent.vanilla_rag import VanillaRAG
from tools.llm_servers.sglang_server import get_llm_server
from tools.responses.openai_stream import to_openai_stream


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Launch llm
    default_llm_server = get_llm_server()
    await default_llm_server.get_server()
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


# Global RAG system instance
rag_systems: Dict[str, RAGInterface] = {
    "vanilla-rag": VanillaRAG(),
    "decomposition-rag": DecompositionRAG(),
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
            # Streaming response
            run = await model.run_streaming(run_request)
            return StreamingResponse(
                to_openai_stream(run, model=request.model),
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
