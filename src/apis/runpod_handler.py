"""
RunPod serverless handler for OpenAI-compatible API.
Supports /v1/models and /v1/chat/completions endpoints through RunPod's OpenAI routing.
"""

import runpod
import json
import os
import hashlib
import uuid
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime

from systems.commercial.azure_o3_research import AzureO3ResearchRAG
from systems.commercial.perplexity_research import PerplexityResearchRAG
from systems.decomposition_rag.decomposition_rag import DecompositionRAG
from systems.rag_interface import RAGInterface, RunRequest, EvaluateRequest
from systems.rag_router.rag_router_llm import RAGRouterLLM
from systems.vanilla_agent.lang_graph_agent import LangGraphAgent
from systems.vanilla_agent.vanilla_agent import VanillaAgent
from systems.vanilla_agent.vanilla_rag import VanillaRAG
from tools.logging_utils import get_logger
from apis.warmup import warmup_models
from tools.responses.openai_stream import to_openai_stream

logger = get_logger('runpod_handler')

# Global RAG systems - same as in openai_router.py
rag_systems: Dict[str, RAGInterface] = {
    "vanilla-rag": VanillaRAG(),
    "vanilla-agent": VanillaAgent(),
    "decomposition-rag": DecompositionRAG(),
    "langgraph-agent": LangGraphAgent(),
    "rag-router-llm": RAGRouterLLM(),
    "perplexity-sonar": PerplexityResearchRAG(model="sonar"),
    "perplexity-sonar-pro": PerplexityResearchRAG(model="sonar-pro"),
    "perplexity-sonar-reasoning": PerplexityResearchRAG(model="sonar-reasoning"),
    "perplexity-sonar-reasoning-pro": PerplexityResearchRAG(model="sonar-reasoning-pro"),
    "perplexity-deep-research": PerplexityResearchRAG(model="sonar-deep-research"),
    "azure-o3-deep-research": AzureO3ResearchRAG(),
}

# Authentication setup
API_KEY = os.getenv("API_KEY")

# Global initialization flag
_initialized = False


async def initialize_handler():
    """Initialize the handler by warming up models."""
    global _initialized
    if not _initialized:
        logger.info("Initializing RunPod handler and warming up models...")
        await warmup_models()
        _initialized = True
        logger.info("Handler initialization complete")


def verify_api_key(api_key: Optional[str] = None) -> bool:
    """Verify API key if one is set in environment variables."""
    if API_KEY is None:
        return True
    return api_key == API_KEY


def generate_chat_hash(question: str, model: str) -> str:
    """Generate a hash for caching purposes."""
    cache_input = f"{model}:{question}"
    return hashlib.sha256(cache_input.encode('utf-8')).hexdigest()


def create_models_response() -> Dict[str, Any]:
    """Create OpenAI-compatible models list response."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "rmit-ir"
            }
            for model_id in rag_systems.keys()
        ]
    }


async def create_chat_completion_response(openai_input: Dict[str, Any]) -> Dict[str, Any]:
    """Create non-streaming chat completion response."""
    try:
        # Extract required fields
        model_id = openai_input.get("model")
        messages = openai_input.get("messages", [])

        if not model_id:
            return {"error": "Model is required"}

        if model_id not in rag_systems:
            return {"error": f"Model {model_id} not found"}

        # Extract user question from messages
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return {"error": "No user message found"}

        question = user_messages[-1].get("content", "")
        if not question:
            return {"error": "Empty user message"}

        # Get RAG system and run evaluation
        rag_system = rag_systems[model_id]
        eval_request = EvaluateRequest(query=question, iid="runpod-chat")
        eval_response = await rag_system.evaluate(eval_request)

        # Create OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": eval_response.generated_response,
                        "citations": eval_response.citations,
                        "contexts": eval_response.contexts,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(question.split()),
                "completion_tokens": len(eval_response.generated_response.split()),
                "total_tokens": len(question.split()) + len(eval_response.generated_response.split())
            }
        }

        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        return {"error": f"Error processing chat completion: {str(e)}"}


async def create_streaming_chat_completion(openai_input: Dict[str, Any]):
    """Create streaming chat completion response."""
    try:
        # Extract required fields
        model_id = openai_input.get("model")
        messages = openai_input.get("messages", [])

        if not model_id:
            yield f"data: {json.dumps({'error': 'Model is required'})}\n\n"
            return

        if model_id not in rag_systems:
            yield f"data: {json.dumps({'error': f'Model {model_id} not found'})}\n\n"
            return

        # Extract user question from messages
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            yield f"data: {json.dumps({'error': 'No user message found'})}\n\n"
            return

        question = user_messages[-1].get("content", "")
        if not question:
            yield f"data: {json.dumps({'error': 'Empty user message'})}\n\n"
            return

        # Get RAG system and prepare streaming
        rag_system = rag_systems[model_id]
        run_request = RunRequest(question=question)
        chat_hash = generate_chat_hash(question, model_id)

        # Create start_stream function for to_openai_stream
        stream = await rag_system.run_streaming(run_request)

        # Use to_openai_stream to handle the streaming response
        async for chunk in to_openai_stream(stream, model=model_id, chat_hash=chat_hash):
            yield chunk

    except Exception as e:
        logger.error(f"Error in streaming chat completion: {str(e)}")
        error_chunk = {
            "error": {
                "message": f"Error processing streaming completion: {str(e)}",
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


def handler(job):
    """
    RunPod serverless handler for OpenAI-compatible API.

    Handles requests routed through RunPod's OpenAI compatibility layer.
    Expected job input format:
    {
        "openai_route": "/v1/models" or "/v1/chat/completions",
        "openai_input": {...} # The request body as dict
    }
    """
    try:
        job_input = job.get("input", {})

        # Check for OpenAI-compatible routing
        openai_route = job_input.get("openai_route")
        openai_input = job_input.get("openai_input", {})

        if not openai_route:
            return {"error": "No openai_route provided. Use RunPod's OpenAI-compatible URL format."}

        # Initialize handler if needed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(initialize_handler())

            # Route based on OpenAI endpoint
            if openai_route == "/v1/models":
                logger.info("Processing models list request")
                return create_models_response()

            elif openai_route == "/v1/chat/completions":
                logger.info(
                    f"Processing chat completion request for model: {openai_input.get('model')}")

                # Check for authentication if API key is set
                auth_header = job_input.get("authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    api_key = auth_header[7:]  # Remove "Bearer " prefix
                    if not verify_api_key(api_key):
                        return {"error": "Invalid API key"}
                elif API_KEY is not None:
                    return {"error": "API key required. Please provide Authorization header with Bearer token."}

                # Check if streaming is requested
                stream = openai_input.get("stream", False)

                if stream:
                    logger.info("Streaming response requested")
                    # For streaming, we need to return a generator
                    # RunPod handles this by yielding results

                    async def stream_generator():
                        chunks = []
                        async for chunk in create_streaming_chat_completion(openai_input):
                            chunks.append(chunk)
                        return "\n".join(chunks)

                    result = loop.run_until_complete(stream_generator())
                    return {"stream_response": result}
                else:
                    logger.info("Non-streaming response requested")
                    return loop.run_until_complete(create_chat_completion_response(openai_input))

            else:
                return {"error": f"Unsupported route: {openai_route}"}

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": f"Handler error: {str(e)}"}


if __name__ == "__main__":
    # Start the RunPod serverless handler
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
