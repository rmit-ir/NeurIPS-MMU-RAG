"""
OpenAI streaming response utility.
"""

from typing import AsyncGenerator, Callable, Optional, List
from pydantic import BaseModel
from systems.rag_interface import RunStreamingResponse, CitationItem
from tools.cache_lru import LRUCache


TTL = 3600 * 24  # 1 day
MAX_CACHE_SIZE = 1000  # Max number of cached responses
chat_resp_cache = LRUCache(max_size=MAX_CACHE_SIZE)


# OpenAI API Response Models
class OpenAIDelta(BaseModel):
    """Delta content in OpenAI streaming response."""
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    citations: Optional[List[CitationItem]] = None
    role: Optional[str] = None


class OpenAIChoice(BaseModel):
    """Choice object in OpenAI streaming response."""
    index: int
    delta: OpenAIDelta
    finish_reason: Optional[str] = None


class OpenAIStreamChunk(BaseModel):
    """OpenAI streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChoice]


class OpenAIError(BaseModel):
    """OpenAI error response."""
    message: str
    type: str
    code: str


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response wrapper."""
    error: OpenAIError


async def to_openai_stream(
    start_stream: Callable[[], AsyncGenerator[RunStreamingResponse, None]],
    model: str = "placeholder",
    chat_hash: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert RAG system responses to OpenAI SSE format.

    Args:
        start_stream: Function that returns AsyncGenerator of RunStreamingResponse objects
        model: Model name to include in the response
        chat_hash: Optional hash key for caching responses

    Yields:
        SSE formatted strings for OpenAI-compatible streaming endpoint
    """
    # Check cache first if chat_hash is provided
    if chat_hash:
        cached_response = await chat_resp_cache.get(chat_hash)
        if cached_response:
            # Yield cached response chunks
            for chunk_data in cached_response:
                yield chunk_data
            return

    chunk_id = 0
    accumulated_chunks = []  # Store chunks for caching

    try:
        async for response in start_stream():
            chunk_id += 1

            chunk: OpenAIStreamChunk
            if response.is_intermediate:
                chunk = OpenAIStreamChunk(
                    id=f"chatcmpl-{chunk_id}",
                    created=1234567890,  # You might want to use actual timestamp
                    model=model,
                    choices=[
                        OpenAIChoice(
                            index=0,
                            delta=OpenAIDelta(
                                reasoning_content=response.intermediate_steps,
                                citations=response.citations),
                            finish_reason="stop" if response.complete else None
                        )
                    ]
                )
            else:
                chunk = OpenAIStreamChunk(
                    id=f"chatcmpl-{chunk_id}",
                    created=1234567890,
                    model=model,
                    choices=[
                        OpenAIChoice(
                            index=0,
                            delta=OpenAIDelta(
                                content=response.final_report,
                                citations=response.citations),
                            finish_reason="stop" if response.complete else None
                        )
                    ]
                )

            # Format as SSE
            chunk_data = f"data: {chunk.model_dump_json()}\n\n"
            yield chunk_data

            # Accumulate for caching if chat_hash is provided
            if chat_hash:
                accumulated_chunks.append(chunk_data)

            # Handle errors
            if response.error:
                error_response = OpenAIErrorResponse(
                    error=OpenAIError(
                        message=response.error,
                        type="server_error",
                        code="internal_error"
                    )
                )
                error_data = f"data: {error_response.model_dump_json()}\n\n"
                yield error_data
                if chat_hash:
                    accumulated_chunks.append(error_data)
                break

            # Break out, if complete
            if response.complete:
                break

        # Add final DONE message
        done_data = "data: [DONE]\n\n"
        yield done_data
        if chat_hash:
            accumulated_chunks.append(done_data)
            # Cache the complete response with 1 hour TTL
            await chat_resp_cache.put(chat_hash, accumulated_chunks, ttl=TTL)

    except Exception as e:
        # Send error response and stop stream
        error_response = OpenAIErrorResponse(
            error=OpenAIError(
                message=f"Error processing request: {str(e)}",
                type="server_error",
                code="internal_error"
            )
        )
        yield f"data: {error_response.model_dump_json()}\n\n"
