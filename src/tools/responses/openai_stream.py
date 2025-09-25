"""
OpenAI streaming response utility.
"""

from typing import AsyncGenerator, Callable, Optional, List
from pydantic import BaseModel
from systems.rag_interface import RunStreamingResponse, CitationItem
from tools.responses.stream_queue import get_or_start_stream


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
    Convert RAG system responses to OpenAI SSE format with queue-based caching.

    Args:
        start_stream: Function that returns AsyncGenerator of RunStreamingResponse objects
        model: Model name to include in the response
        chat_hash: Optional hash key for queue management and caching

    Yields:
        SSE formatted strings for OpenAI-compatible streaming endpoint
    """
    if not chat_hash:
        print('No chat_hash, direct stream')
        # No caching, stream directly
        async for chunk_data in _convert_stream_to_openai(start_stream, model):
            yield chunk_data
        return

    # Use queue system for caching and multiple subscribers
    async def stream_factory():
        print('Using stream_factory with chat_hash:', chat_hash)
        async for chunk_data in _convert_stream_to_openai(start_stream, model):
            yield chunk_data

    print('Getting or starting stream with chat_hash:', chat_hash)
    async for chunk_data in get_or_start_stream(chat_hash, stream_factory):
        print('Yielding chunk_data:', chunk_data)
        if chunk_data and not chunk_data.startswith("ERROR:"):
            yield chunk_data


async def _convert_stream_to_openai(
    start_stream: Callable[[], AsyncGenerator[RunStreamingResponse, None]],
    model: str = "placeholder"
) -> AsyncGenerator[str, None]:
    """Convert RAG stream to OpenAI format."""
    chunk_id = 0

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
                break

            # Break out, if complete
            if response.complete:
                break

        # Add final DONE message
        yield "data: [DONE]\n\n"

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
