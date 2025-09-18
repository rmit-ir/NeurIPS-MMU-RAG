"""
OpenAI streaming response utility.
"""

from typing import AsyncGenerator, Callable, Optional, List
from pydantic import BaseModel
from systems.rag_interface import RunStreamingResponse


# OpenAI API Response Models
class OpenAIDelta(BaseModel):
    """Delta content in OpenAI streaming response."""
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
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
    model: str = "placeholder"
) -> AsyncGenerator[str, None]:
    """
    Convert RAG system responses to OpenAI SSE format.

    Args:
        rag_responses: AsyncGenerator of RunStreamingResponse objects
        model: Model name to include in the response

    Yields:
        SSE formatted strings for OpenAI-compatible streaming endpoint
    """
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
                                reasoning_content=response.intermediate_steps),
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
                            delta=OpenAIDelta(content=response.final_report),
                            finish_reason="stop" if response.complete else None
                        )
                    ]
                )

            # Format as SSE
            yield f"data: {chunk.model_dump_json()}\n\n"

            # Send final chunk if complete
            if response.complete:
                yield "data: [DONE]\n\n"
                break

            # Handle errors
            if response.error:
                error_response = OpenAIErrorResponse(
                    error=OpenAIError(
                        message=response.error,
                        type="server_error",
                        code="internal_error"
                    )
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
                break

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
