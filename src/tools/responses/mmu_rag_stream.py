"""
MMU-RAG streaming response utility.
"""

import json
from typing import AsyncGenerator, Callable
from systems.rag_interface import RunStreamingResponse


async def to_mmu_rag_stream(
    start_stream: Callable[[], AsyncGenerator[RunStreamingResponse, None]]
) -> AsyncGenerator[str, None]:
    """
    Convert RAG system responses to MMU-RAG SSE format.

    Args:
        rag_responses: AsyncGenerator of RunStreamingResponse objects

    Yields:
        SSE formatted strings for MMU-RAG streaming endpoint
    """
    try:
        async for response in start_stream():
            # Convert response to dict for JSON serialization
            response_dict = {
                "intermediate_steps": response.intermediate_steps,
                "final_report": response.final_report,
                "is_intermediate": response.is_intermediate,
                "complete": response.complete
            }

            # Add optional fields if present
            if response.citations:
                response_dict["citations"] = response.citations
            if response.error:
                response_dict["error"] = response.error

            # Format as SSE
            yield f"data: {json.dumps(response_dict)}\n\n"

            # Stop streaming if complete or error
            if response.complete or response.error:
                break

    except Exception as e:
        # Send error response and stop stream
        error_response = {
            "error": f"Error processing request: {str(e)}",
            "complete": True
        }
        yield f"data: {json.dumps(error_response)}\n\n"
