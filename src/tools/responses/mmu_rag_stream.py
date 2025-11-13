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
    # Accumulators for content
    accumulated_intermediate_steps = ""
    accumulated_final_report = ""
    accumulated_citations = None

    try:
        async for response in start_stream():
            # Accumulate intermediate steps with separator
            if response.intermediate_steps:
                if accumulated_intermediate_steps and accumulated_intermediate_steps.endswith('\n'):
                    # Only add separator if accumulated content ends with a line break
                    accumulated_intermediate_steps += "|||---|||"
                accumulated_intermediate_steps += response.intermediate_steps

            # Accumulate final report content
            if response.final_report:
                accumulated_final_report += response.final_report

            # Store citations (latest ones win)
            if response.citations:
                accumulated_citations = response.citations

            # Prepare response dict based on whether this is intermediate or final
            if response.is_intermediate:
                # For intermediate responses, send accumulated intermediate_steps
                response_dict = {
                    "intermediate_steps": accumulated_intermediate_steps,
                    "final_report": None,
                    "is_intermediate": True,
                    "complete": False
                }
            else:
                # For final responses (answer phase), preserve intermediate history and send accumulated final_report
                response_dict = {
                    "intermediate_steps": accumulated_intermediate_steps,
                    "final_report": accumulated_final_report,
                    "is_intermediate": False,
                    "complete": response.complete
                }

            # Add optional fields if present
            if accumulated_citations:
                # Convert CitationItem objects to URL strings for MMU-RAG format
                response_dict["citations"] = [citation["url"]
                                              for citation in accumulated_citations]
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
            "intermediate_steps": accumulated_intermediate_steps if accumulated_intermediate_steps else None,
            "final_report": accumulated_final_report if accumulated_final_report else None,
            "error": f"Error processing request: {str(e)}",
            "complete": True
        }
        yield f"data: {json.dumps(error_response)}\n\n"
