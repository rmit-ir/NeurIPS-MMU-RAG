"""
Abstract interface for RAG systems following MMU-RAG challenge requirements.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, AsyncGenerator, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel


class CitationItem(TypedDict):
    """TypedDict for citation objects with structured metadata."""

    url: str
    icon_url: Optional[str]
    date: Optional[str]
    title: Optional[str]
    text: Optional[str]
    sid: Optional[str]  # short id, 1, 2, 3, or 1_1, 1_2, etc.


# MMU-RAG Challenge Request/Response Models
class EvaluateRequest(BaseModel):
    """Request model for /evaluate endpoint."""

    query: str
    iid: str


class EvaluateResponse(BaseModel):
    """Response model for /evaluate endpoint."""

    query_id: str  # same as iid from the request
    generated_response: str  # system's generated answer
    citations: List[str]  # list of citations
    contexts: List[str]  # list of actual document contexts used for generation


class RunRequest(BaseModel):
    """Request model for /run endpoint."""

    question: str


class RunStreamingResponse(BaseModel):
    """Response model for streaming /run endpoint."""

    intermediate_steps: Optional[str] = None
    final_report: Optional[str] = None
    is_intermediate: bool = False
    complete: bool = False
    # extra field on top of the OpenAI format
    citations: Optional[List[CitationItem]] = None
    error: Optional[str] = None
    # TODO: respond models used for each step in response, so frontend knows how to parse results


class RAGInterface(ABC):
    """Abstract base class for RAG systems following MMU-RAG challenge requirements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this RAG system."""
        pass

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request for the /evaluate endpoint.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with query_id, generated_response, citations, and contexts
        """
        # Convert EvaluateRequest to RunRequest
        run_request = RunRequest(question=request.query)

        # Get the streaming function
        stream_func = await self.run_streaming(run_request)

        # Collect all streaming responses
        generated_response = ""
        citations: List[CitationItem] = []
        error_msg: Optional[str] = None

        try:
            # Iterate through all streaming responses
            async for response in stream_func():
                # Collect final report content (the actual answer)
                if response.final_report and not response.is_intermediate:
                    generated_response += response.final_report

                # Collect citations when available
                if response.citations:
                    citations = response.citations

                # Check for errors
                if response.error:
                    error_msg = response.error

                # Break if complete
                if response.complete:
                    break

        except Exception as e:
            error_msg = str(e)
            generated_response = f"Error processing query: {error_msg}"

        # Handle error cases
        if error_msg and not generated_response:
            generated_response = f"Error processing query: {error_msg}"

        # Extract citation URLs and contexts
        citation_urls: List[str] = []
        contexts: List[str] = []

        for citation in citations:
            if citation.get("url"):
                citation_urls.append(citation["url"])
            if citation.get("text"):
                contexts.append(citation.get("text") or "")

        return EvaluateResponse(
            query_id=request.iid,
            generated_response=generated_response.strip()
            if generated_response
            else "No response generated",
            citations=citation_urls,
            contexts=contexts,
        )

    @abstractmethod
    async def run_streaming(
        self, request: RunRequest
    ) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request for the /run endpoint.

        Args:
            request: RunRequest containing the question

        Returns:
            A callable that returns an AsyncGenerator yielding StreamingResponse objects
            following the MMU-RAG streaming format
        """
        pass
