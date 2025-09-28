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
    citations: Optional[List[CitationItem]] = None
    error: Optional[str] = None


class RAGInterface(ABC):
    """Abstract base class for RAG systems following MMU-RAG challenge requirements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this RAG system."""
        pass

    @abstractmethod
    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request for the /evaluate endpoint.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with query_id and generated_response
        """
        pass

    @abstractmethod
    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request for the /run endpoint.

        Args:
            request: RunRequest containing the question

        Yields:
            StreamingResponse objects following the MMU-RAG streaming format
        """
        pass
