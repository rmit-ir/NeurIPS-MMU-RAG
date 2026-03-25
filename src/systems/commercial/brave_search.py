"""
BraveSearchRAG implementation using Brave's Answers API.
Calls Brave Answers API to get AI-generated answers grounded in web search.
"""

import asyncio
from typing import AsyncGenerator, Callable, List

from systems.rag_interface import RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from tools.brave_answers import brave_answers, BraveAnswerResult
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url


class BraveSearchRAG(RAGInterface):
    """RAG system using Brave's Answers API for AI-grounded search answers."""

    def __init__(self, enable_research: bool = False):
        self.enable_research = enable_research
        self.logger = get_logger('BraveSearchRAG')

    @property
    def name(self) -> str:
        return "brave-search-rag"

    def _to_citations(self, result: BraveAnswerResult) -> List[CitationItem]:
        """Convert Brave answer citations to CitationItems."""
        seen_urls: dict[str, int] = {}
        items: List[CitationItem] = []
        for c in result.citations:
            if c.url in seen_urls:
                continue
            seen_urls[c.url] = c.number
            items.append(CitationItem(
                sid=str(c.number),
                url=c.url,
                icon_url=to_icon_url(c.url),
                date=None,
                title=None,
                text=c.snippet,
                chunk_idx=None,
            ))
        return items

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        async def stream():
            try:
                yield RunStreamingResponse(
                    intermediate_steps="Searching with Brave Answers API...",
                    is_intermediate=True,
                    complete=False,
                )

                result = await brave_answers(
                    query=request.question,
                    enable_citations=True,
                    enable_research=self.enable_research,
                )

                citations = self._to_citations(result)

                if result.total_cost > 0:
                    self.logger.info("Brave Answers usage",
                                     queries=result.queries_used,
                                     tokens_in=result.tokens_in,
                                     tokens_out=result.tokens_out,
                                     cost=result.total_cost)

                yield RunStreamingResponse(
                    final_report=result.content,
                    citations=citations,
                    is_intermediate=False,
                    complete=True,
                )

            except Exception as e:
                yield RunStreamingResponse(
                    error=f"Error during Brave search: {str(e)}",
                    is_intermediate=False,
                    complete=True,
                )

        return stream


async def main():
    """Simple test of BraveSearchRAG."""
    rag = BraveSearchRAG()
    request = RunRequest(question="What are the latest advances in multimodal RAG?")
    stream_func = await rag.run_streaming(request)

    async for response in stream_func():
        if response.intermediate_steps:
            print(f"Step: {response.intermediate_steps}")
        elif response.final_report:
            print(f"Final: {response.final_report}")
        elif response.error:
            print(f"Error: {response.error}")


if __name__ == "__main__":
    asyncio.run(main())
