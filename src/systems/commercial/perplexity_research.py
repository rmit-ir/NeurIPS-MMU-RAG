"""
PerplexityResearchRAG implementation using Perplexity's sonar-deep-research model.
Calls Perplexity API to search and answer using provided query.
"""

import os
import asyncio
from typing import AsyncGenerator, Callable, List
import aiohttp
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.retry_utils import retry

PERPLEXITY_MODELS = set(['sonar', 'sonar-pro',
                         'sonar-reasoning', 'sonar-reasoning-pro',
                         'sonar-deep-research'])
"""
See https://docs.perplexity.ai/guides/chat-completions-guide for API specs and parameters

See https://docs.perplexity.ai/getting-started/models for available models.

Search models:

1. sonar: quick search for factual topics
2. sonar-pro: quick search for complex queries

Reasoning models:

1. sonar-reasoning: complex analysis and step-by-step thinking
2. sonar-reasoning-pro: DeepSeek-R1 and CoT

Research models:

1. sonar-deep-research: comprehensive topic reports, exhaustive web research
"""


class PerplexityResearchRAG(RAGInterface):
    """RAG system using Perplexity's sonar-deep-research model for comprehensive research."""

    def __init__(self, model="sonar-reasoning"):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "PERPLEXITY_API_KEY environment variable is required")
        if model not in PERPLEXITY_MODELS:
            raise ValueError(
                f"Invalid model '{model}'. Must be one of: {', '.join(PERPLEXITY_MODELS)}")

        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.logger = get_logger('PerplexityResearchRAG')

    @property
    def name(self) -> str:
        return "perplexity-research-rag"

    @retry(max_retries=8, retry_on=(aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError))
    async def _make_perplexity_request(self, query: str, reasoning_effort: str = "medium") -> dict:
        """Make a request to Perplexity API."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "reasoning_effort": reasoning_effort
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print('error_text', error_text)
                    raise Exception(
                        f"Perplexity API error {response.status}: {error_text}")

                return await response.json()

    def _extract_citations(self, response_data: dict) -> List[CitationItem]:
        """Extract citations from Perplexity response."""
        citations = response_data.get("citations", [])
        if not isinstance(citations, list):
            return []

        citation_items = []
        for url in citations:
            if isinstance(url, str):
                citation_items.append(CitationItem(
                    url=url,
                    icon_url=to_icon_url(url),
                    date=None,
                    title=None
                ))
        return citation_items

    def _extract_content(self, response_data: dict) -> str:
        """Extract the main content from Perplexity response."""
        choices = response_data.get("choices", [])
        if not choices:
            return "No response generated"

        message = choices[0].get("message", {})
        return message.get("content", "No content available")

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request using Perplexity's deep research capabilities.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with query_id and generated_response
        """
        try:
            # Use low reasoning effort for faster evaluation responses
            response_data = await self._make_perplexity_request(
                request.query,
                reasoning_effort="low"
            )

            generated_response = self._extract_content(response_data)
            citations = self._extract_citations(response_data)

            return EvaluateResponse(
                query_id=request.iid,
                citations=[citation["url"] for citation in citations],
                contexts=[],  # Document contexts not available from Perplexity API
                generated_response=generated_response
            )

        except Exception as e:
            self.logger.error("Error processing evaluation request",
                              query_id=request.iid, error=str(e))
            raise

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request using Perplexity's deep research capabilities.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function that yields RunStreamingResponse objects
        """
        async def stream():
            try:
                # Initial step - indicate research is starting
                yield RunStreamingResponse(
                    intermediate_steps="Starting research with perplexity-research...",
                    is_intermediate=True,
                    complete=False
                )

                # Make the actual API call with high reasoning effort for comprehensive research
                response_data = await self._make_perplexity_request(
                    request.question,
                    reasoning_effort="low"
                )
                print('response_data', response_data)

                # Extract final response and citations
                final_report = self._extract_content(response_data)
                citations = self._extract_citations(response_data)

                # TODO: skipped costs for now
                # Add usage information if available
                usage = response_data.get("usage", {})
                if usage:
                    search_queries = usage.get("num_search_queries", None)
                    total_cost = usage.get("cost", {}).get("total_cost", 0)

                    yield RunStreamingResponse(
                        intermediate_steps=f"Found {search_queries} search queries. Total cost: ${total_cost:.3f}",
                        is_intermediate=True,
                        complete=False
                    )

                # Final response with complete results
                yield RunStreamingResponse(
                    final_report=final_report,
                    citations=citations,
                    is_intermediate=False,
                    complete=True
                )

            except Exception as e:
                # Yield error response
                yield RunStreamingResponse(
                    error=f"Error during research: {str(e)}",
                    is_intermediate=False,
                    complete=True
                )

        return stream


async def main():
    """Simple test of PerplexityResearchRAG."""
    rag = PerplexityResearchRAG(model="sonar-pro")

    # Test with a simple question
    request = RunRequest(question="I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building.")
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
