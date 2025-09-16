"""
AzureO3ResearchRAG implementation using Azure OpenAI o3 model.
Calls Azure OpenAI o3 API to research and answer using provided query.
"""

import os
import asyncio
import aiohttp
from typing import AsyncGenerator, Callable
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse
from tools.logging_utils import get_logger
from tools.retry_utils import retry


class AzureO3ResearchRAG(RAGInterface):
    """RAG system using Azure OpenAI o3 model for comprehensive research."""

    def __init__(self, model="o3-deep-research"):
        # Azure OpenAI endpoint and deployment configuration
        self.endpoint = os.getenv("AZURE_API_ENDPOINT")
        self.api_key = os.getenv("AZURE_API_KEY")
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "AZURE_API_ENDPOINT and AZURE_API_KEY environment variables are required")

        # Build the complete API URL
        self.model = model
        self.api_path = f"/openai/deployments/{self.model}/chat/completions"
        self.api_url = self.endpoint + self.api_path
        self.api_version = "2025-01-01-preview"
        self.logger = get_logger('AzureO3ResearchRAG')

        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @property
    def name(self) -> str:
        return "azure-o3-research-rag"

    def _create_research_messages(self, query: str) -> list:
        """Create messages for research-focused prompt."""
        # system_message = """You are an expert researcher with deep analytical capabilities.
        # Your task is to provide comprehensive, well-structured research on the given topic.

        # Please provide:
        # 1. A thorough analysis of the topic
        # 2. Key insights and findings
        # 3. Multiple perspectives where relevant
        # 4. Evidence-based conclusions

        # Be detailed, accurate, and insightful in your response."""

        return [
            # TODO: maybe somehow re-enable system message, if useful
            # {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

    @retry(max_retries=8, retry_on=(aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError))
    async def _make_api_request(self, messages: list) -> str:
        """Make API request to Azure OpenAI o3 model."""
        payload = {"messages": messages, "model": self.model}
        url = f"{self.api_url}?api-version={self.api_version}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"API request failed with status {response.status}: {error_text}")

                result = await response.json()
                return result["choices"][0]["message"]["content"]

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request using Azure OpenAI o3 model.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with query_id and generated_response
        """
        try:
            # Create messages for evaluation
            messages = self._create_research_messages(request.query)

            # Make API request
            generated_response = await self._make_api_request(messages)

            return EvaluateResponse(
                query_id=request.iid,
                citations=[],
                generated_response=generated_response
            )

        except Exception as e:
            self.logger.error("Error processing evaluation request",
                              query_id=request.iid, error=str(e))
            raise

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request using Azure OpenAI o3 model.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function that yields RunStreamingResponse objects
        """
        async def stream():
            try:

                yield RunStreamingResponse(
                    intermediate_steps=f"Starting research with {self.model}...",
                    is_intermediate=True,
                    complete=False
                )

                messages = self._create_research_messages(request.question)
                # Execute the API request
                final_report = await self._make_api_request(messages)

                # Final response with complete results
                # Note: Azure o3 doesn't provide external citations like Perplexity
                yield RunStreamingResponse(
                    final_report=final_report,
                    citations=[],  # o3 model doesn't provide external citations
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
    """Simple test of AzureO3ResearchRAG."""
    try:
        rag = AzureO3ResearchRAG()

        # Test with a simple question
        request = RunRequest(
            question="I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building.")
        stream_func = await rag.run_streaming(request)

        async for response in stream_func():
            if response.intermediate_steps:
                print(f"Step: {response.intermediate_steps}")
            elif response.final_report:
                print(f"Final: {response.final_report}")
            elif response.error:
                print(f"Error: {response.error}")

    except Exception as e:
        print(f"Failed to initialize or run AzureO3ResearchRAG: {e}")


if __name__ == "__main__":
    asyncio.run(main())
