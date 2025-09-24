import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, List, Optional
from openai.types.chat import ChatCompletionMessageParam
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from tools.llm_servers.vllm_server import get_llm_mgr
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.web_search import SearchError, SearchResult, search_fineweb
from tools.doc_truncation import truncate_docs


class VanillaRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        gpu_memory_utilization: Optional[float] = 0.6,
        max_model_len: Optional[int] = 20000,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retrieval_words_threshold: int = 5000,
    ):
        """
        Initialize VanillaRAG with LLM server.

        Args:
            model_id: The model ID to use for LLM server
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_words_threshold = retrieval_words_threshold

        self.logger = get_logger("vanilla_rag")
        self.llm_client = None

    @property
    def name(self) -> str:
        return "vanilla-rag"

    async def _ensure_llm_client(self):
        if not self.llm_client:
            llm_mgr = get_llm_mgr(
                model_id=self.model_id,
                reasoning_parser=self.reasoning_parser,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                api_key=self.api_key
            )
            self.llm_client = await llm_mgr.get_openai_client(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    def _to_context(self, results: list[SearchResult | SearchError]) -> str:
        # Filter out SearchError objects and get only SearchResult objects
        search_results = [r for r in results if isinstance(r, SearchResult)]

        # Truncate documents to prevent context from being too long
        truncated_results = truncate_docs(
            search_results, self.retrieval_words_threshold)

        context = "<search-results>"
        context += "\n".join([f"""
Webpage [ID={r.sid}] [URL={r.url}] [Date={r.date}]:

{r.text}""" for r in truncated_results])
        context += "</search-results>"
        return context

    def _llm_messages(self, results: list[SearchResult | SearchError], query: str) -> List[ChatCompletionMessageParam]:
        # Create a simple RAG prompt
        system_message = f"""You are a knowledgeable AI search assistant.

Your search engine has returned a list of relevant webpages based on the user's query, listed below in <search-results> tags.

The next user message is the full user query, and you need to explain and answer the search query based on the search results. Do not make up answers that are not supported by the search results. If the search results do not have the necessary information for you to answer the search query, say you don't have enough information for the search query.

Keep your response concise and to the point, and do not answer to greetings or chat with the user.

Current time at UTC+00:00 timezone: {datetime.now(timezone.utc)}
Search results knowledge cutoff: 01 Jan 2022
/nothink
"""
        system_message = \
            str(system_message) + self._to_context(results)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request using LLM server.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with generated answer
        """
        try:
            await self._ensure_llm_client()
            if not self.llm_client:
                raise RuntimeError("LLM client is not initialized.")

            # Search for relevant documents
            results = await search_fineweb(request.query, k=5)
            messages = self._llm_messages(results, request.query)

            # Generate response using LLM
            generated_response, _ = await self.llm_client.complete_chat(messages)

            return EvaluateResponse(
                query_id=request.iid,
                citations=[],  # TODO: Add citation extraction logic
                generated_response=generated_response or "Answer unavailable."
            )

        except Exception as e:
            self.logger.error("Error in evaluate", error=str(e))
            return EvaluateResponse(
                query_id=request.iid,
                citations=[],
                generated_response=f"Error processing query: {str(e)}"
            )

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request using LLM server.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function for streaming responses
        """
        async def stream():
            try:
                # TODO: this message is not sent to frontend, it's blocked by LLM startup
                # Ensure server is running
                yield RunStreamingResponse(
                    intermediate_steps="Initializing LLM server...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                await self._ensure_llm_client()
                if not self.llm_client:
                    raise RuntimeError("LLM server failed to launch\n\n")

                yield RunStreamingResponse(
                    intermediate_steps="Processing question with language model...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                yield RunStreamingResponse(
                    intermediate_steps=f"Searching: {request.question}\n\n",
                    is_intermediate=True,
                    complete=False
                )
                results = await search_fineweb(request.question, k=5)
                md_urls = '\n'.join(
                    [f"- {r.url}" for r in results if isinstance(r, SearchResult)])
                yield RunStreamingResponse(
                    intermediate_steps=f"""Found {len(results)} results

{md_urls}\n\n""",
                    is_intermediate=True,
                    complete=False
                )
                messages = self._llm_messages(results, request.question)

                yield RunStreamingResponse(
                    intermediate_steps="Starting to answer\n\n",
                    is_intermediate=True,
                    complete=False
                )

                async for chunk in self.llm_client.complete_chat_streaming(messages):
                    if chunk.choices[0].finish_reason is not None:
                        # Stream finished
                        break
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        yield RunStreamingResponse(
                            intermediate_steps=delta.reasoning_content,
                            is_intermediate=True,
                            complete=False
                        )
                    elif hasattr(delta, 'content') and delta.content:
                        yield RunStreamingResponse(
                            final_report=delta.content,
                            is_intermediate=False,
                            complete=False
                        )
                    # otherwise ignore empty deltas

                citations = [
                    CitationItem(
                        url=r.url,
                        icon_url=to_icon_url(r.url),
                        title=None
                    )
                    for r in results if isinstance(r, SearchResult)
                ]
                # Final response
                yield RunStreamingResponse(
                    citations=citations,  # TODO: Add real citation extraction logic
                    is_intermediate=False,
                    complete=True
                )

            except Exception as e:
                self.logger.error("Error in run_streaming", error=str(e))
                yield RunStreamingResponse(
                    final_report=f"Error processing question: {str(e)}",
                    citations=[],
                    is_intermediate=False,
                    complete=True,
                    error=str(e)
                )

        return stream


if __name__ == "__main__":
    import asyncio

    async def main():
        """Simple test execution for VanillaRAG."""
        print("Testing VanillaRAG with LLM server...")

        # Initialize VanillaRAG
        rag = VanillaRAG(
            model_id="Qwen/Qwen3-4B",
            api_key=None,  # Optional API key
            temperature=0.0,
            max_tokens=4096
        )

        try:
            # Test 1: Evaluate method
            print("\n=== Testing Evaluate Method ===")
            eval_request = EvaluateRequest(
                query="What is artificial intelligence?",
                iid="test-001"
            )

            eval_response = await rag.evaluate(eval_request)
            print(f"Query ID: {eval_response.query_id}")
            print(f"Response: {eval_response.generated_response}")
            print(f"Citations: {eval_response.citations}")

            # Test 2: Streaming method
            print("\n=== Testing Streaming Method ===")
            run_request = RunRequest(
                question="Explain the concept of machine learning in simple terms."
            )

            stream_func = await rag.run_streaming(run_request)
            print("Streaming response:")

            print_type = 'intermediate'
            async for response in stream_func():
                if response.is_intermediate:
                    if response.intermediate_steps:
                        if print_type != 'intermediate':
                            print_type = 'intermediate'
                            print(
                                f"\n[THINK] {response.intermediate_steps}\n\n")
                        print(response.intermediate_steps, end="", flush=True)
                else:
                    if print_type != 'final':
                        print_type = 'final'
                        print(f"\n[FINAL] {response.final_report}\n\n")
                    if response.final_report:
                        print(response.final_report, end="", flush=True)
                    if response.citations:
                        print(f"Citations: {response.citations}")
                    if response.error:
                        print(f"Error: {response.error}")

        except Exception as e:
            print(f"Error during testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
