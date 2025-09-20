from typing import AsyncGenerator, Callable, List, Optional
from openai.types.chat import ChatCompletionMessageParam
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse
from tools.llm_servers.sglang_server import launch_server, terminate_server
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.web_search import SearchError, SearchResult, search_fineweb
from tools.doc_truncation import truncate_docs


class VanillaRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        mem_fraction_static: Optional[float] = 0.4,
        max_running_requests: Optional[int] = 4,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retrieval_words_threshold: int = 5000,
    ):
        """
        Initialize VanillaRAG with SGLang server.

        Args:
            model_id: The model ID to use for SGLang server
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_words_threshold = retrieval_words_threshold

        self.logger = get_logger("vanilla_rag")
        self.server_process = None
        self.client = None
        self.server_host = None
        self.api_base = None
        self._is_processing = False
        self._is_llm_starting = False

    async def _ensure_server_running(self):
        """Ensure SGLang server is running and client is initialized."""
        # if server and client are not initialized, start them
        if not (self.server_process and self.client):
            # however, only start if not already starting
            if not self._is_llm_starting:
                try:
                    self._is_llm_starting = True
                    self.logger.info("Starting SGLang server",
                                     model_id=self.model_id)
                    self.server_process, self.server_host, self.api_base, port = launch_server(
                        model_id=self.model_id,
                        reasoning_parser=self.reasoning_parser,
                        mem_fraction_static=self.mem_fraction_static,
                        max_running_requests=self.max_running_requests,
                        api_key=self.api_key
                    )

                    self.client = GeneralOpenAIClient(
                        api_base=self.api_base,
                        api_key=self.api_key,
                        model_id=self.model_id,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        llm_name="vanilla_rag_sglang"
                    )
                    self.logger.info(
                        "SGLang server and client initialized", port=port)
                finally:
                    self._is_llm_starting = False
            # otherwise continue waiting

        # Add timeout to prevent infinite loop
        max_wait_time = 3600  # seconds
        wait_time = 0
        # wait for server process and host to be set
        while not (self.client and self.server_host):
            if wait_time >= max_wait_time:
                raise RuntimeError(
                    f"Server failed to initialize within {max_wait_time} seconds")
            await asyncio.sleep(1)
            wait_time += 1

    def _shutdown_server(self):
        """Shutdown the SGLang server."""
        if self.server_process:
            terminate_server(self.server_process)
            self.server_process = None
            self.client = None
            self.api_base = None
            self.logger.info("SGLang server terminated")

    @property
    def name(self) -> str:
        return "vanilla-rag"

    @property
    def is_running(self) -> bool:
        """Check if the SGLang server is running."""
        return self.server_process is not None and self.client is not None

    @property
    def is_processing(self) -> bool:
        """Check if the system is currently processing a request."""
        return self._is_processing

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
        system_message = """You are a knowledgeable AI search assistant.

Your search engine has returned a list of relevant webpages based on the user's query, listed below in <search-results> tags.

The next user message is the full user query, and you need to explain and answer the search query based on the search results. Do not make up answers that are not supported by the search results. If the search results do not have the necessary information for you to answer the search query, say you don't have enough information for the search query.

Keep your response concise and to the point, and do not answer to greetings or chat with the user."""
        system_message = \
            str(system_message) + self._to_context(results)

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request using SGLang server.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with generated answer
        """
        self._is_processing = True
        try:
            await self._ensure_server_running()
            if not self.client:
                raise RuntimeError("SGLang client is not initialized.")

            # Search for relevant documents
            results = await search_fineweb(request.query, k=5)
            messages = self._llm_messages(results, request.query)

            # Generate response using SGLang
            generated_response, _ = self.client.complete_chat(messages)

            return EvaluateResponse(
                query_id=request.iid,
                citations=[],  # TODO: Add citation extraction logic
                generated_response=generated_response
            )

        except Exception as e:
            self.logger.error("Error in evaluate", error=str(e))
            return EvaluateResponse(
                query_id=request.iid,
                citations=[],
                generated_response=f"Error processing query: {str(e)}"
            )
        finally:
            self._is_processing = False

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request using SGLang server.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function for streaming responses
        """
        async def stream():
            self._is_processing = True
            try:
                # Ensure server is running
                yield RunStreamingResponse(
                    intermediate_steps="Initializing SGLang server...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                await self._ensure_server_running()
                if not self.client:
                    raise RuntimeError("SGLang server failed to launch\n\n")

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
                md_urls = '\n'.join([f"- {r.url}" for r in results if isinstance(r, SearchResult)])
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

                async for chunk in self.client.complete_chat_streaming(messages):
                    if chunk.choices[0].finish_reason is not None:
                        # Stream finished
                        break
                    delta = chunk.choices[0].delta
                    if delta.reasoning_content:
                        yield RunStreamingResponse(
                            intermediate_steps=delta.reasoning_content,
                            is_intermediate=True,
                            complete=False
                        )
                    elif delta.content:
                        yield RunStreamingResponse(
                            final_report=delta.content,
                            is_intermediate=False,
                            complete=False
                        )
                    # otherwise ignore empty deltas

                # Final response
                yield RunStreamingResponse(
                    citations=[],  # TODO: Add citation extraction logic
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
            finally:
                self._is_processing = False

        return stream

    def __del__(self):
        """Cleanup when object is destroyed."""
        self._shutdown_server()


if __name__ == "__main__":
    import asyncio

    async def main():
        """Simple test execution for VanillaRAG."""
        print("Testing VanillaRAG with SGLang server...")

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
        finally:
            # Cleanup
            print(
                f"\nFinal status - Is running: {rag.is_running}, Is processing: {rag.is_processing}")
            rag._shutdown_server()
            print("Test completed.")

    # Run the async main function
    asyncio.run(main())
