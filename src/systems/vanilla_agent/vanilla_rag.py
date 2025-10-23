import asyncio
from typing import AsyncGenerator, Callable, Optional
from systems.rag_interface import EvaluateRequest, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.rag_util_fn import build_llm_messages, get_default_llms, inter_resp
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker
from tools.web_search import SearchResult, search_clueweb
from tools.docs_utils import truncate_docs, update_docs_sids


class VanillaRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        gpu_memory_utilization: Optional[float] = 0.6,
        max_model_len: Optional[int] = 20000,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        retrieval_words_threshold: int = 5000,
        enable_think: bool = True,
        k_docs: int = 30,
        cw22_a: bool = True,
        num_qvs: int = 3,
    ):
        """
        Initialize VanillaRAG with LLM server.

        Args:
            model_id: The model ID to use for LLM server
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            max_tokens: Maximum tokens to generate
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.retrieval_words_threshold = retrieval_words_threshold
        self.enable_think = enable_think
        self.k_docs = k_docs
        self.cw22_a = cw22_a
        self.num_qvs = num_qvs

        self.logger = get_logger("vanilla_rag")
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

    @property
    def name(self) -> str:
        return "vanilla-rag"

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        async def stream():
            try:
                yield inter_resp(f"Processing question: {request.question}\n\n", silent=False, logger=self.logger)
                llm, reranker = await get_default_llms()

                yield inter_resp(f"Searching: {request.question}\n\n", silent=False, logger=self.logger)
                docs = await search_clueweb(request.question,
                                            k=self.k_docs, cw22_a=self.cw22_a)
                total_docs = len(docs)
                yield inter_resp(f"Found {total_docs} documents for {request.question}\n\n", silent=False, logger=self.logger)

                docs = [r for r in docs if isinstance(r, SearchResult)]
                docs = await reranker.rerank(request.question, docs)
                reranked_docs = len(docs)
                yield inter_resp(f"Reranked {reranked_docs} documents for {request.question}\n\n", silent=False, logger=self.logger)

                docs = truncate_docs(docs, self.retrieval_words_threshold)
                docs = update_docs_sids(docs)
                md_urls = '\n'.join(
                    [f"- {r.url}" for r in docs if isinstance(r, SearchResult)])
                yield inter_resp(f"""Search returned {total_docs}, identified {reranked_docs} relevant, truncated to {len(docs)} web pages.

{md_urls}\n\n""", silent=False, logger=self.logger)

                yield inter_resp("Starting final answer\n\n", silent=False, logger=self.logger)
                messages = build_llm_messages(
                    docs, request.question, self.enable_think)
                async for chunk in llm.complete_chat_streaming(messages):
                    if chunk.choices[0].finish_reason is not None:
                        # Stream finished
                        break
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        # still intermediate steps
                        yield inter_resp(delta.reasoning_content, silent=True, logger=self.logger)
                    elif hasattr(delta, 'content') and delta.content:
                        # final report
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
                        date=str(r.date) if r.date else None,
                        sid=r.sid,
                        title=None,
                        text=r.text
                    )
                    for r in docs if isinstance(r, SearchResult)
                ]
                # Final response
                yield RunStreamingResponse(
                    citations=citations,
                    is_intermediate=False,
                    complete=True
                )

            except Exception as e:
                self.logger.exception("Error in run_streaming")
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
                        print(f"\n\nCitations: {response.citations}")
                    if response.error:
                        print(f"\n\nError: {response.error}")

        except Exception as e:
            print(f"Error during testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
