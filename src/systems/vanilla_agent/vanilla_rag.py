import asyncio
from typing import Any, AsyncGenerator, Callable, Optional
from systems.rag_interface import EvaluateRequest, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.rag_util_fn import build_llm_messages, get_default_llms, inter_resp, search_w_qv
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker
from tools.web_search import SearchResult
from tools.docs_utils import truncate_docs, update_docs_sids


class VanillaRAG(RAGInterface):
    def __init__(
        self,
        reasoning_parser: Optional[str] = "qwen3",
        gpu_memory_utilization: Optional[float] = 0.6,
        max_model_len: Optional[int] = 25_000,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        retrieval_words_threshold: int = 18000,
        # Actually only supports default model + Qwen3 /nothink style
        enable_think: bool = True,
        k_docs: int = 30,
        cw22_a: bool = True,
        num_qvs: int = 3,
        skip_rerank: bool = False,
        alt_llm_api_base: Optional[str] = None,
        alt_llm_api_key: Optional[str] = None,
        alt_llm_model: Optional[str] = None,
        # alt model like gpt-oss should be configured in alt_llm_reasoning_effort,
        alt_llm_reasoning_effort: Optional[str] = None,
        alt_reranker_api_base: Optional[str] = None,
        alt_reranker_api_key: Optional[str] = None,
        alt_reranker_model: Optional[str] = None,
        pre_flight_llm: bool = False,
        pre_flight_reranker: bool = False,
    ):
        """
        Initialize VanillaRAG with LLM server.

        Args:
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            max_tokens: Maximum tokens to generate
            skip_rerank: Whether to skip reranking step (default False)
            alt_llm_api_base: Alternative LLM API base URL
            alt_llm_api_key: Alternative LLM API key
            alt_llm_model: Alternative LLM model name
            alt_llm_reasoning_effort: Alternative LLM reasoning effort setting
            alt_reranker_api_base: Alternative reranker API base URL
            alt_reranker_api_key: Alternative reranker API key
            alt_reranker_model: Alternative reranker model name
            pre_flight_llm: Whether to perform pre-flight check for LLM
            pre_flight_reranker: Whether to perform pre-flight check for reranker
        """
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
        self.skip_rerank = skip_rerank
        self.alt_llm_api_base = alt_llm_api_base
        self.alt_llm_api_key = alt_llm_api_key
        self.alt_llm_model = alt_llm_model
        self.alt_llm_reasoning_effort: Any = alt_llm_reasoning_effort
        self.alt_reranker_api_base = alt_reranker_api_base
        self.alt_reranker_api_key = alt_reranker_api_key
        self.alt_reranker_model = alt_reranker_model
        self.pre_flight_llm = pre_flight_llm
        self.pre_flight_reranker = pre_flight_reranker

        self.logger = get_logger("vanilla_rag")
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

        self.logger.info("Initialized VanillaRAG",
                         max_tokens=self.max_tokens,
                         retrieval_words_threshold=self.retrieval_words_threshold,
                         enable_think=self.enable_think,
                         k_docs=self.k_docs,
                         cw22_a=self.cw22_a,
                         num_qvs=self.num_qvs,
                         alt_llm_api_base=self.alt_llm_api_base,
                         alt_llm_model=self.alt_llm_model,
                         alt_reranker_api_base=self.alt_reranker_api_base,
                         alt_reranker_model=self.alt_reranker_model)

    @property
    def name(self) -> str:
        return "vanilla-rag"

    async def get_active_models(self):
        if self.alt_llm_api_base and self.alt_llm_model:
            alt_llm = GeneralOpenAIClient(model_id=self.alt_llm_model,
                                          api_base=self.alt_llm_api_base,
                                          api_key=self.alt_llm_api_key,
                                          # Cerebras only use this for GPT-OSS, for Qwen3, use /nothink in system prompt
                                          reasoning_effort=self.alt_llm_reasoning_effort,
                                          max_retries=3)
        if self.alt_reranker_api_base and self.alt_reranker_model:
            alt_reranker = GeneralReranker(model_id=self.alt_reranker_model,
                                           api_base=self.alt_reranker_api_base,
                                           api_key=self.alt_reranker_api_key)
        if (self.alt_llm_api_base and self.alt_llm_model and
                self.alt_reranker_api_base and self.alt_reranker_model):
            self.logger.info("Using alternative LLM and Reranker for VanillaRAG",
                             alt_llm_api_base=self.alt_llm_api_base,
                             alt_reranker_api_base=self.alt_reranker_api_base)
            return alt_llm, alt_reranker

        llm, reranker = await get_default_llms()
        if self.alt_llm_api_base and self.alt_llm_model:
            return alt_llm, reranker
        if self.alt_reranker_api_base and self.alt_reranker_model:
            return llm, alt_reranker
        return llm, reranker

    async def pre_flight_models(self) -> None:
        from openai.types.chat import ChatCompletionMessageParam
        from typing import List
        from tools.reranker_vllm import _dummy_search_result as _search_result

        llm, reranker = await self.get_active_models()
        if self.pre_flight_llm:
            self.logger.info("Performing pre-flight check for LLM")
            test_messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": "Hello, how are you?"}
            ]
            async for chunk in llm.complete_chat_streaming(test_messages, max_tokens=1):
                self.logger.info("Pre-flight LLM response received",
                                 response=chunk)

        if self.pre_flight_reranker:
            self.logger.info("Performing pre-flight check for Reranker")
            test_query = "Where is the capital of China?"
            test_docs = [
                _search_result("1", "The capital city of China is Beijing."),
                _search_result("2", "The capital city of China is Shanghai."),
            ]
            ranked_docs = await reranker.rerank(test_query, test_docs)
            self.logger.info("Pre-flight Reranker response received",
                             ranked_doc_ids=[doc.sid for doc in ranked_docs])

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        async def stream():
            try:
                # Run pre-flight checks but don't await
                asyncio.create_task(self.pre_flight_models())

                llm, reranker = await self.get_active_models()

                yield inter_resp(f"Searching: {request.question}\n\n", silent=False, logger=self.logger)
                # docs = await search_clueweb(request.question,
                #                             k=self.k_docs, cw22_a=self.cw22_a)
                qvs, docs = await search_w_qv(request.question, num_qvs=self.num_qvs, enable_think=self.enable_think, logger=self.logger, cw22_a=self.cw22_a, preset_llm=llm)
                total_docs = len(docs)
                qvs_str = "; ".join(qvs)
                yield inter_resp(f"Searched: {qvs_str}, found {len(docs)} documents\n\n",
                                 silent=False, logger=self.logger)

                docs = [r for r in docs if isinstance(r, SearchResult)]

                if not self.skip_rerank:
                    yield inter_resp("Reranking documents...\n\n", silent=False, logger=self.logger)
                    docs = await reranker.rerank(request.question, docs)
                docs = truncate_docs(docs, self.retrieval_words_threshold)
                docs = update_docs_sids(docs)
                reranked_docs = len(docs)
                yield inter_resp(f"""Search returned {total_docs}, identified {reranked_docs} relevant, truncated to {len(docs)} web pages.""", silent=False, logger=self.logger)

                yield inter_resp("Starting final answer\n\n", silent=False, logger=self.logger)
                messages = build_llm_messages(
                    docs, request.question, self.enable_think, model_id=llm.model_id)
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
                    complete=True,
                    metadata={
                        "answer_model_id": llm.model_id,
                        "query_variants_model_id": llm.model_id,
                    },
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
