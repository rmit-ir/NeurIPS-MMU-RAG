import asyncio
import json
from openai.types.chat import ChatCompletionMessageParam
from typing import AsyncGenerator, Callable, List, Optional, Tuple
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.rag_util_fn import build_llm_messages, build_to_context, get_default_llms, inter_resp, reformulate_query
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult, search_clueweb
from tools.docs_utils import atruncate_docs, calc_tokens, calc_tokens_str, update_docs_sids


class VanillaAgent(RAGInterface):
    def __init__(
        self,
        context_length: int = 20_000,
        context_tokens: int = 14_904,  # 20_000 - 5096 (for prompt and answer)
        max_tries: int = 5,
        cw22_a: bool = True,
    ):
        """
        Initialize VanillaAgent with LLM server.

        Args:
            model_id: The model ID to use for LLM server
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        self.context_length = context_length
        self.context_tokens = context_tokens
        self.max_tries = max_tries
        self.cw22_a = cw22_a

        self.logger = get_logger("vanilla_agent")
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

    @property
    def name(self) -> str:
        return "vanilla-agent"

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        return EvaluateResponse(
            query_id=request.iid,
            citations=[],
            contexts=[],
            generated_response=f"Not implemented, update RAGInterface to use run_streaming()"
        )

    async def review_documents(self, question: str, next_query: str, acc_queries: List[str], acc_summaries: List[str], docs: List[SearchResult]) -> Tuple[bool, str | None, List[SearchResult], str | None]:
        llm, reranker = await get_default_llms()
        docs = update_docs_sids(docs)
        GRADE_PROMPT = """You are an expert in answering user question "{question}". We are doing research on user's question and currently working on aspect "{next_query}"

Review the search results and see if they are sufficient for answering the user question. Please consider:

1. Does the user want a simple answer or a comprehensive explanation?
2. Does the search results fully addresses the user's query and any subcomponents?
3. When you answer 'yes' in <is-sufficient>, we will proceed to generate the final answer based on these results. If you answer 'no', we will continue the next turn of using a different query to search, and let you review again.
4. If information is missing or uncertain, always return 'no' in <is-sufficient> xml tags for clarification, and generate a new query enclosed in xml markup <new-query>your query</new-query> indicating the clarification needed. If the search results are too off, try clarifying sub-components of the question first, or make reasonable guess. If you think the search results are sufficient, return 'yes' in <is-sufficient> xml tags, and do not provide <new-query> tag.
5. Identify new documents that are important for answering the question but not included in previous documents, and list their IDs (# in ID=[#]) in a comma-separated format within <useful-docs> xml tags. If multiple documents are similar, choose the one with better quality. Do not provide duplicated documents that have been included in previous turns. If no new documents are useful, leave <useful-docs></useful-docs> empty.
6. If useful-docs is not empty, provide a brief summary of what these documents discuss within <useful-docs-summary> xml tags, in 1-2 sentences.

Response format:

- <is-sufficient>yes or no</is-sufficient>
- <new-query>your new query</new-query> (only if is-sufficient is 'no')
- <useful-docs>1,2,3</useful-docs> (list of document IDs that are useful for answering the question, separated by commas)
- <useful-docs-summary></useful-docs-summary> (short summary of what these useful documents are talking about, just the summary, only if useful-docs is not empty)

{prev_questions}
{prev_docs_summaries}
Here is the current question: "{next_query}"
Here is the search results for current question:

{context}
"""
        context = build_to_context(docs)
        prompt = GRADE_PROMPT.format(
            question=question,
            next_query=next_query,
            prev_questions="Here are the previous questions we already tried: " +
            "; ".join(acc_queries) if acc_queries else "",
            prev_docs_summaries="Here are the summaries of previous documents we collected for this question, do not duplicate: " +
            "; ".join(acc_summaries) if acc_summaries else "",
            context=context)
        prompt_tokens = calc_tokens_str(prompt)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        resp_text = ""
        available_context = self.context_length - prompt_tokens - 100
        async for chunk in llm.complete_chat_streaming(messages, max_tokens=available_context):
            if chunk.choices[0].finish_reason is not None:
                break
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                print(delta.reasoning_content, end="", flush=True)
            elif hasattr(delta, 'content') and delta.content:
                print(delta.content, end="", flush=True)
                resp_text += delta.content

        resp_text = resp_text.strip().lower() if resp_text else ""
        is_sufficient = extract_tag_val(resp_text, "is-sufficient") == "yes"
        new_query = extract_tag_val(resp_text, "new-query")
        useful_doc_ids_str = extract_tag_val(resp_text, "useful-docs")
        useful_docs_summary = extract_tag_val(resp_text, "useful-docs-summary")

        self.logger.info("Review documents completed",
                         question=question,
                         next_query=next_query,
                         acc_queries=acc_queries,
                         acc_summaries=acc_summaries,
                         is_sufficient=is_sufficient,
                         new_query=new_query,
                         useful_doc_ids=useful_doc_ids_str,
                         useful_docs_summary=useful_docs_summary)

        useful_docs = []
        if useful_doc_ids_str:
            useful_doc_ids = [id_.strip() for id_
                              in useful_doc_ids_str.split(",") if id_.strip().isdigit()]
            useful_docs = [doc for doc in docs if doc.sid in useful_doc_ids]

        if not is_sufficient and not new_query:
            is_sufficient = True  # force to yes if no new query

        return is_sufficient, new_query, useful_docs, useful_docs_summary

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        async def stream():
            try:
                llm, reranker = await get_default_llms()
                acc_docs: List[SearchResult] = []
                acc_summaryies: List[str] = []
                acc_queries: List[str] = []
                next_query = request.question
                tries = 0
                while sum(calc_tokens(d) for d in acc_docs) < self.context_tokens and tries < self.max_tries:
                    tries += 1
                    # step 1: search
                    docs = await search_clueweb(next_query, k=10, cw22_a=self.cw22_a)
                    docs = [r for r in docs if isinstance(r, SearchResult)]
                    if not docs:
                        # query has issue, reformulate and continue
                        next_query = await reformulate_query(next_query)
                        continue

                    # step 2: rerank
                    docs_reranked = await reranker.rerank(next_query, docs)
                    docs_truncated = await atruncate_docs(docs_reranked, self.context_tokens)

                    # step 3: LLM to review
                    is_enough, _next_query, useful_docs, useful_docs_summary = await self.review_documents(request.question, next_query, acc_queries, acc_summaryies, docs_truncated)
                    acc_docs.extend(useful_docs)
                    acc_queries.append(next_query)

                    if useful_docs_summary:
                        acc_summaryies.append(useful_docs_summary)
                        yield inter_resp(f"Found documents: {useful_docs_summary}\n\n",
                                         silent=False, logger=self.logger)

                    if is_enough:
                        sum_tokens = sum(calc_tokens(d)
                                         for d in acc_docs)
                        if sum_tokens >= self.context_tokens:
                            acc_docs = await atruncate_docs(acc_docs, self.context_tokens)
                        break
                    else:
                        if not _next_query:
                            break
                        next_query = _next_query

                    yield inter_resp(f"Need more information, so far we have {len(acc_docs)} relevant documents\n\n",
                                     silent=False, logger=self.logger)
                    yield inter_resp(f"Next search({(tries+1)}/{self.max_tries}): {next_query}\n\n",
                                     silent=False, logger=self.logger)

                # truncate before answering
                acc_docs = await atruncate_docs(acc_docs, self.context_tokens)
                acc_docs = update_docs_sids(acc_docs)

                yield inter_resp("Starting final answer\n\n", silent=False, logger=self.logger)
                messages = build_llm_messages(
                    acc_docs, request.question, True)

                prompt_tokens = calc_tokens_str(json.dumps(messages))
                available_context = self.context_length - prompt_tokens - 1000
                async for chunk in llm.complete_chat_streaming(messages, max_tokens=available_context):
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
                    for r in acc_docs if isinstance(r, SearchResult)
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
    import sys
    import asyncio

    async def main():
        """Simple test execution for VanillaAgent."""
        print("Testing VanillaAgent with LLM server...")

        # Initialize VanillaAgent
        rag = VanillaAgent()

        try:
            q = sys.argv[1] if len(sys.argv) > 1 \
                else "What is the capital of France?"
            run_request = RunRequest(question=q)

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
                        print(f"\n\nCitations: {len(response.citations)}")
                    if response.error:
                        print(f"\n\nError: {response.error}")

        except Exception as e:
            print(f"Error during testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
