import asyncio
import json
import re
from openai.types.chat import ChatCompletionMessageParam
from typing import AsyncGenerator, Callable, List, Optional, Tuple
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.rag_util_fn import build_llm_messages, build_to_context, get_default_llms, inter_resp, reformulate_query, search_w_qv
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult
from tools.docs_utils import atruncate_docs, calc_tokens, calc_tokens_str, update_docs_sids


class VanillaAgent(RAGInterface):
    def __init__(
        self,
        context_length: int = 20_000,
        context_tokens: int = 14_904,  # 20_000 - 5096 (for prompt and answer)
        num_qvs: int = 5,  # number of query variants to use in search
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
        self.num_qvs = num_qvs
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
        GRADE_PROMPT = """You are an expert in answering user question "{question}". We are doing research on user's question and currently working on aspect "{next_query}"

Go through each document in the search results, judge if they are sufficient for answering the user question. Please consider:

1. Does the user want a simple answer or a comprehensive explanation? For comprehensive explanation, we may need a wider range of documents from different aspects.
2. Does the search results fully addresses the user's query and any subcomponents?
3. For controversial, convergent, divergent, evaluative, complex systemic, ethical-moral, problem-solving, recommendation questions that will benefit from multiple aspects, try to tackle the question from different aspects to form a balanced, comprehensive view.
4. When you answer 'yes' in <is-sufficient>, we will proceed to generate the final answer based on these results. If you answer 'no', we will continue the next turn of using your new query to search, and let you review again.
5. If information is missing or uncertain, always return 'no' in <is-sufficient> xml tags for clarification, and generate a new query enclosed in xml markup <new-query>your query</new-query> indicating the clarification needed. If the search results are too off, try clarifying sub-components of the question first, or make reasonable guess. If you think the search results are sufficient, return 'yes' in <is-sufficient> xml tags.
6. Identify unique, new documents that are important for answering the question but not included in previous documents, and list their IDs (# in ID=[#]) in a comma-separated format within <useful-docs> xml tags. If multiple documents are similar, choose the one with better quality. Do not provide duplicated documents that have been included in previous turns. If no new documents are useful, leave <useful-docs></useful-docs> empty. Only compare to previous documents description, do not compare to their IDs, their IDs are using a different index and will not not match.
7. If useful-docs is not empty, provide a brief summary of what these documents discuss within <useful-docs-summary> xml tags, in 1-2 sentences.

Response format:

- <is-sufficient>yes or no</is-sufficient> (if yes, then provide <useful-docs> and <useful-docs-summary> tags; if 'no', then provide <new-query> tag)
- <new-query>your new query</new-query> (only if is-sufficient is 'no')
- <useful-docs>1,2,3</useful-docs> (list of document IDs that are useful for answering the question, separated by commas)
- <useful-docs-summary></useful-docs-summary> (short summary of what these useful documents are talking about, just the summary, only if useful-docs is not empty)

{prev_questions}
{prev_docs_summaries}
Here is the current question: "{next_query}"
Here is the search results for current question:

"""
        prompt = GRADE_PROMPT.format(
            question=question,
            next_query=next_query,
            prev_questions="Here are the previous questions we already tried: " +
            "; ".join(acc_queries) if acc_queries else "",
            prev_docs_summaries="Here are the summaries of previous documents we collected for this question, do not duplicate: " +
            "; ".join(acc_summaries) if acc_summaries else "")
        prompt_tokens = calc_tokens_str(prompt)

        # truncate docs to fit in context
        anser_max_tokens = 2048
        redundant_tokens = 1024  # for doc header, prompt template, and safety margin overhead
        available_context = self.context_length - \
            prompt_tokens - anser_max_tokens - redundant_tokens
        docs_truncated = await atruncate_docs(docs, available_context)
        context = build_to_context(docs_truncated)
        prompt += context
        self.logger.info("Truncate documents for review",
                         model_context_length=self.context_length,
                         prompt_tokens=prompt_tokens,
                         answer_max_tokens=anser_max_tokens,
                         available_context=available_context,
                         original_count=len(docs),
                         truncated_count=len(docs_truncated),
                         actual_tokens=calc_tokens_str(prompt),
                         IDs=[d.sid for d in docs_truncated])

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        resp_text = ""
        async for chunk in llm.complete_chat_streaming(messages, max_tokens=anser_max_tokens):
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
        useful_docs_summary = re.sub(r'[Dd]ocuments?\s*\d+\s*', '', useful_docs_summary) \
            if useful_docs_summary else None

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
                acc_docs_id_set = set()
                acc_doc_base_count = 0
                acc_summaryies: List[str] = []
                acc_queries: List[str] = []
                next_query = request.question
                qv_think_enabled = False
                tries = 0
                while sum(calc_tokens(d) for d in acc_docs) < self.context_tokens and tries < self.max_tries:
                    tries += 1
                    # step 1: search
                    qvs, docs = await search_w_qv(next_query, num_qvs=self.num_qvs, enable_think=qv_think_enabled, logger=self.logger)
                    docs = [r for r in docs if isinstance(r, SearchResult)]
                    qvs_str = ", ".join(qvs)
                    yield inter_resp(f"Search with query variants: {qvs_str}, found {len(docs)} documents\n\n",
                                     silent=False, logger=self.logger)

                    # step 2: rerank
                    docs_reranked = await reranker.rerank(next_query, docs)

                    if not docs_reranked:
                        # ---- error case
                        # query has issue, reformulate and continue, this should be rare since we are searching with query variants
                        next_query = await reformulate_query(next_query)
                        qv_think_enabled = True
                        yield inter_resp(f"Found no relevent documents for this query, so far we have {len(acc_docs)} relevant documents\n\n",
                                         silent=False, logger=self.logger)
                        yield inter_resp(f"Next search with better query variants ({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                        continue

                    # step 3: LLM to review
                    # docs_truncated = await atruncate_docs(docs_reranked, self.context_tokens)
                    docs_reranked = update_docs_sids(
                        # base_count: every time the id will start from previous max + 1
                        docs_reranked, base_count=acc_doc_base_count)
                    is_enough, _next_query, useful_docs, useful_docs_summary = await self.review_documents(request.question, next_query, acc_queries, acc_summaryies, docs_reranked)
                    acc_queries.append(next_query)

                    # update acc_doc_base_count and acc_docs
                    acc_doc_base_count += len(useful_docs)
                    for d in useful_docs:
                        if d.id not in acc_docs_id_set:
                            acc_docs_id_set.add(d.id)
                            acc_docs.append(d)

                    if useful_docs_summary:
                        acc_summaryies.append(useful_docs_summary)
                        yield inter_resp(f"Found documents: {useful_docs_summary}\n\n",
                                         silent=False, logger=self.logger)

                    # had enough documents to answer
                    sum_tokens = sum(calc_tokens(d) for d in acc_docs)
                    if sum_tokens >= self.context_tokens:
                        yield inter_resp(f"Read too much, let's answer with what we have so far\n\n",
                                         silent=False, logger=self.logger)
                        acc_docs = await atruncate_docs(acc_docs, self.context_tokens)
                        break

                    # we are done
                    if is_enough:
                        break
                    
                    if not _next_query:
                        # ---- error case
                        # review had problem, not enough info, and no next query, we have to continue, same as above error case
                        next_query = await reformulate_query(next_query)
                        qv_think_enabled = True
                        yield inter_resp(f"Found no relevent documents for this query, so far we have {len(acc_docs)} relevant documents\n\n",
                                         silent=False, logger=self.logger)
                        yield inter_resp(f"Next search with better query variants ({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                        continue

                    # continue next turn
                    next_query = _next_query
                    yield inter_resp(f"Need more information, so far we have {len(acc_docs)} relevant documents\n\n",
                                     silent=False, logger=self.logger)
                    yield inter_resp(f"Next search({(tries+1)}/{self.max_tries}): {next_query}\n\n",
                                     silent=False, logger=self.logger)

                # truncate before answering
                acc_docs = await atruncate_docs(acc_docs, self.context_tokens)
                acc_docs = update_docs_sids(acc_docs)

                yield inter_resp(f"Starting final answer with {len(acc_docs)} documents\n\n",
                                 silent=False, logger=self.logger)
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
            start_time = asyncio.get_event_loop().time()

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
                        print(f"\n[FINAL] {response.final_report}")
                    if response.final_report:
                        print(response.final_report, end="", flush=True)
                    if response.citations:
                        print(f"\n\nCitations: {len(response.citations)}")
                    if response.error:
                        print(f"\n\nError: {response.error}")
            end_time = asyncio.get_event_loop().time()
            print(f"\n\nTotal time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
