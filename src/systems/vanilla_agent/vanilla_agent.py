import asyncio
import json
from datetime import datetime, timezone
from openai.types.chat import ChatCompletionMessageParam
from typing import Any, AsyncGenerator, Callable, Dict, List, NamedTuple, Optional, Tuple
from systems.rag_interface import RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from systems.vanilla_agent.rag_util_fn import build_llm_messages, build_to_context, get_default_llms, inter_resp, reformulate_query, search_w_qv
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.path_utils import to_icon_url
from tools.reranker_vllm import GeneralReranker, _dummy_search_result as _search_result
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult
from tools.docs_utils import atruncate_docs, calc_tokens, calc_tokens_str, update_docs_sids


class QueryHistoryItem(NamedTuple):
    """Represents a single search query and its results in the query history."""
    query: str
    doc_count: int
    summary: str


class VanillaAgent(RAGInterface):
    def __init__(
        self,
        context_length: int = 25_000,  # LLM context length in tokens
        docs_review_max_tokens: int = 4096,
        answer_max_tokens: int = 4096,
        num_qvs: int = 5,  # number of query variants to use in search
        max_tries: int = 5,
        cw22_a: bool = True,
        alt_llm_api_base: Optional[str] = None,
        alt_llm_api_key: Optional[str] = None,
        alt_llm_model: Optional[str] = None,
        alt_llm_reasoning_effort: Optional[str] = None,
        alt_reranker_api_base: Optional[str] = None,
        alt_reranker_api_key: Optional[str] = None,
        alt_reranker_model: Optional[str] = None,
        pre_flight_llm: bool = False,
        pre_flight_reranker: bool = False,
    ):
        """
        Initialize VanillaAgent with LLM server.
        """
        self.context_length = context_length
        self.docs_review_max_tokens = docs_review_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.num_qvs = num_qvs
        self.max_tries = max_tries
        self.cw22_a = cw22_a
        self.alt_llm_api_base = alt_llm_api_base
        self.alt_llm_api_key = alt_llm_api_key
        self.alt_llm_model = alt_llm_model
        self.alt_llm_reasoning_effort: Any = alt_llm_reasoning_effort
        self.alt_reranker_api_base = alt_reranker_api_base
        self.alt_reranker_api_key = alt_reranker_api_key
        self.alt_reranker_model = alt_reranker_model
        self.pre_flight_llm = pre_flight_llm
        self.pre_flight_reranker = pre_flight_reranker

        self.logger = get_logger("vanilla_agent")
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

        self.logger.info("Initialized VanillaAgent",
                         context_length=self.context_length,
                         docs_review_max_tokens=self.docs_review_max_tokens,
                         answer_max_tokens=self.answer_max_tokens,
                         num_qvs=self.num_qvs,
                         max_tries=self.max_tries,
                         cw22_a=self.cw22_a,
                         alt_llm_api_base=self.alt_llm_api_base,
                         alt_llm_api_key=self.alt_llm_api_key,
                         alt_llm_model=self.alt_llm_model,
                         alt_llm_reasoning_effort=self.alt_llm_reasoning_effort,
                         alt_reranker_api_base=self.alt_reranker_api_base,
                         alt_reranker_api_key=self.alt_reranker_api_key,
                         alt_reranker_model=self.alt_reranker_model)

    @property
    def name(self) -> str:
        return "vanilla-agent"

    def _format_query_history(self, query_history: List[QueryHistoryItem]) -> str:
        """Format query history into a readable string for the prompt.

        Args:
            query_history: List of QueryHistoryItem containing query, doc_count, and summary

        Returns:
            Formatted string describing the search history
        """
        if not query_history:
            return ""

        history_str = "\n=== SEARCH HISTORY ===\n"
        history_str += "Here is the complete history of our previous searches and what we found:\n\n"

        for i, entry in enumerate(query_history, 1):
            history_str += f"Search #{i}:\n"
            history_str += f"  Query: \"{entry.query}\"\n"
            history_str += f"  Result: Found {entry.doc_count} useful document(s)\n"
            history_str += f"  Summary: {entry.summary}\n\n"

        history_str += "=== END SEARCH HISTORY ===\n\n"
        history_str += "IMPORTANT: Do not repeat these queries in <new-query>. Consider what information we already have vs what is still missing.\n\n"

        return history_str

    async def get_default_llms(self):
        if self.alt_llm_api_base and self.alt_llm_model:
            alt_llm = GeneralOpenAIClient(model_id=self.alt_llm_model,
                                          api_base=self.alt_llm_api_base,
                                          api_key=self.alt_llm_api_key,
                                          reasoning_effort=self.alt_llm_reasoning_effort,
                                          max_retries=3)
        if self.alt_reranker_api_base and self.alt_reranker_model:
            alt_reranker = GeneralReranker(model_id=self.alt_reranker_model,
                                           api_base=self.alt_reranker_api_base,
                                           api_key=self.alt_reranker_api_key)
        if (self.alt_llm_api_base and self.alt_llm_model and
                self.alt_reranker_api_base and self.alt_reranker_model):
            self.logger.info("Using alternative LLM and Reranker for VanillaAgent",
                             alt_llm_api_base=self.alt_llm_api_base,
                             alt_reranker_api_base=self.alt_reranker_api_base)
            return alt_llm, alt_reranker

        llm, reranker = await get_default_llms()
        if self.alt_llm_api_base and self.alt_llm_model:
            return alt_llm, reranker
        if self.alt_reranker_api_base and self.alt_reranker_model:
            return llm, alt_reranker
        return llm, reranker

    async def review_documents(self, question: str, next_query: str, query_history: List[QueryHistoryItem], docs: List[SearchResult]) -> Tuple[bool, str | None, List[SearchResult], str | None]:
        llm, reranker = await self.get_default_llms()
        GRADE_PROMPT = """You are an expert in answering user question "{question}". We are doing research on user's question and currently working on aspect "{next_query}"

Current time at UTC+00:00 timezone: {current_time}

Go through each document in the search results, judge if they are sufficient for answering the user question. Please consider:

1. Does the user want a simple answer or a comprehensive explanation? For comprehensive explanation, we may need searching with different aspects to cover a wider range of perspectives.
2. Does the search results fully addresses the user's query and any sub-components?
3. For controversial, convergent, divergent, evaluative, complex systemic, ethical-moral, problem-solving, recommendation questions that will benefit from multiple aspects, try to search from different aspects to form a balanced, comprehensive view.
4. Don't mention irrelevant, off-topic documents.
5. When you answer 'yes' in <is-sufficient>, we will proceed to generate the final answer based on these results. If you answer 'no', we will continue the next turn of using your new query to search, and let you review again.
6. If information is missing or uncertain, always return 'no' in <is-sufficient> xml tags for clarification, and generate a new query enclosed in xml markup <new-query>your query</new-query> indicating the clarification needed. If the search results are too off, try clarifying sub-components of the question first, or make reasonable guess. If you think the search results are sufficient, return 'yes' in <is-sufficient> xml tags.
7. If current search results have very limited information, try use different techniques, query expansion, query relaxation, query segmentation, use different synonyms, use reasonable guess and different keywords to get relevant results, and put the new query in <>new-query>your query</new-query> xml tags. If there are previous search queries, do not repeat them in the new query, we know they don't work.
8. Identify unique, new documents that are important for answering the question but not included in previous documents, and list their IDs (# in ID=[#]) in a comma-separated format within <useful-docs> xml tags. If multiple documents are similar, choose the one with better quality. Do not provide duplicated documents that have been included in previous turns. If no new documents are useful, leave <useful-docs></useful-docs> empty.
9. If useful-docs is not empty, provide a brief summary of what these documents discuss within <useful-docs-summary> xml tags, in 1-2 sentences and mention what is still missing. Start your summary with "These documents discuss...". Do not mention specific document IDs in the summary.
10. Your purpose is to judge documents relevance against the question, not to provide the final answer yet, do not answer the question.

Response format:

- <is-sufficient>yes or no</is-sufficient> (For all of the documents we have collected, including previous documents, do we have enough information to answer the user question? If yes, then provide <useful-docs> and <useful-docs-summary> tags; if 'no', then provide <new-query> tag)
- <new-query>your new query</new-query> (only if is-sufficient is 'no')
- <useful-docs>1,2,3</useful-docs> (list of document IDs that are useful for answering the question, separated by commas)
- <useful-docs-summary></useful-docs-summary> (short summary of what these useful documents are talking about and what is missing, just the summary, only if useful-docs is not empty)

{query_history_section}

Here is the current search query: "{next_query}"
Here is the search results for current search query:

"""
        query_history_str = self._format_query_history(query_history)

        prompt = GRADE_PROMPT.format(
            question=question,
            next_query=next_query,
            current_time=datetime.now(timezone.utc),
            query_history_section=query_history_str)
        prompt_tokens = calc_tokens_str(prompt)

        # truncate docs to fit in context
        answer_max_tokens = self.docs_review_max_tokens
        redundant_tokens = 1024  # for doc header, prompt template, and safety margin overhead
        available_context = self.context_length - \
            prompt_tokens - answer_max_tokens - redundant_tokens
        docs_truncated = await atruncate_docs(docs, available_context)
        context = build_to_context(docs_truncated)
        prompt += context
        self.logger.info("Truncate documents for review",
                         model_context_length=self.context_length,
                         prompt_tokens=prompt_tokens,
                         answer_max_tokens=answer_max_tokens,
                         available_context=available_context,
                         original_count=len(docs),
                         truncated_count=len(docs_truncated),
                         actual_tokens=calc_tokens_str(prompt),
                         IDs=[d.sid for d in docs_truncated])

        messages: List[ChatCompletionMessageParam] = [
            {"role": "user", "content": prompt}
        ]
        resp_text = ""
        async for chunk in llm.complete_chat_streaming(messages, max_tokens=answer_max_tokens):
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
                         query_history=query_history,
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

    async def pre_flight_models(self) -> None:
        llm, reranker = await self.get_default_llms()
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

                yield inter_resp(f"Searching question: {request.question}\n\n",
                                 silent=False, logger=self.logger)
                llm, reranker = await self.get_default_llms()
                acc_docs: List[SearchResult] = []
                acc_docs_id_set = set()
                acc_doc_base_count = 0
                # Track complete query-result history
                query_history: List[QueryHistoryItem] = []
                next_query = request.question
                qv_think_enabled = False
                tries = 0
                # 5096 is reserved for answer gen
                answer_max_tokens = self.answer_max_tokens + 1024
                context_tokens_limit = self.context_length - answer_max_tokens
                # ---------------------------------------------------------------
                # AGENT LOOP
                while sum(calc_tokens(d) for d in acc_docs) < context_tokens_limit and tries < self.max_tries:
                    tries += 1
                    # step 1: search
                    qvs, docs = await search_w_qv(next_query, num_qvs=self.num_qvs, enable_think=qv_think_enabled, logger=self.logger, preset_llm=llm)
                    docs = [r for r in docs if isinstance(r, SearchResult)]
                    qvs_str = "; ".join(qvs)
                    yield inter_resp(f"Search completed: {qvs_str}\n\n",
                                     silent=False, logger=self.logger)

                    # step 2: rerank
                    yield inter_resp(f"Reranking {len(docs)} documents...\n\n",
                                     silent=False, logger=self.logger)
                    docs_reranked = await reranker.rerank(next_query, docs)

                    if not docs_reranked:
                        # ---- error case
                        # query has issue, reformulate and continue, this should be rare since we are searching with query variants
                        qv_think_enabled = True
                        yield inter_resp(f"Found no relevant documents, so far we have {len(acc_docs)} relevant documents, reformulating query...\n\n",
                                         silent=False, logger=self.logger)
                        next_query = await reformulate_query(next_query, preset_llm=llm)
                        yield inter_resp(f"Next search ({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                        continue

                    # step 3: LLM to review
                    # docs_truncated = await atruncate_docs(docs_reranked, self.context_tokens)
                    docs_reranked = update_docs_sids(
                        # base_count: every time the id will start from previous max + 1
                        docs_reranked, base_count=acc_doc_base_count)
                    yield inter_resp("Reviewing documents for relevance and sufficiency...\n\n",
                                     silent=False, logger=self.logger)
                    is_enough, _next_query, useful_docs, useful_docs_summary = await self.review_documents(request.question, next_query, query_history, docs_reranked)

                    # Record this query and its results in history
                    query_history.append(QueryHistoryItem(
                        query=next_query,
                        doc_count=len(useful_docs),
                        summary=useful_docs_summary if useful_docs_summary else 'No relevant documents found'
                    ))

                    # update acc_doc_base_count and acc_docs
                    acc_doc_base_count += len(useful_docs)
                    for d in useful_docs:
                        if d.id not in acc_docs_id_set:
                            acc_docs_id_set.add(d.id)
                            acc_docs.append(d)

                    if useful_docs_summary:
                        yield inter_resp(f"Found documents: {useful_docs_summary}\n\n",
                                         silent=False, logger=self.logger)

                    # we are done
                    if is_enough:
                        break

                    if not _next_query:
                        # ---- error case
                        # review had problem, not enough info, and no next query, we have to continue, same as above error case
                        qv_think_enabled = True
                        yield inter_resp(f"Found no relevant documents for this query, so far we have {len(acc_docs)} relevant documents\n\n",
                                         silent=False, logger=self.logger)
                        next_query = await reformulate_query(next_query, preset_llm=llm)
                        yield inter_resp(f"Next search with better query variants ({(tries)}/{self.max_tries}): {next_query}\n\n",
                                         silent=False, logger=self.logger)
                        continue

                    # continue next turn
                    next_query = _next_query
                    yield inter_resp(f"Need more information, so far we have {len(acc_docs)} relevant documents\n\n",
                                     silent=False, logger=self.logger)
                    yield inter_resp(f"Next search({(tries)}/{self.max_tries}): {next_query}\n\n",
                                     silent=False, logger=self.logger)
                # END AGENT LOOP
                # ---------------------------------------------------------------

                # truncate before answering
                acc_docs = await atruncate_docs(acc_docs, context_tokens_limit)
                acc_docs = update_docs_sids(acc_docs)

                yield inter_resp(f"Starting final answer with {len(acc_docs)} documents\n\n",
                                 silent=False, logger=self.logger)
                messages = build_llm_messages(
                    acc_docs, request.question, True)

                prompt_tokens = calc_tokens_str(json.dumps(messages))
                gen_max_tokens = self.context_length - prompt_tokens - 1000
                async for chunk in llm.complete_chat_streaming(messages, max_tokens=gen_max_tokens):
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
