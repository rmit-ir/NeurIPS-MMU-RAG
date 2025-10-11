from typing import AsyncGenerator, Callable, List, Optional, Dict
import asyncio
import re
from openai.types.chat import ChatCompletionMessageParam
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.path_utils import to_icon_url
from tools.web_search import SearchResult, search_clueweb
from tools.logging_utils import get_logger


class DecompositionRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        gpu_memory_utilization: Optional[float] = 0.6,
        max_model_len: Optional[int] = 20000,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        search_results_k: int = 3,
        max_context_length: int = 3000,
        max_sub_queries: int = 5,
        cw22_a: bool = True,
    ):
        """
        Initialize DecompositionRAG with vLLM server and FineWeb search.

        Args:
            model_id: The model ID to use for vLLM server
            reasoning_parser: Parser for reasoning models
            gpu_memory_utilization: GPU memory utilization fraction
            max_model_len: Maximum model length
            api_key: API key for the server (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            search_results_k: Number of search results to retrieve per sub-query
            max_context_length: Maximum length of context per sub-query
            max_sub_queries: Maximum number of sub-queries to generate
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.search_results_k = search_results_k
        self.max_context_length = max_context_length
        self.max_sub_queries = max_sub_queries
        self.cw22_a = cw22_a

        self.logger = get_logger("decomposition_rag")
        self.llm_client = None

    async def _ensure_llm_client(self):
        if not self.llm_client:
            llm_mgr = get_llm_mgr(VllmConfig(model_id=self.model_id,
                                             reasoning_parser=self.reasoning_parser,
                                             gpu_memory_utilization=self.gpu_memory_utilization,
                                             max_model_len=self.max_model_len,
                                             api_key=self.api_key))
            self.llm_client = await llm_mgr.get_openai_client(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries."""
        try:
            self.logger.info("Decomposing query", query=query)
            await self._ensure_llm_client()
            if not self.llm_client:
                raise RuntimeError("LLM client is not initialized.")

            decomposition_prompt = f"""
You are an expert at breaking down complex questions into simpler, focused sub-questions.

Given this complex query: "{query}"

Break it down into 2-5 simpler, focused sub-questions that would help answer the original query comprehensively.
Each sub-question should be:
- Specific and focused
- Answerable with available information
- Non-overlapping where possible
- Listed as separate questions

Format your response as a numbered list:
1. First sub-question
2. Second sub-question
3. Third sub-question
...

Only output the numbered list, nothing else.
"""

            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": "You are a helpful assistant that decomposes complex queries."},
                {"role": "user", "content": decomposition_prompt}
            ]

            response, _ = await self.llm_client.complete_chat(messages)

            # Parse the numbered list
            lines = response.strip().split('\n') if response else []
            sub_queries = []
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    # Remove the number and dot
                    sub_query = re.sub(r'^\d+\.\s*', '', line).strip()
                    if sub_query:
                        sub_queries.append(sub_query)

            # Limit to max_sub_queries
            sub_queries = sub_queries[:self.max_sub_queries]

            self.logger.info("Decomposed into sub-queries",
                             count=len(sub_queries), sub_queries=sub_queries)
            return sub_queries

        except Exception as e:
            self.logger.error("Error decomposing query", error=str(e))
            # Fallback: return the original query as a single sub-query
            return [query]

    async def _retrieve_documents(self, query: str) -> List[Dict[str, str]]:
        """Retrieve relevant documents using ClueWeb search."""
        try:
            self.logger.info("Searching ClueWeb", query=query,
                             k=self.search_results_k)
            search_results = await search_clueweb(query=query, k=self.search_results_k, cw22_a=self.cw22_a)
            search_results = [
                res for res in search_results if isinstance(res, SearchResult)]

            documents = []
            for result in search_results:
                content = result.text
                if content:
                    documents.append({
                        "content": content[:self.max_context_length],
                        "url": result.url,
                        "text": content,  # full text
                        "sid": result.sid,
                    })

            self.logger.info("Retrieved documents", count=len(documents))
            return documents

        except Exception as e:
            self.logger.error("Error retrieving documents", error=str(e))
            return []

    def _format_context(self, documents: List[Dict[str, str]]) -> str:
        """Format retrieved documents into context for the prompt."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for doc in documents:
            sid = doc.get("sid", "unknown")
            content = doc.get("content", "")
            url = doc.get("url", "")

            context_part = f"[{sid}]\n{content}"
            if url:
                context_part += f"\nSource: {url}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    async def _answer_sub_query(self, sub_query: str, context: str) -> str:
        """Generate an answer for a single sub-query using the provided context."""
        try:
            if not self.llm_client:
                raise RuntimeError("LLM client is not initialized.")
            system_message = (
                "You are a helpful AI assistant. Answer the question using only the provided context. "
                "Be concise but comprehensive. If the context doesn't contain relevant information, "
                "say so clearly. Cite sources using [sid] when possible."
                "/nothink"
            )

            user_message = f"Context:\n{context}\n\nQuestion: {sub_query}"

            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            response, _ = await self.llm_client.complete_chat(messages)
            return response.strip() if response else "No response generated"

        except Exception as e:
            self.logger.error("Error answering sub-query", error=str(e))
            return f"Error answering sub-query: {str(e)}"

    def _prepare_synthesis_messages(self, original_query: str, sub_queries: List[str], sub_answers: List[str]) -> List[ChatCompletionMessageParam]:
        """Prepare messages for synthesizing individual sub-query answers into a comprehensive final answer."""
        synthesis_prompt = f"""
Original Query: {original_query}

Sub-questions and their answers:
"""

        for i, (sub_query, answer) in enumerate(zip(sub_queries, sub_answers), 1):
            synthesis_prompt += f"\n{i}. {sub_query}\nAnswer: {answer}\n"

        synthesis_prompt += """
Based on the above sub-question answers, provide a comprehensive and well-structured final answer to the original query.
Synthesize the information coherently, avoid redundancy, and ensure the answer is complete.
If there are any contradictions or gaps, note them clearly.
Cite sources using [sid] when possible.
"""

        return [
            {"role": "system", "content": "You are an expert at synthesizing information from multiple sources into coherent answers."},
            {"role": "user", "content": synthesis_prompt}
        ]

    async def _synthesize_answers(self, original_query: str, sub_queries: List[str], sub_answers: List[str]) -> str:
        """Synthesize individual sub-query answers into a comprehensive final answer."""
        try:
            if not self.llm_client:
                raise RuntimeError("LLM client is not initialized.")

            messages = self._prepare_synthesis_messages(
                original_query, sub_queries, sub_answers)
            final_answer, _ = await self.llm_client.complete_chat(messages)
            return final_answer.strip() if final_answer else "Answer unavailable"

        except Exception as e:
            self.logger.error("Error synthesizing answers", error=str(e))
            # Fallback: just concatenate the answers
            return "\n\n".join([f"Q{i+1}: {q}\nA: {a}" for i, (q, a) in enumerate(zip(sub_queries, sub_answers))])

    @property
    def name(self) -> str:
        return "decomposition-rag"

    async def _process_sub_query(self, i: int, sub_queries: List[str], sub_query: str) -> tuple[int, str, str, List[Dict[str, str]]]:
        self.logger.info(
            f"Processing sub-query {i+1}/{len(sub_queries)}", sub_query=sub_query)

        # Retrieve documents for this sub-query
        documents = await self._retrieve_documents(sub_query)

        # Format context
        context = self._format_context(documents)

        # Generate answer for this sub-query
        answer = await self._answer_sub_query(sub_query, context)

        return i, sub_query, answer, documents

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request using decomposition RAG.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with generated answer
        """
        self._is_processing = True
        try:
            await self._ensure_llm_client()
            if not self.llm_client:
                raise RuntimeError("LLM client is not initialized.")

            # Step 1: Decompose the query
            sub_queries = await self._decompose_query(request.query)
            self.logger.info("Query decomposed", sub_queries=sub_queries)

            # Step 2: Answer each sub-query in parallel
            sub_answers = []
            all_documents = []

            # Process all sub-queries concurrently
            tasks = [self._process_sub_query(i, sub_queries, sub_query)
                     for i, sub_query in enumerate(sub_queries)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle any exceptions
            # Initialize with correct size and type
            sub_answers: List[str] = [""] * len(sub_queries)
            for idx, result in enumerate(results):
                if isinstance(result, Exception) or isinstance(result, BaseException):
                    self.logger.error(
                        "Error processing sub-query", error=str(result))
                    # Handle the exception by using a fallback answer
                    sub_answers[idx] = f"Error processing sub-query {idx+1}: {str(result)}"
                else:
                    i, sub_query, answer, documents = result
                    sub_answers[i] = answer
                    all_documents.extend(documents)

            # Fill any empty values with error messages (in case of exceptions)
            for i, answer in enumerate(sub_answers):
                if not answer:
                    sub_answers[i] = f"Error processing sub-query {i+1}"

            # Step 3: Synthesize final answer
            final_answer = await self._synthesize_answers(request.query, sub_queries, sub_answers)

            # Deduplicate documents by sid
            global_docs = {}
            for doc in all_documents:
                sid = doc.get('sid')
                if sid and sid not in global_docs:
                    global_docs[sid] = doc

            # Extract citations and contexts
            valid_docs = [doc for doc in global_docs.values() if doc.get("url") and doc.get("text")]
            citations = [doc["url"] for doc in valid_docs]
            contexts = [doc["text"] for doc in valid_docs]

            return EvaluateResponse(
                query_id=request.iid,
                citations=citations,
                contexts=contexts,
                generated_response=final_answer
            )

        except Exception as e:
            self.logger.error("Error in evaluate", error=str(e))
            return EvaluateResponse(
                query_id=request.iid,
                citations=[],
                contexts=[],
                generated_response=f"Error processing query: {str(e)}"
            )
        finally:
            self._is_processing = False

    def _inter_resp(self, desc: str, silent: bool = False) -> RunStreamingResponse:
        if not silent:
            self.logger.info(f"Intermediate step | {desc}")
        return RunStreamingResponse(
            intermediate_steps=desc,
            is_intermediate=True,
            complete=False
        )

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request using decomposition RAG.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function for streaming responses
        """
        async def stream():
            self._is_processing = True
            try:
                yield self._inter_resp("Preparing to answer...\n\n")
                await self._ensure_llm_client()
                if not self.llm_client:
                    raise RuntimeError("LLM server failed to launch")

                yield self._inter_resp("Decomposing complex query into sub-questions...\n\n")

                # Step 1: Decompose the query
                sub_queries = await self._decompose_query(request.question)

                yield self._inter_resp(f"Query decomposed into {len(sub_queries)} sub-questions. Processing each sub-question...\n\n")

                # Step 2: Answer each sub-query in parallel
                sub_answers = []
                all_documents = []

                yield self._inter_resp(f"Processing {len(sub_queries)} sub-questions...\n\n")

                # Process all sub-queries concurrently
                tasks = [self._process_sub_query(i, sub_queries, sub_query)
                         for i, sub_query in enumerate(sub_queries)]

                # Use asyncio.as_completed to show progress as tasks complete
                completed_count = 0
                sub_answers: List[str] = [""] * len(sub_queries)

                for coro in asyncio.as_completed(tasks):
                    try:
                        i, sub_query, answer, documents = await coro
                        sub_answers[i] = answer
                        all_documents.extend(documents)
                        completed_count += 1

                        yield self._inter_resp(f"✓ Completed sub-question {completed_count}/{len(sub_queries)}: {sub_query}\n\n")

                    except Exception as e:
                        self.logger.error(
                            "Error processing sub-query in streaming", error=str(e))
                        completed_count += 1
                        yield self._inter_resp(f"✗ Error in sub-question ({completed_count}/{len(sub_queries)} total)\n\n")

                # Fill any empty values with error messages (in case of exceptions)
                for i, answer in enumerate(sub_answers):
                    if not answer:
                        sub_answers[i] = f"Error processing sub-query {i+1}"

                yield self._inter_resp("Synthesizing comprehensive answer from all sub-question responses...\n\n")

                # Step 3: Stream the synthesis process
                messages = self._prepare_synthesis_messages(
                    request.question, sub_queries, sub_answers)

                yield self._inter_resp("Starting final answer\n\n")

                # Stream the final synthesis using complete_chat_streaming like vanilla_rag
                async for chunk in self.llm_client.complete_chat_streaming(messages):
                    if chunk.choices[0].finish_reason is not None:
                        # Stream finished
                        break
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        yield self._inter_resp(delta.reasoning_content, silent=True)
                    elif hasattr(delta, 'content') and delta.content:
                        yield RunStreamingResponse(
                            final_report=delta.content,
                            is_intermediate=False,
                            complete=False
                        )
                    # otherwise ignore empty deltas

                # Deduplicate documents by sid
                global_docs = {}
                for doc in all_documents:
                    sid = doc.get('sid')
                    if sid and sid not in global_docs:
                        global_docs[sid] = doc

                # Extract citations
                citations = []
                unique_urls = set()
                for doc in global_docs.values():
                    if doc.get("url") and doc.get("text") and doc["url"] not in unique_urls:
                        unique_urls.add(doc["url"])
                        citations.append(CitationItem(
                            url=doc["url"],
                            icon_url=to_icon_url(doc["url"]),
                            date=None,
                            title=None,
                            sid=doc.get("sid"),
                            text=doc["text"],
                        ))

                # Final response
                yield RunStreamingResponse(
                    citations=citations,
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


if __name__ == "__main__":
    import asyncio

    async def main():
        """Simple test execution for DecompositionRAG."""
        print("Testing DecompositionRAG with FineWeb search...")

        # Initialize DecompositionRAG
        rag = DecompositionRAG(
            model_id="Qwen/Qwen3-4B",
            api_key=None,
            temperature=0.0,
            max_tokens=4096,
            search_results_k=2,  # Fewer results per sub-query
            max_sub_queries=3
        )

        try:
            # Test with a complex query
            print("\n=== Testing Evaluate Method ===")
            eval_request = EvaluateRequest(
                query="What are the main differences between machine learning and deep learning, and how do they relate to artificial intelligence?",
                iid="test-001"
            )

            eval_response = await rag.evaluate(eval_request)
            print(f"Query ID: {eval_response.query_id}")
            print(f"Response: {eval_response.generated_response}")
            print(f"Citations: {eval_response.citations}")

            # Test streaming
            print("\n=== Testing Streaming Method ===")
            run_request = RunRequest(
                question="Explain the impact of climate change on biodiversity and what measures can be taken to mitigate it."
            )

            stream_func = await rag.run_streaming(run_request)
            print("Streaming response:")

            async for response in stream_func():
                if response.is_intermediate:
                    if response.intermediate_steps:
                        print(response.intermediate_steps, end="", flush=True)
                else:
                    if response.final_report:
                        print(f"\n[FINAL ANSWER]\n{response.final_report}\n")
                    if response.citations:
                        print(f"Citations: {response.citations}")
                    if response.error:
                        print(f"Error: {response.error}")

        except Exception as e:
            print(f"Error during testing: {str(e)}")

    asyncio.run(main())
