from typing import AsyncGenerator, Callable, List, Optional, Dict
import asyncio
import re
from openai.types.chat import ChatCompletionMessageParam
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse
from tools.llm_servers.sglang_server import launch_server, terminate_server
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.web_search import fineweb_search
from tools.logging_utils import get_logger


class DecompositionRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        mem_fraction_static: Optional[float] = 0.4,
        max_running_requests: Optional[int] = 4,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        search_results_k: int = 3,
        max_context_length: int = 3000,
        max_sub_queries: int = 5,
    ):
        """
        Initialize DecompositionRAG with SGLang server and FineWeb search.

        Args:
            model_id: The model ID to use for SGLang server
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            search_results_k: Number of search results to retrieve per sub-query
            max_context_length: Maximum length of context per sub-query
            max_sub_queries: Maximum number of sub-queries to generate
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.search_results_k = search_results_k
        self.max_context_length = max_context_length
        self.max_sub_queries = max_sub_queries

        self.logger = get_logger("decomposition_rag")
        self.server_process = None
        self.client = None
        self.server_host = None
        self.api_base = None
        self._is_processing = False
        self._is_llm_starting = False

    async def _ensure_server_running(self):
        """Ensure SGLang server is running and client is initialized."""
        if not (self.server_process and self.client):
            if not self._is_llm_starting:
                try:
                    self._is_llm_starting = True
                    self.logger.info("Starting SGLang server", model_id=self.model_id)
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
                        llm_name="decomposition_rag_sglang"
                    )
                    self.logger.info("SGLang server and client initialized", port=port)
                finally:
                    self._is_llm_starting = False

        max_wait_time = 3600
        wait_time = 0
        while not (self.client and self.server_host):
            if wait_time >= max_wait_time:
                raise RuntimeError(f"Server failed to initialize within {max_wait_time} seconds")
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

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries."""
        try:
            self.logger.info("Decomposing query", query=query)

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

            messages = [
                {"role": "system", "content": "You are a helpful assistant that decomposes complex queries."},
                {"role": "user", "content": decomposition_prompt}
            ]

            response, _ = self.client.complete_chat(messages)

            # Parse the numbered list
            sub_queries = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    # Remove the number and dot
                    sub_query = re.sub(r'^\d+\.\s*', '', line).strip()
                    if sub_query:
                        sub_queries.append(sub_query)

            # Limit to max_sub_queries
            sub_queries = sub_queries[:self.max_sub_queries]

            self.logger.info("Decomposed into sub-queries", count=len(sub_queries), sub_queries=sub_queries)
            return sub_queries

        except Exception as e:
            self.logger.error("Error decomposing query", error=str(e))
            # Fallback: return the original query as a single sub-query
            return [query]

    async def _retrieve_documents(self, query: str) -> List[Dict[str, str]]:
        """Retrieve relevant documents using FineWeb search."""
        try:
            self.logger.info("Searching FineWeb", query=query, k=self.search_results_k)
            search_results = await fineweb_search(query=query, k=self.search_results_k)

            documents = []
            for result in search_results:
                if isinstance(result, dict) and "_error" not in result:
                    title = result.get("title", "")
                    content = result.get("text", result.get("content", ""))
                    url = result.get("url", "")

                    if content:
                        documents.append({
                            "title": title,
                            "content": content[:self.max_context_length],
                            "url": url
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
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"Document {i}")
            content = doc.get("content", "")
            url = doc.get("url", "")

            context_part = f"[{i}] {title}\n{content}"
            if url:
                context_part += f"\nSource: {url}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    async def _answer_sub_query(self, sub_query: str, context: str) -> str:
        """Generate an answer for a single sub-query using the provided context."""
        try:
            system_message = (
                "You are a helpful AI assistant. Answer the question using only the provided context. "
                "Be concise but comprehensive. If the context doesn't contain relevant information, "
                "say so clearly. Cite sources using document numbers [1], [2], etc. when possible."
            )

            user_message = f"Context:\n{context}\n\nQuestion: {sub_query}"

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            response, _ = self.client.complete_chat(messages)
            return response.strip()

        except Exception as e:
            self.logger.error("Error answering sub-query", error=str(e))
            return f"Error answering sub-query: {str(e)}"

    async def _synthesize_answers(self, original_query: str, sub_queries: List[str], sub_answers: List[str]) -> str:
        """Synthesize individual sub-query answers into a comprehensive final answer."""
        try:
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
"""

            messages = [
                {"role": "system", "content": "You are an expert at synthesizing information from multiple sources into coherent answers."},
                {"role": "user", "content": synthesis_prompt}
            ]

            final_answer, _ = self.client.complete_chat(messages)
            return final_answer.strip()

        except Exception as e:
            self.logger.error("Error synthesizing answers", error=str(e))
            # Fallback: just concatenate the answers
            return "\n\n".join([f"Q{i+1}: {q}\nA: {a}" for i, (q, a) in enumerate(zip(sub_queries, sub_answers))])

    @property
    def name(self) -> str:
        return "decomposition-rag"

    @property
    def is_running(self) -> bool:
        """Check if the SGLang server is running."""
        return self.server_process is not None and self.client is not None

    @property
    def is_processing(self) -> bool:
        """Check if the system is currently processing a request."""
        return self._is_processing

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
            await self._ensure_server_running()
            if not self.client:
                raise RuntimeError("SGLang client is not initialized.")

            # Step 1: Decompose the query
            sub_queries = await self._decompose_query(request.query)
            self.logger.info("Query decomposed", sub_queries=sub_queries)

            # Step 2: Answer each sub-query
            sub_answers = []
            all_documents = []

            for i, sub_query in enumerate(sub_queries):
                self.logger.info(f"Processing sub-query {i+1}/{len(sub_queries)}", sub_query=sub_query)

                # Retrieve documents for this sub-query
                documents = await self._retrieve_documents(sub_query)
                all_documents.extend(documents)

                # Format context
                context = self._format_context(documents)

                # Generate answer for this sub-query
                answer = await self._answer_sub_query(sub_query, context)
                sub_answers.append(answer)

            # Step 3: Synthesize final answer
            final_answer = await self._synthesize_answers(request.query, sub_queries, sub_answers)

            # Extract citations
            citations = []
            for doc in all_documents:
                if doc.get("url"):
                    citations.append(doc["url"])

            return EvaluateResponse(
                query_id=request.iid,
                citations=list(set(citations)),  # Remove duplicates
                generated_response=final_answer
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
        Process a streaming request using decomposition RAG.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function for streaming responses
        """
        async def stream():
            self._is_processing = True
            try:
                yield RunStreamingResponse(
                    intermediate_steps="Initializing SGLang server...",
                    is_intermediate=True,
                    complete=False
                )

                await self._ensure_server_running()
                if not self.client:
                    raise RuntimeError("SGLang server failed to launch")

                yield RunStreamingResponse(
                    intermediate_steps="Decomposing complex query into sub-questions...",
                    is_intermediate=True,
                    complete=False
                )

                # Step 1: Decompose the query
                sub_queries = await self._decompose_query(request.question)

                yield RunStreamingResponse(
                    intermediate_steps=f"Query decomposed into {len(sub_queries)} sub-questions. Processing each sub-question...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                # Step 2: Answer each sub-query
                sub_answers = []
                all_documents = []

                for i, sub_query in enumerate(sub_queries):
                    yield RunStreamingResponse(
                        intermediate_steps=f"Processing sub-question {i+1}/{len(sub_queries)}: {sub_query}\n",
                        is_intermediate=True,
                        complete=False
                    )

                    # Retrieve documents
                    documents = await self._retrieve_documents(sub_query)
                    all_documents.extend(documents)
                    context = self._format_context(documents)

                    # Generate answer
                    answer = await self._answer_sub_query(sub_query, context)
                    sub_answers.append(answer)

                    yield RunStreamingResponse(
                        intermediate_steps=f"✓ Completed sub-question {i+1}\n",
                        is_intermediate=True,
                        complete=False
                    )

                yield RunStreamingResponse(
                    intermediate_steps="Synthesizing comprehensive answer from all sub-question responses...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                # Step 3: Synthesize final answer
                final_answer = await self._synthesize_answers(request.question, sub_queries, sub_answers)

                # Stream the final answer
                yield RunStreamingResponse(
                    final_report=final_answer,
                    is_intermediate=False,
                    complete=False
                )

                # Extract citations
                citations = []
                for doc in all_documents:
                    if doc.get("url"):
                        citations.append(doc["url"])

                # Final response
                yield RunStreamingResponse(
                    citations=list(set(citations)),  # Remove duplicates
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
        finally:
            print(f"\nFinal status - Is running: {rag.is_running}, Is processing: {rag.is_processing}")
            rag._shutdown_server()
            print("Test completed.")

    asyncio.run(main())