from typing import AsyncGenerator, Callable, List, Optional, Dict
import asyncio
import re
from openai.types.chat import ChatCompletionMessageParam
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse, CitationItem
from tools.llm_servers.vllm_server import get_llm_mgr
from tools.path_utils import to_icon_url
from tools.web_search import SearchResult, search_clueweb
from tools.logging_utils import get_logger
from sentence_transformers import SentenceTransformer
import numpy as np


class GRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        gpu_memory_utilization: Optional[float] = 0.6,
        max_model_len: Optional[int] = 20000,
        max_running_requests: Optional[int] = 4,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        search_results_k: int = 3,
        max_context_length: int = 3000,
        max_sub_queries: int = 5,
        # Model for sentence transformer reranking
        rerank_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_top_k: int = 3,
        # Options: "sentence_transformer", "logits"
        reranker: str = "sentence_transformer",
    ):
        """
        Initialize GRAG with decomposition and HYDE.

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
            rerank_model: Model name for sentence transformer reranking
            rerank_top_k: Number of top documents to keep after reranking
            reranker: Type of reranker to use ("sentence_transformer" or "logits")
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_running_requests = max_running_requests
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.search_results_k = search_results_k
        self.max_context_length = max_context_length
        self.max_sub_queries = max_sub_queries
        self.rerank_model = rerank_model
        self.rerank_top_k = rerank_top_k
        self.reranker = reranker

        self.logger = get_logger("grag")
        self.llm_client = None
        self.rerank_model_instance = None

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

    def _ensure_rerank_model(self):
        if not self.rerank_model_instance:
            self.logger.info("Loading reranking model",
                             model=self.rerank_model)
            self.rerank_model_instance = SentenceTransformer(self.rerank_model)

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries."""
        try:
            self.logger.info("Decomposing query", query=query)
            await self._ensure_llm_client()
            if not self.llm_client:
                raise RuntimeError("SGLang client is not initialized.")

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

    async def _generate_hypothetical_answer(self, sub_query: str) -> str:
        """Generate a hypothetical answer for a sub-query using HYDE."""
        try:
            if not self.llm_client:
                raise RuntimeError("Local LLM client is not initialized.")

            hyde_prompt = f"Given the question, write a short hypothetical answer that could be true. Be brief, use keywords and concise.\n\nQuestion: {sub_query}"

            messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": hyde_prompt}
            ]

            response, _ = await self.llm_client.complete_chat(messages)
            return response.strip() if response else sub_query

        except Exception as e:
            self.logger.error(
                "Error generating hypothetical answer", error=str(e))
            return sub_query  # Fallback to original sub-query

    async def _retrieve_documents(self, hypothetical_answer: str) -> List[Dict[str, str]]:
        """Retrieve relevant documents using ClueWeb search with hypothetical answer and rerank them."""
        try:
            # Retrieve more documents initially for reranking
            # Retrieve more for better reranking
            initial_k = max(self.search_results_k * 2, 10)
            self.logger.info("Searching ClueWeb with hypothetical answer",
                             hypothetical_answer=hypothetical_answer,
                             initial_k=initial_k)
            search_results = await search_clueweb(query=hypothetical_answer, k=initial_k)
            search_results = [
                res for res in search_results if isinstance(res, SearchResult)]

            documents = []
            for result in search_results:
                content = result.text
                url = result.url

                if content:
                    documents.append({
                        "content": content[:self.max_context_length],
                        "url": url
                    })

            self.logger.info("Retrieved initial documents",
                             count=len(documents))

            # Rerank documents
            reranked_documents = await self._rerank_documents(hypothetical_answer, documents)

            return reranked_documents

        except Exception as e:
            self.logger.error("Error retrieving documents", error=str(e))
            return []

    async def _rerank_documents(self, query: str, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Rerank documents using either sentence transformer or logits-based reranking."""
        try:
            if not documents:
                return documents

            if self.reranker == "logits":
                return await self._rerank_documents_logits(query, documents)
            else:  # sentence_transformer
                return self._rerank_documents_sentence_transformer(query, documents)

        except Exception as e:
            self.logger.error("Error reranking documents", error=str(e))
            # Return original documents if reranking fails
            return documents

    def _rerank_documents_sentence_transformer(self, query: str, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Rerank documents using sentence transformer based on relevance to query."""
        try:
            if not documents:
                return documents

            self._ensure_rerank_model()

            # Prepare texts for embedding
            doc_texts = [doc["content"] for doc in documents]

            # Compute embeddings
            query_embedding = self.rerank_model_instance.encode(
                [query], convert_to_tensor=True)
            doc_embeddings = self.rerank_model_instance.encode(
                doc_texts, convert_to_tensor=True)

            # Compute cosine similarities
            similarities = np.dot(doc_embeddings.cpu().numpy(
            ), query_embedding.cpu().numpy().T).flatten()

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:self.rerank_top_k]

            # Return reranked documents
            reranked_docs = [documents[i] for i in top_k_indices]

            self.logger.info("Reranked documents with sentence transformer", original_count=len(
                documents), reranked_count=len(reranked_docs))
            return reranked_docs

        except Exception as e:
            self.logger.error(
                "Error in sentence transformer reranking", error=str(e))
            return documents

    async def _rerank_documents_logits(self, query: str, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Rerank documents using logits-based approach."""
        try:
            if not documents:
                return documents

            await self._ensure_llm_client()
            if not self.llm_client:
                raise RuntimeError(
                    "Ollama client is not initialized for logits reranking.")

            # Prepare the prompt template for logits reranking
            prompt_template = """<|system|>
You are a helpful assistant that determines if a document contains information that helps answer a given question. Answer only with 'Yes' or 'No'.
<|user|>
Document: {doc_text}

Question: {question}

Does this document contain information that helps answer this question (only answer 'Yes' or 'No')?
<|assistant|>
"""

            doc_scores = []
            for doc in documents:
                doc_text = doc["content"].replace("\n", " ")
                prompt = prompt_template.format(
                    doc_text=doc_text, question=query)

                messages = [
                    {"role": "user", "content": prompt}
                ]

                # Get completion and check if it starts with "Yes"
                response, _ = await self.llm_client.complete_chat(messages)
                response = response.strip()

                # Score based on whether response starts with "Yes"
                score = 1.0 if response.upper().startswith("YES") else 0.0
                doc_scores.append((doc, score))

            # Sort by score (descending) and take top-k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc,
                             score in doc_scores[:self.rerank_top_k]]

            self.logger.info("Reranked documents with logits", original_count=len(
                documents), reranked_count=len(reranked_docs))
            return reranked_docs

        except Exception as e:
            self.logger.error("Error in logits reranking", error=str(e))
            return documents

    def _format_context(self, documents: List[Dict[str, str]]) -> str:
        """Format retrieved documents into context for the prompt."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            url = doc.get("url", "")

            context_part = f"[{i}]\n{content}"
            if url:
                context_part += f"\nSource: {url}"
            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    async def _answer_sub_query(self, sub_query: str, context: str) -> str:
        """Generate an answer for a single sub-query using the provided context."""
        try:
            if not self.llm_client:
                raise RuntimeError("SGLang client is not initialized.")
            system_message = (
                "You are a helpful AI assistant. Answer the question using only the provided context. "
                "Be concise but comprehensive. If the context doesn't contain relevant information, "
                "say so clearly. Cite sources using document numbers [1], [2], etc. when possible."
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

    async def _synthesize_answers(self, original_query: str, sub_queries: List[str], sub_answers: List[str]) -> str:
        """Synthesize individual sub-query answers into a comprehensive final answer."""
        try:
            if not self.llm_client:
                raise RuntimeError("SGLang client is not initialized.")
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

            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": "You are an expert at synthesizing information from multiple sources into coherent answers."},
                {"role": "user", "content": synthesis_prompt}
            ]

            final_answer, _ = await self.llm_client.complete_chat(messages)
            return final_answer.strip() if final_answer else "Answer unavailable"

        except Exception as e:
            self.logger.error("Error synthesizing answers", error=str(e))
            # Fallback: just concatenate the answers
            return "\n\n".join([f"Q{i+1}: {q}\nA: {a}" for i, (q, a) in enumerate(zip(sub_queries, sub_answers))])

    @property
    def name(self) -> str:
        return "grag"

    async def _process_sub_query(self, i: int, sub_queries: List[str], sub_query: str) -> tuple[int, str, str, List[Dict[str, str]]]:
        self.logger.info(
            f"Processing sub-query {i+1}/{len(sub_queries)}", sub_query=sub_query)

        # Generate hypothetical answer for HYDE
        hypothetical_answer = await self._generate_hypothetical_answer(sub_query)

        # Retrieve documents using the hypothetical answer
        documents = await self._retrieve_documents(hypothetical_answer)

        # Format context
        context = self._format_context(documents)

        # Generate answer for this sub-query
        answer = await self._answer_sub_query(sub_query, context)

        return i, sub_query, answer, documents

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """
        Process an evaluation request using GRAG with decomposition and HYDE.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with generated answer
        """
        self._is_processing = True
        try:
            await self._ensure_llm_client()
            if not self.llm_client:
                raise RuntimeError("SGLang client is not initialized.")

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

            # Extract citations and contexts
            citations = []
            contexts = []
            for doc in all_documents:
                if doc.get("url"):
                    citations.append(doc["url"])
                if doc.get("content") or doc.get("text"):
                    # Use content or text field from the document
                    doc_content = doc.get("content") or doc.get("text", "")
                    if doc_content:
                        contexts.append(doc_content)

            return EvaluateResponse(
                query_id=request.iid,
                citations=list(set(citations)),  # Remove duplicates
                contexts=contexts,  # Actual document contexts used
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

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """
        Process a streaming request using GRAG with decomposition and HYDE.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function for streaming responses
        """
        async def stream():
            self._is_processing = True
            try:
                yield RunStreamingResponse(
                    intermediate_steps="Initializing SGLang server...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                await self._ensure_llm_client()
                if not self.llm_client:
                    raise RuntimeError("SGLang server failed to launch")

                yield RunStreamingResponse(
                    intermediate_steps="Decomposing complex query into sub-questions...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                # Step 1: Decompose the query
                sub_queries = await self._decompose_query(request.question)

                yield RunStreamingResponse(
                    intermediate_steps=f"Query decomposed into {len(sub_queries)} sub-questions. Processing each sub-question with HYDE and reranking...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                # Step 2: Answer each sub-query in parallel
                sub_answers = []
                all_documents = []

                yield RunStreamingResponse(
                    intermediate_steps=f"Processing {len(sub_queries)} sub-questions with HYDE and reranking...\n\n",
                    is_intermediate=True,
                    complete=False
                )

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

                        yield RunStreamingResponse(
                            intermediate_steps=f"✓ Completed sub-question {completed_count}/{len(sub_queries)}: {sub_query}\n\n",
                            is_intermediate=True,
                            complete=False
                        )
                    except Exception as e:
                        self.logger.error(
                            "Error processing sub-query in streaming", error=str(e))
                        completed_count += 1
                        yield RunStreamingResponse(
                            intermediate_steps=f"✗ Error in sub-question ({completed_count}/{len(sub_queries)} total)\n\n",
                            is_intermediate=True,
                            complete=False
                        )

                # Fill any empty values with error messages (in case of exceptions)
                for i, answer in enumerate(sub_answers):
                    if not answer:
                        sub_answers[i] = f"Error processing sub-query {i+1}"

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
                unique_urls = set()
                for doc in all_documents:
                    if doc.get("url") and doc["url"] not in unique_urls:
                        unique_urls.add(doc["url"])
                        citations.append(CitationItem(
                            url=doc["url"],
                            icon_url=to_icon_url(doc["url"]),
                            date=None,
                            title=None,
                            sid=None,
                            text=None,
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
        """Simple test execution for GRAG."""
        print("Testing GRAG with decomposition and HYDE...")

        # Initialize GRAG
        rag = GRAG(
            model_id="Qwen/Qwen3-4B",
            api_key=None,
            temperature=0.0,
            max_tokens=4096,
            search_results_k=2,  # Fewer results per sub-query
            max_sub_queries=3,
            # Using sentence-transformers model for reranking
            rerank_model="sentence-transformers/all-MiniLM-L6-v2",
            rerank_top_k=2
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
