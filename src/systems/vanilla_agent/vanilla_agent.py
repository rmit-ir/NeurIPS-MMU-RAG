import asyncio
from typing import AsyncGenerator, Callable, List, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from openai.types.chat import ChatCompletionMessageParam

# Import existing components
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse
from tools.logging_utils import get_logger
from tools.web_search import SearchResult, search_clueweb
from tools.reranker_vllm import get_reranker
from tools.docs_utils import truncate_docs, update_docs_sids
from tools.llm_servers.general_openai_client import GeneralOpenAIClient


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class VanillaAgent(RAGInterface):
    """
    Agentic RAG system using LangGraph for retrieval decisions.

    This agent uses a state graph to decide when to retrieve documents,
    how to rewrite queries, and when to generate final answers.
    """

    def __init__(
        self,
        api_host: str = "http://localhost:8088",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retrieval_words_threshold: int = 5000,
        enable_think: bool = True,
        k_docs: int = 30,
        cw22_a: bool = True,
    ):
        """Initialize VanillaAgent with agentic capabilities using Qwen3 4B."""
        self.api_host = api_host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_words_threshold = retrieval_words_threshold
        self.enable_think = enable_think
        self.k_docs = k_docs
        self.cw22_a = cw22_a

        self.logger = get_logger("vanilla_agent")

        # Initialize LLM client directly with Qwen3 4B
        self.llm_client = GeneralOpenAIClient(api_base=self.api_host)
        self.reranker = None

        # Build the agentic workflow
        self.workflow = None
        self.graph = None

    @property
    def name(self) -> str:
        return "vanilla-agent"

    async def _ensure_reranker(self):
        """Ensure reranker is initialized."""
        if not self.reranker:
            self.reranker = await get_reranker()

    async def _create_retriever_tool(self):
        """Create a retriever tool that uses the agent's search capabilities."""
        @tool
        async def retrieve_documents(query: str) -> str:
            """Search and return relevant documents for the given query."""
            await self._ensure_reranker()
            if not self.reranker:
                raise ValueError("Reranker is not initialized.")
            try:
                # Use existing search capabilities
                docs = await search_clueweb(query, k=self.k_docs, cw22_a=self.cw22_a)
                docs = [r for r in docs if isinstance(r, SearchResult)]
                docs = await self.reranker.rerank(query, docs)
                docs = truncate_docs(docs, self.retrieval_words_threshold)
                docs = update_docs_sids(docs)

                # Format results for the agent
                if not docs:
                    return "No relevant documents found."

                context = "\n\n".join([
                    f"Webpage ID=[{r.sid}] URL=[{r.url}] Date=[{r.date}]:\n{r.text}"
                    for r in docs
                ])
                return context
            except Exception as e:
                self.logger.error(f"Error in retrieve_documents: {e}")
                return f"Error retrieving documents: {str(e)}"

        return retrieve_documents

    async def _build_workflow(self):
        """Build the agentic workflow using LangGraph with Qwen3 4B."""

        async def retrieve_step(state: MessagesState):
            """Retrieve documents for the query."""
            retriever_tool = await self._create_retriever_tool()
            query = state["messages"][0].content
            context = await retriever_tool.ainvoke({"query": query})
            return {"messages": state["messages"] + [HumanMessage(content=context)]}

        async def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether retrieved documents are relevant using Qwen3 4B."""
            GRADE_PROMPT = (
                "You are a grader assessing relevance of a retrieved document to a user question. "
                "Here is the retrieved document:\n\n{context}\n\n"
                "Here is the user question: {question}\n"
                "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. "
                "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question. "
                "Only respond with 'yes' or 'no'."
            )

            question = state["messages"][0].content
            context = state["messages"][-1].content

            prompt = GRADE_PROMPT.format(question=question, context=context)
            messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            response, cpl = await self.llm_client.complete_chat(messages)
            reasoning_content = cpl.choices[0].message.reasoning_content.strip() \
                if 'reasoning_content' in cpl.choices[0].message else ''
            score = response.strip().lower() if response else "no"
            self.logger.info(
                "grade_documents", question=question, score=score, reasoning=reasoning_content)

            return "generate_answer" if score == "yes" else "rewrite_question"

        async def rewrite_question(state: MessagesState):
            """Rewrite the original user question for better retrieval"""
            REWRITE_PROMPT = (
                "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
                "Here is the initial question:\n"
                "-------\n"
                "{question}\n"
                "-------\n"
                "Formulate an improved question that would be better for search:"
            )

            messages = state["messages"]
            question = messages[0].content
            prompt = REWRITE_PROMPT.format(question=question)

            llm_messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            response, cpl = await self.llm_client.complete_chat(llm_messages)
            reasoning_content = cpl.choices[0].message.reasoning_content.strip() \
                if 'reasoning_content' in cpl.choices[0].message else ''
            self.logger.info(
                "rewrite_question", original_question=question,
                rewritten_question=response or "", reasoning=reasoning_content)
            return {"messages": [HumanMessage(content=response or question)]}

        async def generate_answer(state: MessagesState):
            """Generate final answer based on retrieved context using Qwen3 4B."""
            GENERATE_PROMPT = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n"
                "Question: {question}\n"
                "Context: {context}"
            )

            question = state["messages"][0].content
            context = state["messages"][-1].content
            prompt = GENERATE_PROMPT.format(question=question, context=context)

            llm_messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            response, _ = await self.llm_client.complete_chat(llm_messages)
            return {"messages": [AIMessage(content=response or "I don't know.")]}

        # Build the workflow
        workflow = StateGraph(MessagesState)

        # Add nodes
        workflow.add_node("retrieve", retrieve_step)
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("generate_answer", generate_answer)

        # Add edges
        workflow.add_edge(START, "retrieve")

        # Conditional edges from retrieve
        workflow.add_conditional_edges(
            "retrieve",
            grade_documents,
        )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "retrieve")

        return workflow

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """Process an evaluation request using the agentic workflow."""
        try:
            await self._ensure_reranker()

            if not self.graph:
                self.workflow = await self._build_workflow()
                self.graph = self.workflow.compile()

            # Run the agentic workflow
            final_state = await self.graph.ainvoke({
                "messages": [HumanMessage(content=request.query)]
            })

            # Extract the final response
            final_message = final_state["messages"][-1]
            if hasattr(final_message, 'content'):
                generated_response = final_message.content
            else:
                generated_response = str(final_message)

            return EvaluateResponse(
                query_id=request.iid,
                citations=[],  # TODO: Extract citations from workflow
                contexts=[],   # TODO: Extract contexts from workflow
                generated_response=generated_response or "No response generated."
            )

        except Exception as e:
            self.logger.error("Error in agentic evaluation",
                              query_id=request.iid, error=str(e))
            return EvaluateResponse(
                query_id=request.iid,
                citations=[],
                contexts=[],
                generated_response=f"Agent error: {str(e)}"
            )

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """Process a streaming request using the agentic workflow."""
        async def stream():
            try:
                self.logger.info(
                    f"Processing agentic request: {request.question}")

                yield RunStreamingResponse(
                    intermediate_steps="Vanilla Agent starting agentic workflow...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                await self._ensure_reranker()

                if not self.graph:
                    yield RunStreamingResponse(
                        intermediate_steps="Building agentic workflow graph...\n\n",
                        is_intermediate=True,
                        complete=False
                    )
                    self.workflow = await self._build_workflow()
                    self.graph = self.workflow.compile()

                yield RunStreamingResponse(
                    intermediate_steps="Running agentic decision-making process...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                # Stream through the workflow
                final_response = ""
                async for chunk in self.graph.astream({
                    "messages": [HumanMessage(content=request.question)]
                }):
                    for node, update in chunk.items():
                        yield RunStreamingResponse(
                            intermediate_steps=f"Agent node '{node}' processing...\n",
                            is_intermediate=True,
                            complete=False
                        )

                        if "messages" in update and update["messages"]:
                            last_message = update["messages"][-1]
                            if hasattr(last_message, 'content') and last_message.content:
                                final_response = last_message.content

                # Final response
                yield RunStreamingResponse(
                    final_report=final_response,
                    citations=[],  # TODO: Extract citations from workflow
                    is_intermediate=False,
                    complete=True
                )

            except Exception as e:
                self.logger.exception("Error in agentic streaming")
                yield RunStreamingResponse(
                    final_report=f"Agent error: {str(e)}",
                    citations=[],
                    is_intermediate=False,
                    complete=True,
                    error=str(e)
                )

        return stream


# Test and example usage
if __name__ == "__main__":
    async def main():
        """Simple test execution for VanillaAgent."""
        print("Testing VanillaAgent with Qwen3 4B LLM...")

        # Initialize VanillaAgent
        agent = VanillaAgent(
            api_host="http://localhost:8088",
            temperature=0.0,
            max_tokens=4096
        )

        try:
            # Test evaluation method
            print("\n=== Testing VanillaAgent Agentic Evaluation ===")
            eval_request = EvaluateRequest(
                query="I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building.",
                iid="agent-test-001"
            )

            eval_response = await agent.evaluate(eval_request)
            print(f"Query ID: {eval_response.query_id}")
            print(f"Response: {eval_response.generated_response}")

            # Test streaming method
            print("\n=== Testing VanillaAgent Agentic Streaming ===")
            run_request = RunRequest(
                question="Explain quantum computing and its potential applications."
            )

            stream_func = await agent.run_streaming(run_request)
            print("Agentic streaming response:")

            async for response in stream_func():
                if response.is_intermediate and response.intermediate_steps:
                    print(f"[AGENT] {response.intermediate_steps.strip()}")
                elif response.final_report:
                    print(f"[FINAL] {response.final_report}")
                elif response.error:
                    print(f"[ERROR] {response.error}")

        except Exception as e:
            print(f"Error during agentic testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
