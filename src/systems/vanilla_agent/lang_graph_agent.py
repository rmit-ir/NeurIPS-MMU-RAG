import asyncio
from typing import AsyncGenerator, Callable, List, Literal, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from openai.types.chat import ChatCompletionMessageParam

# Import existing components
from systems.rag_interface import EvaluateRequest, RAGInterface, RunRequest, RunStreamingResponse
from systems.vanilla_agent.agent_tools import MSG_TYPE, pub_msg, to_any
from systems.vanilla_agent.rag_util_fn import build_llm_messages, get_default_llms, inter_resp
from tools.logging_utils import get_logger
from tools.web_search import SearchResult, search_clueweb
from tools.docs_utils import truncate_docs, update_docs_sids


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class CustomState(MessagesState):
    turns_left: int
    total_turns: int
    accumulated_context: List[SearchResult]
    """Each turn we add more documents to this list, """


class LangGraphAgent(RAGInterface):
    """
    Agentic RAG system using LangGraph for retrieval decisions.

    This agent uses a state graph to decide when to retrieve documents,
    how to rewrite queries, and when to generate final answers.
    """

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
        k_docs: int = 20,
        cw22_a: bool = True,
    ):
        """Initialize LangGraphAgent with agentic capabilities using Qwen3 4B."""
        self.max_tokens = max_tokens
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.retrieval_words_threshold = retrieval_words_threshold
        self.enable_think = enable_think
        self.k_docs = k_docs
        self.cw22_a = cw22_a

        self.logger = get_logger("langgraph_agent")

        # Initialize LLM client directly with Qwen3 4B
        self.llm_client = None
        self.reranker = None

        # Build the agentic workflow
        self.workflow = None
        self.graph = None

    @property
    def name(self) -> str:
        return "langgraph-agent"

    async def _create_retriever_tool(self):
        """Create a retriever tool that uses the agent's search capabilities."""
        @tool
        async def retrieve_documents(query: str) -> str:
            """Search and return relevant documents for the given query."""
            llm, reranker = await get_default_llms()
            try:
                self.logger.info(f"Search: {query}")
                # Use existing search capabilities
                docs = await search_clueweb(query, k=self.k_docs, cw22_a=self.cw22_a)
                docs = [r for r in docs if isinstance(r, SearchResult)]
                docs = await reranker.rerank(query, docs)
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

        async def retrieve_step(state: CustomState):
            """Retrieve documents for the query."""
            self.logger.info("[retrieve_step] START")
            retriever_tool = await self._create_retriever_tool()
            query = state["messages"][0].content
            pub_msg("custom_intermediate_step",
                    inter_resp(f"Searching: {query}\n\n", silent=False, logger=self.logger))
            context = await retriever_tool.ainvoke({"query": query})
            self.logger.info("[retrieve_step] END")
            return {**state, "messages": state["messages"] + [HumanMessage(content=context)]}

        async def grade_context(state: CustomState) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether retrieved documents are relevant using Qwen3 4B."""
            self.logger.info("[grade_documents] START")
            llm, reranker = await get_default_llms()
            GRADE_PROMPT = """You are an expert in the field of answering questions like "{question}".

Review the search results and see if they are sufficient for answering the user question. Please consider:

1. Does the user want a simple answer or a comprehensive explanation?
2. Does the search results fully addresses the user's query and any subcomponents?
3. When you answer 'yes', we will proceed to generate the final answer based on these results. If you answer 'no', we will continue the next turn of using a different query to search, and let you review again.
4. If information is missing or uncertain, always return 'no' for clarification, and generate a new query enclosed in xml markup <new-query>your query</new-query> indicating the clarification needed.
5. Have you ran out of turns? If there is no turn left, you must say 'yes'.

Response format:

- You must give a binary score 'yes' or 'no' to indicate whether the search results are sufficient to the question.
- Only respond with 'yes' or 'no'.

Turn left: {turn_left} / {turn_total}.

Here is the user question: {question}

Here is the search results:

{context}
"""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            turn_left: int = to_any(state["turns_left"])
            turn_total: int = to_any(state["total_turns"])
            pub_msg("custom_intermediate_step",
                    inter_resp(f"Grading retrieved documents, turns left {turn_left} / {turn_total}\n\n", silent=False, logger=self.logger))

            prompt = GRADE_PROMPT.format(
                question=question, context=context, turn_left=turn_left, turn_total=turn_total)
            messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            score = ""
            async for chunk in llm.complete_chat_streaming(messages):
                if chunk.choices[0].finish_reason is not None:
                    # Stream finished
                    break
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    # still intermediate steps
                    pub_msg("custom_intermediate_step",
                            inter_resp(delta.reasoning_content, silent=True, logger=self.logger))
                elif hasattr(delta, 'content') and delta.content:
                    # final report
                    score += delta.content

            score = score.strip().lower() if score else "no"
            pub_msg("custom_intermediate_step", RunStreamingResponse(
                final_report=f"Search results are sufficient: {score}",
                is_intermediate=False,
                complete=False
            ))

            self.logger.info("[grade_documents] END")

            return "generate_answer" if score == "yes" else "rewrite_question"

        async def rewrite_question(state: CustomState):
            """Rewrite the original user question for better retrieval"""
            self.logger.info("[rewrite_question] START")
            llm, reranker = await get_default_llms()
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

            response, cpl = await llm.complete_chat(llm_messages)
            reasoning_content = cpl.choices[0].message.reasoning_content.strip() \
                if 'reasoning_content' in cpl.choices[0].message else ''
            self.logger.info(
                "rewrite_question", original_question=question,
                rewritten_question=response or "", reasoning=reasoning_content)
            self.logger.info("[rewrite_question] END")
            return {
                **state,
                "messages": [HumanMessage(content=response or question)],
                "turns_left": max(0, state["turns_left"] - 1)
            }

        async def generate_answer(state: CustomState):
            """Generate final answer based on retrieved context using Qwen3 4B."""
            self.logger.info("[generate_answer] START")
            final_content = ""
            question: str = to_any(state["messages"][0].content)
            context: str = to_any(state["messages"][-1].content)
            llm_client, reranker = await get_default_llms()
            messages = build_llm_messages(context, question, self.enable_think)
            pub_msg("custom_intermediate_step",
                    inter_resp("Starting final answer\n\n", silent=False, logger=self.logger))
            async for chunk in llm_client.complete_chat_streaming(messages):
                if chunk.choices[0].finish_reason is not None:
                    # Stream finished
                    break
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    # still intermediate steps
                    pub_msg("custom_intermediate_step",
                            inter_resp(delta.reasoning_content, silent=True, logger=self.logger))
                elif hasattr(delta, 'content') and delta.content:
                    # final report
                    final_content += delta.content
                    pub_msg("custom_final_answer", RunStreamingResponse(
                        final_report=delta.content,
                        is_intermediate=False,
                        complete=False
                    ))
                # otherwise ignore empty deltas

            citations = [
                # CitationItem(
                #     url=r.url,
                #     icon_url=to_icon_url(r.url),
                #     date=str(r.date) if r.date else None,
                #     sid=r.sid,
                #     title=None,
                #     text=r.text
                # )
                # for r in docs if isinstance(r, SearchResult)
            ]
            # Final response
            pub_msg("custom_final_answer", RunStreamingResponse(
                citations=citations,
                is_intermediate=False,
                complete=False
            ))
            self.logger.info("[generate_answer] END")
            return {**state, "messages": [AIMessage(content=final_content)]}

        # Build the workflow
        workflow = StateGraph(CustomState)

        # Add nodes
        workflow.add_node("retrieve", retrieve_step)
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("generate_answer", generate_answer)

        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges("retrieve", grade_context)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "retrieve")

        return workflow

    def _inter_resp(self, desc: str, silent: bool = False) -> RunStreamingResponse:
        if not silent:
            self.logger.info(f"Intermediate step | {desc}")
        return RunStreamingResponse(
            intermediate_steps=desc,
            is_intermediate=True,
            complete=False
        )

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        """Process a streaming request using the agentic workflow."""
        async def stream():
            try:
                self.logger.info(
                    f"Processing agentic request: {request.question}")

                yield inter_resp("LangGraph Agent starting agentic workflow...\n\n", silent=False, logger=self.logger)

                if not self.graph:
                    yield inter_resp("Building agentic workflow graph...\n\n", silent=False, logger=self.logger)
                    self.workflow = await self._build_workflow()
                    self.graph = self.workflow.compile()

                yield inter_resp("Running agentic workflow...\n\n", silent=False, logger=self.logger)

                # Stream through the workflow
                final_response = ""
                async for (_msg_type, _chunk) in self.graph.astream({
                    "messages": [HumanMessage(content=request.question)],
                    "total_turns": 3,
                    "turns_left": 3,
                    "accumulated_context": []
                }, stream_mode="custom"):
                    # TODO: msg_type might be useful
                    msg_type: MSG_TYPE = to_any(_msg_type)
                    chunk: RunStreamingResponse = to_any(_chunk)
                    yield chunk

                # Final response
                yield RunStreamingResponse(
                    final_report=final_response,
                    is_intermediate=False,
                    complete=True
                )

            except Exception as e:
                self.logger.exception("Error in agentic streaming")
                yield RunStreamingResponse(
                    final_report=f"Agent error: {str(e)}",
                    is_intermediate=False,
                    complete=True,
                    error=str(e)
                )

        return stream


# Test and example usage
if __name__ == "__main__":
    async def main():
        """Simple test execution for LangGraphAgent."""
        print("Testing LangGraphAgent with Qwen3 4B LLM...")

        # Initialize LangGraphAgent
        agent = LangGraphAgent(max_tokens=4096)
        question = "I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building."

        try:
            async def test_eval():
                # Test evaluation method
                print("\n=== Testing LangGraphAgent Agentic Evaluation ===")
                eval_request = EvaluateRequest(
                    query=question, iid="agent-test-001")

                eval_response = await agent.evaluate(eval_request)
                print(f"Query ID: {eval_response.query_id}")
                print(f"Response: {eval_response.generated_response}")

            async def test_streaming():
                # Test streaming method
                print("\n=== Testing LangGraphAgent Agentic Streaming ===")
                print(f"Question: {question}")
                run_request = RunRequest(question=question)

                stream_func = await agent.run_streaming(run_request)
                print("Agentic streaming response:")

                current_step = ''

                async for response in stream_func():
                    if response.error:
                        print(f"[ERROR] {response.error}")
                    elif response.complete:
                        print("\n[STREAM COMPLETE]")
                    elif response.is_intermediate and response.intermediate_steps:
                        if current_step != 'intermediate':
                            print("\n"+"-"*30 + " Intermediate Steps " + "-"*30)
                            current_step = 'intermediate'
                        print(response.intermediate_steps, end='', flush=True)
                    elif response.final_report:
                        if current_step != 'final':
                            print("\n"+"="*30 + " Final Answer " + "="*30)
                            current_step = 'final'
                        print(response.final_report, end='', flush=True)

            # await test_eval()
            await test_streaming()
        except Exception as e:
            print(f"Error during agentic testing: {str(e)}")

    # Run the async main function
    asyncio.run(main())
