from typing import AsyncGenerator, Callable
from systems.decomposition_rag.decomposition_rag import DecompositionRAG
from systems.rag_interface import (
    EvaluateRequest,
    EvaluateResponse,
    RAGInterface,
    RunRequest,
    RunStreamingResponse,
)
from systems.vanilla_agent.vanilla_rag import VanillaRAG
from tools.classifiers.llm_query_complexity import QueryComplexityLLM
from tools.logging_utils import get_logger


class RAGRouterLLM(RAGInterface):
    def __init__(self):
        self.rag_simple_query = VanillaRAG()
        self.rag_complex_query = DecompositionRAG()
        self.query_complexity_model = QueryComplexityLLM()
        self.logger = get_logger('RAGRouterLLM')

    @property
    def name(self) -> str:
        return "rag-router"

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        complexity = await self.query_complexity_model.predict(request.query)
        if complexity.is_simple:
            self.logger.info(
                f"Routing to VanillaRAG for query: {request.query}")
            return await self.rag_simple_query.evaluate(request)
        else:
            self.logger.info(
                f"Routing to DecompositionRAG for query: {request.query}")
            return await self.rag_complex_query.evaluate(request)

    async def run_streaming(
        self, request: RunRequest
    ) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:

        async def route_and_stream():
            # Predict complexity inside the returned callable
            complexity = await self.query_complexity_model.predict(request.question)

            if complexity.is_simple:
                self.logger.info(
                    f"Routing to VanillaRAG for query: {request.question}")
                stream_func = await self.rag_simple_query.run_streaming(request)
            else:
                self.logger.info(
                    f"Routing to DecompositionRAG for query: {request.question}")
                stream_func = await self.rag_complex_query.run_streaming(request)

            # Execute the selected stream and yield all responses
            async for response in stream_func():
                yield response

        return route_and_stream
