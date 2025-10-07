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
from tools.classifiers.query_complexity import QueryComplexity
from tools.logging_utils import get_logger


class RAGRouter(RAGInterface):
    def __init__(self):
        self.rag_simple_query = VanillaRAG()
        self.rag_complex_query = DecompositionRAG()
        self.query_complexity_model = QueryComplexity()
        self.logger = get_logger('RAGRouter')

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        complexity = self.query_complexity_model.predict(request.query)
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
        complexity = self.query_complexity_model.predict(request.question)
        if complexity.is_simple:
            self.logger.info(
                f"Routing to VanillaRAG for query: {request.question}")
            return await self.rag_simple_query.run_streaming(request)
        else:
            self.logger.info(
                f"Routing to DecompositionRAG for query: {request.question}")
            return await self.rag_complex_query.run_streaming(request)
