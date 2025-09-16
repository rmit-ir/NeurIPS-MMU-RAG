from typing import AsyncGenerator, Callable
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse


class VanillaRAG(RAGInterface):
    @property
    def name(self) -> str:
        return "vanilla-rag"

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        return EvaluateResponse(
            query_id=request.iid,
            citations=[],
            generated_response=f"Answer to question {request.query} is: 42"
        )

    async def run_streaming(self, request: RunRequest) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        # TODO: here do somethings right away if needed
        async def stream():
            # only start yielding after demanded stream
            yield RunStreamingResponse(
                intermediate_steps="step 1...",
                is_intermediate=True,
                complete=False
            )
            yield RunStreamingResponse(
                intermediate_steps="step 2...",
                is_intermediate=True,
                complete=False
            )
            # Only the last event can say complete=True
            yield RunStreamingResponse(
                final_report="final answer",
                citations=["doc_id_1", "doc_id_2"],
                is_intermediate=False,
                complete=True
            )
        return stream
