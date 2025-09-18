from typing import AsyncGenerator, Callable, Optional
from systems.rag_interface import EvaluateRequest, EvaluateResponse, RAGInterface, RunRequest, RunStreamingResponse
from tools.llm_servers.sglang_server import launch_server, terminate_server
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger


class VanillaRAG(RAGInterface):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        reasoning_parser: Optional[str] = "qwen3",
        mem_fraction_static: Optional[float] = 0.4,
        max_running_requests: Optional[int] = 4,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        launch_on_init: bool = False
    ):
        """
        Initialize VanillaRAG with SGLang server.

        Args:
            model_id: The model ID to use for SGLang server
            reasoning_parser: Parser for reasoning models
            mem_fraction_static: Memory fraction for static allocation
            max_running_requests: Maximum concurrent requests
            api_key: API key for the server (optional)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            launch_on_init: Whether to launch the server immediately on initialization
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.launch_on_init = launch_on_init

        self.logger = get_logger("vanilla_rag")
        self.server_process = None
        self.client = None
        self.api_base = None
        self._is_processing = False

        # Launch server on init if requested
        if self.launch_on_init:
            self._ensure_server_running()

    def _ensure_server_running(self):
        """Ensure SGLang server is running and client is initialized."""
        if self.server_process is None or self.client is None:
            self.logger.info("Starting SGLang server", model_id=self.model_id)
            self.server_process, self.api_base, port = launch_server(
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
                llm_name="vanilla_rag_sglang"
            )
            self.logger.info("SGLang server and client initialized", port=port)

    def _shutdown_server(self):
        """Shutdown the SGLang server."""
        if self.server_process:
            terminate_server(self.server_process)
            self.server_process = None
            self.client = None
            self.api_base = None
            self.logger.info("SGLang server terminated")

    @property
    def name(self) -> str:
        return "vanilla-rag"

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
        Process an evaluation request using SGLang server.

        Args:
            request: EvaluateRequest containing query and iid

        Returns:
            EvaluateResponse with generated answer
        """
        self._is_processing = True
        try:
            self._ensure_server_running()
            if not self.client:
                raise RuntimeError("SGLang client is not initialized.")

            # Create a simple RAG prompt
            system_message = (
                "You are a helpful AI assistant. Provide accurate and informative answers "
                "to user questions. If you don't know something, say so clearly."
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.query}
            ]

            # Generate response using SGLang
            generated_response, _ = self.client.complete_chat(messages)

            return EvaluateResponse(
                query_id=request.iid,
                citations=[],  # TODO: Add citation extraction logic
                generated_response=generated_response
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
        Process a streaming request using SGLang server.

        Args:
            request: RunRequest containing the question

        Returns:
            Async generator function for streaming responses
        """
        async def stream():
            self._is_processing = True
            try:
                # Ensure server is running
                yield RunStreamingResponse(
                    intermediate_steps="Initializing SGLang server...",
                    is_intermediate=True,
                    complete=False
                )

                self._ensure_server_running()
                if not self.client:
                    raise RuntimeError("SGLang server failed to launch")

                yield RunStreamingResponse(
                    intermediate_steps="Processing question with language model...\n\n",
                    is_intermediate=True,
                    complete=False
                )

                # Create a comprehensive RAG prompt
                system_message = (
                    "You are a knowledgeable AI assistant. Provide detailed, accurate answers "
                    "to user questions. Structure your response clearly and cite sources when possible."
                )

                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.question}
                ]

                # Generate streaming response using SGLang
                full_response = ""
                async for chunk in self.client.complete_chat_streaming(messages):
                    full_response += chunk
                    # Yield intermediate chunks as they come
                    yield RunStreamingResponse(
                        intermediate_steps=chunk,
                        is_intermediate=True,
                        complete=False
                    )

                # Final response
                yield RunStreamingResponse(
                    final_report=full_response,
                    citations=[],  # TODO: Add citation extraction logic
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
