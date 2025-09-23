"""
NOTE: NOT USED YET
Abstract interface for local LLM server implementations.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any


class ServerInterface(ABC):
    """
    Abstract interface for local LLM server implementations.

    This interface defines the contract that all server implementations
    (SGLang, vLLM, etc.) must follow.
    """

    @abstractmethod
    async def launch_server(self) -> Tuple[Any, Callable[[], None], str, str, int]:
        """
        Launch the server as a subprocess asynchronously.

        Returns:
            Tuple containing:
            - server_process: The server process object
            - terminate_fn: Function to terminate the server
            - server_host: The server host URL (e.g., "http://localhost:8000")
            - api_base: The API base URL (e.g., "http://localhost:8000/v1")
            - port: The port number
        """
        pass

    @abstractmethod
    async def wait_for_server_ready(self, server_host: str, timeout: int = 1800) -> None:
        """
        Wait for the server to be ready by performing health checks.

        Args:
            server_host: The server host URL
            timeout: Maximum time to wait in seconds
        """
        pass


class SGLangServerImpl(ServerInterface):
    """SGLang server implementation."""

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-4B",
                 reasoning_parser: Optional[str] = "qwen3",
                 mem_fraction_static: Optional[float] = 0.4,
                 max_running_requests: Optional[int] = 4,
                 api_key: Optional[str] = None):
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.api_key = api_key

    async def launch_server(self) -> Tuple[Any, Callable[[], None], str, str, int]:
        """Launch SGLang server."""
        from sglang.test.doc_patch import launch_server_cmd
        from sglang.utils import terminate_process
        from tools.logging_utils import get_logger

        logger = get_logger("sglang_server")

        command = [
            "python", "-m", "sglang.launch_server",
            "--model", self.model_id,
            *(["--reasoning-parser", self.reasoning_parser]
              if self.reasoning_parser else []),
            "--disable-radix-cache",
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--max-running-requests", str(self.max_running_requests),
            "--host", "0.0.0.0",
            *(["--api-key", self.api_key] if self.api_key else []),
        ]
        logger.info("Launching SGLang server", command=' '.join(command))

        # Run the server launch in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        server_process, port = await loop.run_in_executor(
            None,
            lambda: launch_server_cmd(' '.join(command))
        )

        server_host = f"http://localhost:{port}"
        api_base = f"{server_host}/v1"

        def terminate():
            terminate_process(server_process)

        return server_process, terminate, server_host, api_base, port

    async def wait_for_server_ready(self, server_host: str, timeout: int = 1800) -> None:
        """Wait for SGLang server to be ready."""
        from tools.llm_servers.sglang_utils import wait_for_server
        await wait_for_server(server_host, timeout=timeout, api_key=self.api_key)


class VLLMServerImpl(ServerInterface):
    """vLLM server implementation."""

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-4B",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 port: int = 18020,
                 api_key: Optional[str] = None):
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.port = port
        self.api_key = api_key

    async def launch_server(self) -> Tuple[Any, Callable[[], None], str, str, int]:
        """Launch vLLM server."""
        import subprocess
        import signal
        from tools.logging_utils import get_logger

        logger = get_logger("vllm_server")

        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_id,
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--host", "0.0.0.0",
            "--port", str(self.port),
            *(["--max-model-len", str(self.max_model_len)]
              if self.max_model_len else []),
            *(["--api-key", self.api_key] if self.api_key else []),
        ]
        logger.info("Launching vLLM server", command=' '.join(command))

        # Launch the server process
        server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=None if hasattr(signal, 'SIGKILL') else None
        )

        server_host = f"http://localhost:{self.port}"
        api_base = f"{server_host}/v1"

        def terminate():
            if server_process.poll() is None:  # Process is still running
                server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_process.kill()
                    server_process.wait()

        return server_process, terminate, server_host, api_base, self.port

    async def wait_for_server_ready(self, server_host: str, timeout: int = 1800) -> None:
        """Wait for vLLM server to be ready."""
        from tools.llm_servers.sglang_utils import wait_for_server
        await wait_for_server(server_host, timeout=timeout, api_key=self.api_key)
