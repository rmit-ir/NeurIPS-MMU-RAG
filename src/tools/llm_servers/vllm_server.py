import asyncio
import subprocess
from typing import Callable, Optional, Dict, Tuple, Any

from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.sglang_utils import wait_for_server
from tools.logging_utils import get_logger

logger = get_logger("vllm_server")


def terminate_process(process):
    """Terminate a process and its children."""
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def launch_server(model_id="Qwen/Qwen3-4B",
                        reasoning_parser: Optional[str] = "qwen3",
                        gpu_memory_utilization: Optional[float] = 0.6,
                        max_model_len: Optional[int] = 20000,
                        host: str = "0.0.0.0",
                        port: Optional[int] = None,
                        api_key: Optional[str] = None):
    """
    Launch the vLLM server as a subprocess asynchronously.
    Args:
        model_id (str): The model ID to use.
        reasoning_parser (Optional[str]): The reasoning parser to use (for returning reasoning_content field).
        gpu_memory_utilization (Optional[float]): GPU memory utilization fraction.
        max_model_len (Optional[int]): Maximum model length.
        host (str): Host to bind the server to.
        port (Optional[int]): Port to bind the server to. If None, will use a random available port.
        api_key (Optional[str]): API key for authentication.
    """
    # Find an available port if not specified
    if port is None:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        *(["--reasoning-parser", reasoning_parser] if reasoning_parser else []),
        *(["--gpu-memory-utilization", str(gpu_memory_utilization)]
          if gpu_memory_utilization else []),
        *(["--max-model-len", str(max_model_len)] if max_model_len else []),
        "--host", host,
        "--port", str(port),
    ]

    if api_key:
        command.extend(["--api-key", api_key])

    logger.info("Launching vLLM server", command=' '.join(command))

    # Run the server launch in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    server_process = await loop.run_in_executor(
        None,
        lambda: subprocess.Popen(command)
    )

    server_host = f"http://{host}:{port}"
    api_base = f"{server_host}/v1"

    # Use async server health check
    await wait_for_server(server_host, timeout=1800, api_key=api_key)
    logger.info("vLLM server is running", port=port)

    def terminate():
        terminate_process(server_process)
    return server_process, terminate, server_host, api_base, port


class VLLMServerManager:
    """
    A class to manage multiple vLLM server instances.

    Each instance can run a different model and maintains its own server process,
    configuration, and synchronization lock. This allows for concurrent usage
    of multiple LLMs without interference.

    Usage:
        llm_server = VLLMServerManager(model_id="Qwen/Qwen3-4B")

        # This is where the server is actually launched
        api_base, port, server_host = await llm_server.get_server()

        openai_client = await llm_server.get_openai_client()
        # Use openai_client for requests
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-4B",
                 reasoning_parser: Optional[str] = "qwen3",
                 gpu_memory_utilization: Optional[float] = 0.6,
                 max_model_len: Optional[int] = 20000,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 api_key: Optional[str] = None):
        """
        Initialize vLLM server manager for a specific model configuration.

        Args:
            model_id (str): The model ID to use.
            reasoning_parser (Optional[str]): The reasoning parser to use (for returning reasoning_content field).
            gpu_memory_utilization (Optional[float]): GPU memory utilization fraction.
            max_model_len (Optional[int]): Maximum model length.
            host (str): Host to bind the server to.
            port (Optional[int]): Port to bind the server to.
            api_key (Optional[str]): API key for authentication.
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.host = host
        self.port = port
        self.api_key = api_key

        # Instance-specific server state
        self._server_process = None
        self._server_terminate_fn: Optional[Callable[[], None]] = None
        self._server_host = None
        self._api_base = None
        self._port = None

        # Instance-specific async lock for proper synchronization
        self._server_lock = asyncio.Lock()

        # Logger with model-specific context
        self._logger = get_logger(
            f"vllm_server_{model_id.replace('/', '_')}")

    async def get_server(self) -> Tuple[str, int, str]:
        """
        Get or create a vLLM server instance with proper async synchronization.

        This method ensures that only one server is launched at a time for this instance
        and that multiple concurrent requests will wait for the same server instance.

        With asyncio.Lock(), requests don't need to sleep and poll - they wait
        efficiently until the lock is available. The first request will launch
        the server, and subsequent requests will immediately get the existing server.

        Wait time per request:
        - First request: Time to launch server + health check (typically 30-60 seconds)
        - Concurrent requests: Minimal wait (microseconds) until lock is released
        - Subsequent requests: Immediate return with existing server info

        Returns:
            Tuple containing (api_base, port, server_host)
        """
        # Use async lock to prevent race conditions - no polling needed!
        async with self._server_lock:
            # Check if server is already running
            if (self._server_process and self._server_host and
                    self._api_base and self._port):
                self._logger.info("Using existing vLLM server",
                                  port=self._port, model_id=self.model_id)
                return self._api_base, self._port, self._server_host

            # Launch new server
            self._logger.info("Launching new vLLM server",
                              model_id=self.model_id)
            (self._server_process, self._server_terminate_fn, self._server_host,
             self._api_base, self._port) = await launch_server(
                model_id=self.model_id,
                reasoning_parser=self.reasoning_parser,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                host=self.host,
                port=self.port,
                api_key=self.api_key
            )

            if not (self._server_process and self._server_host and
                    self._api_base and self._port):
                raise RuntimeError(
                    f"Failed to launch vLLM server for model {self.model_id}")

            self._logger.info("vLLM server ready",
                              port=self._port, model_id=self.model_id)
            return self._api_base, self._port, self._server_host

    async def _terminate_server(self):
        """
        Terminate the running vLLM server instance for this manager.
        """
        async with self._server_lock:
            if self._server_process is not None:
                self._logger.info("Terminating vLLM server",
                                  port=self._port, model_id=self.model_id)
                if self._server_terminate_fn:
                    self._server_terminate_fn()

                # Reset instance variables
                self._server_process = None
                self._server_host = None
                self._api_base = None
                self._port = None

                self._logger.info("vLLM server terminated",
                                  model_id=self.model_id)
            else:
                self._logger.info("No vLLM server running to terminate",
                                  model_id=self.model_id)

    async def get_openai_client(self,
                                max_tokens: int = 4096,
                                temperature: float = 0.0) -> GeneralOpenAIClient:
        """
        Get an OpenAI client connected to this vLLM server instance.

        Args:
            max_tokens (int): Maximum tokens for responses.
            temperature (float): Temperature for generation.

        Returns:
            GeneralOpenAIClient: Configured client for this server instance.
        """
        api_base, port, server_host = await self.get_server()

        return GeneralOpenAIClient(
            api_base=api_base,
            api_key=self.api_key,
            model_id=self.model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return (self._server_process is not None and
                self._server_host is not None and
                self._api_base is not None and
                self._port is not None)

    @property
    def server_info(self) -> Dict[str, Any]:
        """Get current server information."""
        return {
            "model_id": self.model_id,
            "is_running": self.is_running,
            "port": self._port,
            "api_base": self._api_base,
            "server_host": self._server_host,
            "reasoning_parser": self.reasoning_parser,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "host": self.host
        }


ALL_VLLM_SERVERS: Dict[str, VLLMServerManager] = {}


def get_llm_mgr(model_id="Qwen/Qwen3-4B",
                reasoning_parser: Optional[str] = "qwen3",
                gpu_memory_utilization: Optional[float] = 0.6,
                max_model_len: Optional[int] = 20000,
                host: str = "0.0.0.0",
                port: Optional[int] = None,
                api_key: Optional[str] = None):
    if model_id not in ALL_VLLM_SERVERS:
        ALL_VLLM_SERVERS[model_id] = VLLMServerManager(
            model_id=model_id,
            reasoning_parser=reasoning_parser,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            host=host,
            port=port,
            api_key=api_key
        )
    return ALL_VLLM_SERVERS[model_id]


async def main():
    """
    Test the async vLLM server implementation.
    """
    model_id = "Qwen/Qwen3-4B"
    api_key = "abc"

    try:
        # Get server instance using the new async method
        llm_server = get_llm_mgr(
            model_id=model_id,
            api_key=api_key
        )
        api_base, port, server_host = await llm_server.get_server()
        logger.info("vLLM server is running", port=port, model_id=model_id)

        # Create OpenAI client
        openai_client = GeneralOpenAIClient(
            api_base=api_base,
            api_key=api_key,
            model_id=model_id,
            temperature=0
        )

        # Test the server
        content, _ = openai_client.complete_chat([
            {"role": "user", "content": "I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building."}
        ])

        logger.info("Response from vLLM server", response=content)

    finally:
        # Clean up server
        await llm_server._terminate_server()


if __name__ == "__main__":
    asyncio.run(main())
