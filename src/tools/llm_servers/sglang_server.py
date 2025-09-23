import asyncio
from typing import Optional, Dict, Tuple, Any

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import terminate_process
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.sglang_utils import wait_for_server
from tools.logging_utils import get_logger

logger = get_logger("sglang_server")


async def launch_server(model_id="Qwen/Qwen3-4B",
                        reasoning_parser: Optional[str] = "qwen3",
                        mem_fraction_static: Optional[float] = 0.4,
                        max_running_requests: Optional[int] = 4,
                        api_key: Optional[str] = None):
    """
    Launch the SGLang server as a subprocess asynchronously.
    Args:
        model_id (str): The model ID to use.
        reasoning_parser (Optional[str]): The reasoning parser to use.
        mem_fraction_static (float): Fraction of memory to allocate statically.
        max_running_requests (int): Maximum number of concurrent running requests.
        api_key (Optional[str]): API key for authentication.
    """
    command = [
        "python", "-m", "sglang.launch_server",
        "--model", model_id,
        *(["--reasoning-parser", reasoning_parser] if reasoning_parser else []),
        "--disable-radix-cache",
        "--mem-fraction-static", str(mem_fraction_static),
        "--max-running-requests", str(max_running_requests),
        "--host", "0.0.0.0",
        *(["--api-key", api_key] if api_key else []),
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

    # Use async server health check
    await wait_for_server(server_host, timeout=1800, api_key=api_key)
    logger.info("SGLang server is running", port=port)
    return server_process, server_host, api_base, port


class SGLangServerManager:
    """
    A class to manage multiple SGLang server instances.

    Each instance can run a different model and maintains its own server process,
    configuration, and synchronization lock. This allows for concurrent usage
    of multiple LLMs without interference.

    Usage:
        llm_server = SGLangServerManager(model_id="Qwen/Qwen3-4B")

        # This is where the server is actually launched
        api_base, port, server_host = await llm_server.get_server()

        openai_client = await llm_server.get_openai_client()
        # Use openai_client for requests
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-4B",
                 reasoning_parser: Optional[str] = "qwen3",
                 mem_fraction_static: Optional[float] = 0.4,
                 max_running_requests: Optional[int] = 4,
                 api_key: Optional[str] = None):
        """
        Initialize SGLang server manager for a specific model configuration.

        Args:
            model_id (str): The model ID to use.
            reasoning_parser (Optional[str]): The reasoning parser to use.
            mem_fraction_static (float): Fraction of memory to allocate statically.
            max_running_requests (int): Maximum number of concurrent running requests.
            api_key (Optional[str]): API key for authentication.
        """
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.mem_fraction_static = mem_fraction_static
        self.max_running_requests = max_running_requests
        self.api_key = api_key

        # Instance-specific server state
        self._server_process = None
        self._server_host = None
        self._api_base = None
        self._port = None

        # Instance-specific async lock for proper synchronization
        self._server_lock = asyncio.Lock()

        # Logger with model-specific context
        self._logger = get_logger(
            f"sglang_server_{model_id.replace('/', '_')}")

    async def get_server(self) -> Tuple[str, int, str]:
        """
        Get or create an SGLang server instance with proper async synchronization.

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
                self._logger.info("Using existing SGLang server",
                                  port=self._port, model_id=self.model_id)
                return self._api_base, self._port, self._server_host

            # Launch new server
            self._logger.info("Launching new SGLang server",
                              model_id=self.model_id)
            (self._server_process, self._server_host,
             self._api_base, self._port) = await launch_server(
                model_id=self.model_id,
                reasoning_parser=self.reasoning_parser,
                mem_fraction_static=self.mem_fraction_static,
                max_running_requests=self.max_running_requests,
                api_key=self.api_key
            )

            if not (self._server_process and self._server_host and
                    self._api_base and self._port):
                raise RuntimeError(
                    f"Failed to launch SGLang server for model {self.model_id}")

            self._logger.info("SGLang server ready",
                              port=self._port, model_id=self.model_id)
            return self._api_base, self._port, self._server_host

    async def terminate_server(self):
        """
        Terminate the running SGLang server instance for this manager.
        """
        async with self._server_lock:
            if self._server_process is not None:
                self._logger.info("Terminating SGLang server",
                                  port=self._port, model_id=self.model_id)
                terminate_process(self._server_process)

                # Reset instance variables
                self._server_process = None
                self._server_host = None
                self._api_base = None
                self._port = None

                self._logger.info("SGLang server terminated",
                                  model_id=self.model_id)
            else:
                self._logger.info("No SGLang server running to terminate",
                                  model_id=self.model_id)

    async def get_openai_client(self,
                                max_tokens: int = 4096,
                                temperature: float = 0.0) -> GeneralOpenAIClient:
        """
        Get an OpenAI client connected to this SGLang server instance.

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
            "mem_fraction_static": self.mem_fraction_static,
            "max_running_requests": self.max_running_requests
        }


ALL_SGLANG_SERVERS: Dict[str, SGLangServerManager] = {}


def get_llm_mgr(model_id="Qwen/Qwen3-4B",
                reasoning_parser: Optional[str] = "qwen3",
                mem_fraction_static: Optional[float] = 0.4,
                max_running_requests: Optional[int] = 4,
                api_key: Optional[str] = None):
    if model_id not in ALL_SGLANG_SERVERS:
        ALL_SGLANG_SERVERS[model_id] = SGLangServerManager(
            model_id=model_id,
            reasoning_parser=reasoning_parser,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            api_key=api_key
        )
    return ALL_SGLANG_SERVERS[model_id]


async def main():
    """
    Test the async SGLang server implementation.
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
        logger.info("SGLang server is running", port=port, model_id=model_id)

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

        logger.info("Response from SGLang server", response=content)

    finally:
        # Clean up server
        await llm_server.terminate_server()


if __name__ == "__main__":
    asyncio.run(main())
