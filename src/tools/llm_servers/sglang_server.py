import asyncio
from typing import Optional

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

################################################################################
# global running server instance
running_server_process = None
running_server_host = None
running_api_base = None
running_port = None
# running_model_id = None # let's not support changing model for now
# running_params: Optional[dict] = None
# Async lock for proper synchronization
_server_lock = asyncio.Lock()


async def get_sglang_server(model_id="Qwen/Qwen3-4B",
                            reasoning_parser: Optional[str] = "qwen3",
                            mem_fraction_static: Optional[float] = 0.4,
                            max_running_requests: Optional[int] = 4,
                            api_key: Optional[str] = None):
    """
    Get or create an SGLang server instance with proper async synchronization.

    This function ensures that only one server is launched at a time and that
    multiple concurrent requests will wait for the same server instance.

    With asyncio.Lock(), requests don't need to sleep and poll - they wait
    efficiently until the lock is available. The first request will launch
    the server, and subsequent requests will immediately get the existing server.

    Wait time per request:
    - First request: Time to launch server + health check (typically 30-60 seconds)
    - Concurrent requests: Minimal wait (microseconds) until lock is released
    - Subsequent requests: Immediate return with existing server info

    Args:
        model_id (str): The model ID to use.
        reasoning_parser (Optional[str]): The reasoning parser to use.
        mem_fraction_static (float): Fraction of memory to allocate statically.
        max_running_requests (int): Maximum number of concurrent running requests.
        api_key (Optional[str]): API key for authentication.

    Returns:
        Tuple containing (server_process, server_host, api_base, port)
    """
    global running_server_process, running_server_host, running_api_base, running_port

    # Use async lock to prevent race conditions - no polling needed!
    async with _server_lock:
        # Check if server is already running
        if running_server_process and running_server_host and running_api_base and running_port:
            logger.info("Using existing SGLang server", port=running_port)
            return running_api_base, running_port, running_server_host

        # Launch new server
        logger.info("Launching new SGLang server", model_id=model_id)
        running_server_process, running_server_host, running_api_base, running_port = await launch_server(
            model_id=model_id,
            reasoning_parser=reasoning_parser,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            api_key=api_key
        )
        if not (running_server_process and running_server_host and running_api_base and running_port):
            raise RuntimeError("Failed to launch SGLang server properly")

        logger.info("SGLang server ready", port=running_port)
        return running_api_base, running_port, running_server_host


async def terminate_sglang_server():
    """
    Terminate the running SGLang server instance.
    """
    global running_server_process, running_server_host, running_api_base, running_port

    async with _server_lock:
        if running_server_process is not None:
            logger.info("Terminating SGLang server", port=running_port)
            terminate_process(running_server_process)

            # Reset global variables
            running_server_process = None
            running_server_host = None
            running_api_base = None
            running_port = None

            logger.info("SGLang server terminated")
        else:
            logger.info("No SGLang server running to terminate")


async def get_openai_client(model_id="Qwen/Qwen3-4B",
                            reasoning_parser: Optional[str] = "qwen3",
                            mem_fraction_static: Optional[float] = 0.4,
                            max_running_requests: Optional[int] = 4,
                            api_key: Optional[str] = None,
                            temperature: float = 0.0):
    api_base, port, server_host = await get_sglang_server(
        model_id=model_id,
        reasoning_parser=reasoning_parser,
        mem_fraction_static=mem_fraction_static,
        max_running_requests=max_running_requests,
        api_key=api_key
    )

    return GeneralOpenAIClient(
        api_base=api_base,
        api_key=api_key,
        model_id=model_id,
        temperature=temperature
    )
################################################################################


async def main():
    """
    Test the async SGLang server implementation.
    """
    model_id = "Qwen/Qwen3-4B"
    api_key = "abc"

    try:
        # Get server instance using the new async method
        api_base, port, server_host = await get_sglang_server(
            model_id=model_id,
            api_key=api_key
        )

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
        await terminate_sglang_server()


if __name__ == "__main__":
    asyncio.run(main())
