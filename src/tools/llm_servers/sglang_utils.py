import asyncio
import time
from typing import Optional

import aiohttp
from tools.logging_utils import get_logger


logger = get_logger("sglang_utils")


async def wait_for_server(base_url: str, timeout: Optional[int] = None, api_key: Optional[str] = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
        api_key: Optional API key for authentication
    """
    start_time = time.perf_counter()
    if not timeout:
        timeout = 3600  # Default to 1 hour if no timeout is provided

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(f"{base_url}/v1/models", headers=headers) as response:
                    if response.status == 200:
                        logger.info("Server is ready.", url=base_url)
                        return

                if timeout and time.perf_counter() - start_time > timeout:
                    raise TimeoutError(
                        "Server did not become ready within timeout period")
            except aiohttp.ClientError:
                pass  # Continue trying

            await asyncio.sleep(1)
