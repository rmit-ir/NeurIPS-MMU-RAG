import asyncio
import time
from typing import Optional

import aiohttp
from tools.logging_utils import get_logger


logger = get_logger("sglang_utils")


async def test_server(base_url: str, api_key: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    async def _request_server(session: aiohttp.ClientSession):
        async with session.get(f"{base_url}/v1/models", headers=headers) as response:
            return response.status == 200

    if session:
        return await _request_server(session)
    else:
        async with aiohttp.ClientSession() as session:
            return await _request_server(session)


async def wait_for_server(base_url: str, timeout: Optional[int] = None, api_key: Optional[str] = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Return:

    1. When 200

    Raise:

    1. When timeout is reached.

    Otherwise keep trying.
    """
    start_time = time.perf_counter()
    tried_times = 0
    if not timeout:
        timeout = 3600  # Default to 1 hour if no timeout is provided

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                success = await test_server(base_url, api_key, session)
                if success:
                    logger.info("Server is ready.", url=base_url)
                    return

                tried_times += 1
                duration = time.perf_counter() - start_time
                if timeout and duration > timeout:
                    raise TimeoutError(
                        f"Server not ready after {tried_times} tries in {duration:.2f} seconds.")
            except aiohttp.ClientError:
                pass  # Continue trying

            await asyncio.sleep(1)
