import time
from typing import Optional

import requests
from tools.logging_utils import get_logger


logger = get_logger("sglang_server")


def wait_for_server(base_url: str, timeout: Optional[int] = None, api_key: Optional[str] = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    if not timeout:
        timeout = 3600  # Default to 1 hour if no timeout is provided

    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}"} if api_key else None,
            )
            if response.status_code == 200:
                logger.info("Server is ready.", url=base_url)
                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError(
                    "Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)
