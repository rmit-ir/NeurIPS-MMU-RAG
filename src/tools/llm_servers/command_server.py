"""Generic command server for managing subprocess-based servers with async coordination."""

import asyncio
import subprocess
import socket
from typing import List, NamedTuple, Callable, Awaitable

from tools.logging_utils import get_logger

logger = get_logger('command_server')


def terminate_process(process):
    """Terminate a process and its children."""
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def find_available_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class RunningServer(NamedTuple):
    process: subprocess.Popen
    terminate: Callable[[], None]


async def launch_command_server(command: List[str],
                                health_check_fn: Callable[[], Awaitable[None]],
                                server_name: str = "Command Server") -> RunningServer:
    """
    Launch a command server as a subprocess asynchronously.

    Args:
        command: Complete command list for subprocess execution
        health_check_fn: Async function to check if server is healthy
        server_name: Name for logging purposes

    Returns:
        Tuple containing (server_process, terminate_fn)
    """
    logger.info(f"Launching {server_name}", command=' '.join(command))

    # Run the server launch in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    server_process = await loop.run_in_executor(
        None,
        lambda: subprocess.Popen(command)
    )

    # Run health check
    await health_check_fn()
    logger.info(f"{server_name} is running")

    def terminate():
        terminate_process(server_process)

    return RunningServer(process=server_process, terminate=terminate)
