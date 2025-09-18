import asyncio
import time
from typing import Optional

import requests
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import terminate_process
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.sglang_utils import wait_for_server
from tools.logging_utils import get_logger

logger = get_logger("sglang_server")


def launch_server(model_id="Qwen/Qwen3-4B",
                  reasoning_parser: Optional[str] = "qwen3",
                  mem_fraction_static: Optional[float] = 0.4,
                  max_running_requests: Optional[int] = 4,
                  api_key: Optional[str] = None):
    """
    Launch the SGLang server as a subprocess.
    Args:
        model_id (str): The model ID to use.
        mem_fraction_static (float): Fraction of memory to allocate statically.
        max_running_requests (int): Maximum number of concurrent running requests.
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
    server_process, port = launch_server_cmd(' '.join(command))

    server_host = f"http://localhost:{port}"
    api_base = f"{server_host}/v1"
    wait_for_server(server_host, timeout=1800, api_key=api_key)
    logger.info("SGLang server is running", port=port)
    return server_process, api_base, port


def terminate_server(server_process):
    terminate_process(server_process)
    logger.info("SGLang server terminated.")


async def main():
    model_id = "Qwen/Qwen3-4B"
    api_key = "abc"
    server_process, api_base, port = launch_server(model_id=model_id,
                                                   api_key=api_key)
    openai_client = GeneralOpenAIClient(api_base=api_base,
                                        api_key=api_key,
                                        model_id=model_id,
                                        temperature=0)
    content = openai_client.complete_chat([
        {"role": "user", "content": "I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community.' I'm also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building."}
    ])
    logger.info("Response from SGLang server", response=content)
    terminate_server(server_process)


if __name__ == "__main__":
    asyncio.run(main())
