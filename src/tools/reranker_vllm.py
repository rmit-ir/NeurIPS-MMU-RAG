"""VLLM-based reranker implementation using Qwen3-Reranker."""

import asyncio
import atexit
import json
import signal
import subprocess
import sys
import threading
from typing import List, Optional, Dict, Any, Tuple, Callable
import aiohttp

from tools.web_search import SearchResult
from tools.llm_servers.sglang_utils import wait_for_server
from tools.logging_utils import get_logger

logger = get_logger('reranker_vllm')


def terminate_process(process):
    """Terminate a process and its children."""
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def launch_reranker_server(model_id="Qwen/Qwen3-Reranker-0.6B",
                                 gpu_memory_utilization: Optional[float] = 0.2,
                                 max_model_len: Optional[int] = 16000,
                                 kv_cache_memory_bytes: Optional[int] = None,
                                 hf_overrides: Optional[Dict[str, Any]] = None,
                                 host: str = "0.0.0.0",
                                 port: Optional[int] = None,
                                 api_key: Optional[str] = None):
    """
    Launch the vLLM reranker server as a subprocess asynchronously.

    Args:
        model_id (str): The reranker model ID to use.
        gpu_memory_utilization (Optional[float]): GPU memory utilization fraction.
        max_model_len (Optional[int]): Maximum model length.
        kv_cache_memory_bytes (Optional[int]): KV cache memory in bytes.
        hf_overrides (Optional[Dict[str, Any]]): HuggingFace model overrides.
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

    # Default KV cache memory for reranker
    if kv_cache_memory_bytes is None:
        kv_cache_memory_bytes = 2 * 1024 * 1024 * 1024  # 2GB

    # Default hf_overrides for Qwen3-Reranker
    if hf_overrides is None:
        hf_overrides = {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True
        }

    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--runner", "pooling",
        *(["--gpu-memory-utilization", str(gpu_memory_utilization)]
          if gpu_memory_utilization else []),
        *(["--max-model-len", str(max_model_len)] if max_model_len else []),
        *(["--kv-cache-memory-bytes", str(kv_cache_memory_bytes)]
          if kv_cache_memory_bytes else []),
        *(["--hf-overrides", json.dumps(hf_overrides)] if hf_overrides else []),
        "--host", host,
        "--port", str(port),
    ]

    if api_key:
        command.extend(["--api-key", api_key])

    logger.info("Launching vLLM reranker server", command=' '.join(command))

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
    logger.info("vLLM reranker server is running", port=port)

    def terminate():
        terminate_process(server_process)
    return server_process, terminate, server_host, api_base, port


class GeneralReranker:
    """VLLM-based reranker using Qwen3-Reranker model with API server."""

    def __init__(self,
                 model_name="Qwen/Qwen3-Reranker-0.6B",
                 drop_irrelevant_threshold: Optional[float] = 0.5,
                 gpu_memory_utilization: Optional[float] = 0.2,
                 max_model_len: Optional[int] = 16000,
                 kv_cache_memory_bytes: Optional[int] = None,
                 hf_overrides: Optional[Dict[str, Any]] = None,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 api_key: Optional[str] = None):
        """Initialize the VLLM reranker with API server."""
        self.model_name = model_name
        self.drop_irrelevant_threshold = drop_irrelevant_threshold
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.hf_overrides = hf_overrides
        self.host = host
        self.port = port
        self.api_key = api_key

        # Server state
        self._server_process = None
        self._server_terminate_fn: Optional[Callable[[], None]] = None
        self._server_host = None
        self._api_base = None
        self._port = None

        # Async lock for proper synchronization
        self._server_lock = asyncio.Lock()

        # HTTP client for making requests
        self._http_client: Optional[aiohttp.ClientSession] = None

        # Templates that work for Qwen3-Reranker only
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        # Logger with model-specific context
        self._logger = get_logger(f"reranker_{model_name.replace('/', '_')}")

    async def _get_server(self) -> Tuple[str, int, str]:
        """
        Get or create a vLLM reranker server instance with proper async synchronization.

        Returns:
            Tuple containing (api_base, port, server_host)
        """
        async with self._server_lock:
            # Check if server is already running
            if (self._server_process and self._server_host and
                    self._api_base and self._port):
                self._logger.info("Using existing vLLM reranker server",
                                  port=self._port, model_name=self.model_name)
                return self._api_base, self._port, self._server_host

            # Launch new server
            self._logger.info("Launching new vLLM reranker server",
                              model_name=self.model_name)
            (self._server_process, self._server_terminate_fn, self._server_host,
             self._api_base, self._port) = await launch_reranker_server(
                model_id=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                kv_cache_memory_bytes=self.kv_cache_memory_bytes,
                hf_overrides=self.hf_overrides,
                host=self.host,
                port=self.port,
                api_key=self.api_key
            )

            if not (self._server_process and self._server_host and
                    self._api_base and self._port):
                raise RuntimeError(
                    f"Failed to launch vLLM reranker server for model {self.model_name}")

            self._logger.info("vLLM reranker server ready",
                              port=self._port, model_name=self.model_name)
            return self._api_base, self._port, self._server_host

    def _get_http_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client."""
        if self._http_client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            timeout = aiohttp.ClientTimeout(total=300.0)  # 5 minutes timeout
            self._http_client = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
        return self._http_client

    def _cut_to_words(self, text: str, max_words: int) -> str:
        """Cut the text to the first max_words words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words])

    def _search_result_to_text(self, result: SearchResult) -> str:
        """Convert SearchResult to formatted text."""
        return f"Web URL: {result.url.strip()}\n\nContent: {result.text.strip()}\n\n"

    async def _score_via_api(self, query_fmt: str, docs_fmt: List[str]) -> List[float]:
        """
        Score documents via API requests to the vLLM server.

        Args:
            query_fmt: Formatted query string
            docs_fmt: List of formatted document strings

        Returns:
            List of scores for each document
        """
        try:
            api_base, _, _ = await self._get_server()
            http_client = self._get_http_client()

            # Prepare the request payload for vLLM score endpoint
            payload = {
                "model": self.model_name,
                "query": query_fmt,
                "documents": docs_fmt
            }

            # Make request to the score endpoint
            async with http_client.post(
                f"{api_base}/score",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                scores = result.get("scores", [0.0] * len(docs_fmt))

            return scores

        except Exception as e:
            self._logger.error("Error during vLLM API scoring", error=str(e))
            # Fallback: return zero scores
            return [0.0] * len(docs_fmt)

    async def rerank(self, query: str, search_results: List[SearchResult], max_words: int = 4000) -> List[SearchResult]:
        """
        Asynchronously rerank search results based on query relevance.

        Args:
            query: The search query
            search_results: List of SearchResult objects to rerank
            max_words: Maximum words per document (default: 4000)

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        if not search_results:
            self._logger.warning("No search results to rerank")
            return []

        instruction = (
            "Given the web search query, is the retrieved document "
            "(1) from a high quality and relevant website based on the URL, "
            "(2) published recently, and "
            "(3) contains key information that helps answering the query?"
        )

        # Format query and docs
        query_fmt = self.query_template.format(
            prefix=self.prefix,
            instruction=instruction,
            query=query
        )
        docs_fmt = [
            self.document_template.format(
                doc=self._cut_to_words(
                    self._search_result_to_text(result), max_words),
                suffix=self.suffix
            )
            for result in search_results
        ]

        # Get scores from vLLM via API
        scores = await self._score_via_api(query_fmt, docs_fmt)

        self._logger.info("Re-ranking completed",
                          num_results=len(search_results),
                          query_length=len(query))

        # Create ranked results
        ranked_results = [
            result._replace(score=score)
            for result, score in zip(search_results, scores)
        ]

        # Sort by score descending
        ranked_results.sort(key=lambda x: x.score or 0.0, reverse=True)

        if self.drop_irrelevant_threshold is not None:
            # Filter out results with scores below threshold
            ranked_results = [
                res for res in ranked_results
                if (res.score or 0.0) > self.drop_irrelevant_threshold
            ]
            self._logger.info("Filtered irrelevant results",
                              num_remaining=len(ranked_results))

        return ranked_results

    def _terminate_server(self):
        """Terminate the running vLLM reranker server instance."""
        if self._server_process is not None:
            self._logger.info("Terminating vLLM reranker server",
                              port=self._port, model_name=self.model_name)
            if self._server_terminate_fn:
                self._server_terminate_fn()

            # Reset instance variables
            self._server_process = None
            self._server_host = None
            self._api_base = None
            self._port = None

            self._logger.info("vLLM reranker server terminated",
                              model_name=self.model_name)
        else:
            self._logger.info("No vLLM reranker server running to terminate",
                              model_name=self.model_name)

        # Close HTTP client
        if self._http_client:
            asyncio.create_task(self._http_client.close())
            self._http_client = None

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
            "model_name": self.model_name,
            "is_running": self.is_running,
            "port": self._port,
            "api_base": self._api_base,
            "server_host": self._server_host,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "host": self.host,
            "drop_irrelevant_threshold": self.drop_irrelevant_threshold
        }


# Global registry for reranker instances
ALL_RERANKERS: Dict[str, GeneralReranker] = {}


def cleanup_all_rerankers(signum: Optional[int], frame: Optional[Any]):
    """Cleanup function to terminate all running vLLM reranker servers."""
    logger.info(f"Clean up reranker servers on signal {signum}, frame {frame}")
    if not ALL_RERANKERS:
        return

    logger.info("Cleaning up all vLLM reranker servers before exit")

    # Start termination threads for all running servers
    threads = []
    for reranker in ALL_RERANKERS.values():
        if reranker.is_running:
            thread = threading.Thread(target=reranker._terminate_server)
            thread.start()
            threads.append(thread)

    # Wait for all terminations to complete
    for thread in threads:
        thread.join(timeout=10)
    sys.exit(0)


# Register cleanup function for normal exit and termination signals
atexit.register(lambda: cleanup_all_rerankers(None, None))
signal.signal(signal.SIGTERM, cleanup_all_rerankers)
signal.signal(signal.SIGINT, cleanup_all_rerankers)

# On Unix systems, also handle SIGHUP
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, cleanup_all_rerankers)


def get_reranker(model_name="Qwen/Qwen3-Reranker-0.6B",
                 drop_irrelevant_threshold: Optional[float] = 0.5,
                 gpu_memory_utilization: Optional[float] = 0.2,
                 max_model_len: Optional[int] = 16000,
                 kv_cache_memory_bytes: Optional[int] = None,
                 hf_overrides: Optional[Dict[str, Any]] = None,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 api_key: Optional[str] = None) -> GeneralReranker:
    """
    Get a reranker instance, creating if necessary.

    Args:
        model_name: The reranker model name
        drop_irrelevant_threshold: Threshold for filtering irrelevant results
        gpu_memory_utilization: GPU memory utilization fraction
        max_model_len: Maximum model length
        kv_cache_memory_bytes: KV cache memory in bytes
        hf_overrides: HuggingFace model overrides
        host: Host to bind the server to
        port: Port to bind the server to
        api_key: API key for authentication

    Returns:
        GeneralReranker instance
    """
    if model_name not in ALL_RERANKERS:
        ALL_RERANKERS[model_name] = GeneralReranker(
            model_name=model_name,
            drop_irrelevant_threshold=drop_irrelevant_threshold,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            hf_overrides=hf_overrides,
            host=host,
            port=port,
            api_key=api_key
        )
    return ALL_RERANKERS[model_name]
