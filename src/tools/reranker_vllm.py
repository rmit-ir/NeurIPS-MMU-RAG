"""VLLM-based reranker implementation using Qwen3-Reranker."""

import asyncio
import atexit
import json
import signal
import sys
from typing import List, Optional, Dict, Any, Tuple
import aiohttp

from tools.web_search import SearchResult
from tools.llm_servers.command_server import launch_command_server, find_available_port, RunningServer
from tools.llm_servers.sglang_utils import wait_for_server
from tools.logging_utils import get_logger

logger = get_logger('reranker_vllm')

# Global server state
_reranker_server: Optional[RunningServer] = None
_server_lock = asyncio.Lock()
_server_host = None
_api_base = None
_port = None


def build_reranker_command(model_id: str,
                           gpu_memory_utilization: Optional[float] = 0.2,
                           max_model_len: Optional[int] = 16000,
                           kv_cache_memory_bytes: Optional[int] = None,
                           hf_overrides: Optional[Dict[str, Any]] = None,
                           host: str = "0.0.0.0",
                           port: Optional[int] = None,
                           api_key: Optional[str] = None) -> Tuple[List[str], str, str, int]:
    """Build command for vLLM reranker server."""
    # Find an available port if not specified
    if port is None:
        port = find_available_port()

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

    server_host = f"http://{host}:{port}"
    api_base = f"{server_host}/v1"

    return command, server_host, api_base, port


async def ensure_reranker_server(model_id: str,
                                 gpu_memory_utilization: Optional[float],
                                 max_model_len: Optional[int],
                                 kv_cache_memory_bytes: Optional[int],
                                 hf_overrides: Optional[Dict[str, Any]],
                                 host: str,
                                 port: Optional[int],
                                 api_key: Optional[str]) -> Tuple[str, str, int]:
    """Ensure reranker server is running, launch if needed."""
    global _reranker_server, _server_host, _api_base, _port

    async with _server_lock:
        # Check if server is already running
        if _reranker_server and _server_host and _api_base and _port:
            logger.info("Using existing vLLM reranker server", port=_port)
            return _api_base, _server_host, _port

        # Build command and server info
        command, server_host, api_base, server_port = build_reranker_command(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            hf_overrides=hf_overrides,
            host=host,
            port=port,
            api_key=api_key
        )

        # Health check function
        async def health_check():
            await wait_for_server(server_host, timeout=1800, api_key=api_key)

        # Launch server using command server
        _reranker_server = await launch_command_server(
            command=command,
            health_check_fn=health_check,
            server_name="vLLM Reranker Server"
        )

        _server_host = server_host
        _api_base = api_base
        _port = server_port

        logger.info("vLLM reranker server ready", port=_port)
        return _api_base, _server_host, _port


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

        # HTTP client for making requests
        self._http_client: Optional[aiohttp.ClientSession] = None

        # Templates that work for Qwen3-Reranker only
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        # Logger with model-specific context
        self._logger = get_logger(f"reranker_{model_name.replace('/', '_')}")

    async def _get_server(self) -> Tuple[str, str, int]:
        """Get or ensure reranker server is running."""
        return await ensure_reranker_server(
            model_id=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            hf_overrides=self.hf_overrides,
            host=self.host,
            port=self.port,
            api_key=self.api_key
        )

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
        global _reranker_server
        if _reranker_server is not None:
            self._logger.info("Terminating vLLM reranker server",
                              port=_port, model_name=self.model_name)
            _reranker_server.terminate()

        # Close HTTP client
        if self._http_client:
            asyncio.create_task(self._http_client.close())
            self._http_client = None

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return _reranker_server is not None

    @property
    def server_info(self) -> Dict[str, Any]:
        """Get current server information."""
        return {
            "model_name": self.model_name,
            "is_running": self.is_running,
            "port": _port,
            "api_base": _api_base,
            "server_host": _server_host,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "host": self.host,
            "drop_irrelevant_threshold": self.drop_irrelevant_threshold
        }


# Global registry for reranker instances
ALL_RERANKERS: Dict[str, GeneralReranker] = {}


def cleanup_all_rerankers(signum: Optional[int], frame: Optional[Any]):
    """Cleanup function to terminate all running vLLM reranker servers."""
    global _reranker_server
    logger.info(f"Clean up reranker servers on signal {signum}, frame {frame}")

    if _reranker_server:
        logger.info("Cleaning up vLLM reranker server before exit")
        _reranker_server.terminate()
        _reranker_server = None

    sys.exit(0)


# Register cleanup function for normal exit and termination signals
atexit.register(lambda: cleanup_all_rerankers(None, None))
signal.signal(signal.SIGTERM, cleanup_all_rerankers)
signal.signal(signal.SIGINT, cleanup_all_rerankers)

# On Unix systems, also handle SIGHUP
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, cleanup_all_rerankers)


async def get_reranker(model_name="Qwen/Qwen3-Reranker-0.6B",
                       drop_irrelevant_threshold: Optional[float] = 0.5,
                       gpu_memory_utilization: Optional[float] = 0.2,
                       max_model_len: Optional[int] = 16000,
                       kv_cache_memory_bytes: Optional[int] = None,
                       hf_overrides: Optional[Dict[str, Any]] = None,
                       host: str = "0.0.0.0",
                       port: Optional[int] = None,
                       api_key: Optional[str] = None) -> GeneralReranker:
    """
    Get a reranker instance, creating if necessary and ensuring server is started.
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

    # Ensure the server is started
    reranker = ALL_RERANKERS[model_name]
    await reranker._get_server()
    return reranker


async def main():
    """
    Test the vLLM reranker implementation.
    """
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    api_key = "abc"

    def _search_result(sid: str, text: str) -> SearchResult:
        return SearchResult(
            url="https://example.com",
            text=text,
            score=None,
            date=None,
            dump=None,
            file_path=None,
            metadata={},
            language=None,
            language_score=None,
            token_count=len(text.split()),
            type="clue_web",
            sid=sid,
            id=sid,
        )

    # Create sample search results for testing
    sample_results = [
        _search_result(
            "1", "This article discusses machine learning and artificial intelligence applications in modern technology."),
        _search_result(
            "2", "A comprehensive guide to cooking pasta with various Italian recipes and techniques."),
        _search_result(
            "3", "Deep learning neural networks are transforming how we approach complex AI problems."),
    ]

    # Get reranker instance
    reranker = await get_reranker(
        model_name=model_name,
        api_key=api_key,
        drop_irrelevant_threshold=0.3
    )

    logger.info("vLLM reranker server is running",
                model_name=model_name,
                server_info=reranker.server_info)

    # Test reranking with a query about AI
    query = "What are the latest developments in artificial intelligence and machine learning?"

    reranked_results = await reranker.rerank(query, sample_results)

    logger.info("Reranking completed successfully",
                query=query,
                num_original=len(sample_results),
                num_reranked=len(reranked_results))

    # Print results
    for i, result in enumerate(reranked_results):
        logger.info(f"Rank {i+1}",
                    text=result.text[:100] +
                    ("..." if len(result.text) > 100 else ""),
                    score=result.score,
                    url=result.url)


if __name__ == "__main__":
    asyncio.run(main())
