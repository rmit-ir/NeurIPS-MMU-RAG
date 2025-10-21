import asyncio

from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.logging_utils import get_logger
from tools.reranker_vllm import RerankerConfig, get_reranker


logger = get_logger('warmup')


async def warmup_models():
    # Launch both services in parallel using separate threads
    logger.info("Starting reranker_vllm reranker and LLM manager in parallel...")

    vllm_config = VllmConfig(model_id="Qwen/Qwen3-4B",
                             reasoning_parser="qwen3",
                             gpu_memory_utilization=0.75,
                             max_model_len=25_000,
                             # model=7.56GB, arch=1.4+4.1+0.61=6.11GB, kv_cache=8GB
                             kv_cache_memory=8*1024**3,
                             max_num_seqs=5,)
    reranker_config = RerankerConfig(model_id="Qwen/Qwen3-Reranker-0.6B",
                                     gpu_memory_utilization=0.2,
                                     max_model_len=16_000,
                                     kv_cache_memory_bytes=3*1024**3,
                                     max_num_seqs=3,)

    vllm_mgr = get_llm_mgr(vllm_config)
    # Run both initialization tasks concurrently
    await asyncio.gather(
        # main model has to run first, otherwise it will OOM
        vllm_mgr.get_server(),
        get_reranker(reranker_config, drop_irrelevant_threshold=0.5),
    )

    logger.info("Reranker and LLM manager ready.")
