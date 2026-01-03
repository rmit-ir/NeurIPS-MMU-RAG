import asyncio
from typing import List, Optional, Tuple
from openai.types.chat import ChatCompletionMessageParam
from structlog import BoundLogger
from systems.rag_interface import RunStreamingResponse
from systems.vanilla_agent.model_config import default_config, gpt_oss_config
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.reranker_vllm import GeneralReranker, get_reranker
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult
from tools.lse_search import search_clueweb
from tools.docs_utils import reciprocal_rank_fusion


def build_llm_messages(
    results: str | list[SearchResult],
    query: str,
    enable_think: bool,
    model_id: Optional[str] = None
) -> List[ChatCompletionMessageParam]:
    """Build LLM messages using the appropriate model config.

    Args:
        results: Search results or context string
        query: User's question
        enable_think: Whether to enable thinking mode
        model_id: Model identifier to determine config (optional)

    Returns:
        List of chat completion messages
    """
    # Determine which config to use
    if model_id and gpt_oss_config.is_gpt_oss_model(model_id):
        config = gpt_oss_config
    else:
        config = default_config

    return config.build_answer_messages(results, query, enable_think)


def build_to_context(results: list[SearchResult]) -> str:
    """Build context string from search results.

    Uses default_config as both configs have the same implementation.
    """
    return default_config.build_to_context(results)


def inter_resp(desc: str, silent: bool, logger: BoundLogger) -> RunStreamingResponse:
    if not silent:
        logger.info(f"Intermediate step | {desc}")
    return RunStreamingResponse(
        intermediate_steps=desc,
        is_intermediate=True,
        complete=False
    )


async def generate_qvs(query: str,
                       num_qvs: int,
                       enable_think: bool,
                       logger: BoundLogger,
                       preset_llm: Optional[GeneralOpenAIClient] = None,
                       model_id: Optional[str] = None) -> List[str]:
    """Generate a list of query variants"""
    if not preset_llm:
        llm, _ = await get_default_llms()
    else:
        llm = preset_llm

    # Determine which config to use
    if model_id and gpt_oss_config.is_gpt_oss_model(model_id):
        config = gpt_oss_config
    else:
        config = default_config

    system_prompt = config.QUERY_VARIANTS_PROMPT(num_qvs, enable_think)
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {query}"},
    ]
    response, _ = await llm.complete_chat(messages)
    if response:
        queries_str = extract_tag_val(response.strip(), "queries", True)
        if queries_str:
            variants = [line.strip("- ").strip()
                        for line in queries_str.split("\n") if line.strip()]
            return variants[:num_qvs]
    logger.warning("Failed to generate query variants, using original query.",
                   query=query, enable_think=enable_think, response=response)
    return [query]


async def reformulate_query(query: str, preset_llm: Optional[GeneralOpenAIClient] = None, model_id: Optional[str] = None) -> str:
    """Reformulate the query to improve search results"""
    if not preset_llm:
        llm, _ = await get_default_llms()
    else:
        llm = preset_llm

    # TODO: extract this (as well as all other places using this code) to a function get_model_config() in model_config/__init__.py
    # Determine which config to use
    if model_id and gpt_oss_config.is_gpt_oss_model(model_id):
        config = gpt_oss_config
    else:
        config = default_config

    system_prompt = config.REFORMULATE_QUERY_PROMPT()
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {query}"},
    ]
    response, _ = await llm.complete_chat(messages)
    if response:
        return response.strip()
    return query


async def search_w_qv(query: str,
                      num_qvs: int,
                      enable_think: bool,
                      logger: BoundLogger,
                      cw22_a: bool = True,
                      preset_llm: Optional[GeneralOpenAIClient] = None) -> Tuple[List[str], List[SearchResult]]:
    """Search with query variants and merge results using Reciprocal Rank Fusion"""
    queries = await generate_qvs(query, num_qvs, enable_think, logger=logger, preset_llm=preset_llm)
    queries = set([query, *queries])
    ranked_lists = await asyncio.gather(*[
        search_clueweb(query=q, k=10, cw22_a=cw22_a) for q in queries])

    # Apply Reciprocal Rank Fusion to combine rankings
    all_results = reciprocal_rank_fusion(ranked_lists)

    logger.info("Search with query variants completed, merged with RRF",
                original_query=query,
                num_variants=len(queries),
                all_results=len(all_results))

    return list(queries), all_results


global_llm_client: GeneralOpenAIClient | None = None
global_reranker: GeneralReranker | None = None
global_llm_model_id: str | None = None


async def get_default_llms():
    global global_llm_client, global_reranker, global_llm_model_id
    if not global_llm_client:
        llm_mgr = get_llm_mgr(VllmConfig())
        global_llm_client = await llm_mgr.get_openai_client(
            max_tokens=4_096,
        )
        # Get model_id from the client
        global_llm_model_id = getattr(global_llm_client, 'model_id', None)
    if not global_reranker:
        global_reranker = await get_reranker()
    return global_llm_client, global_reranker, global_llm_model_id


async def main():
    print(await reformulate_query("I'm using vllm to run Qwen/Qwen3-4B model, now I'm sending it a string prompt, how can I use python libraries calculate how many tokens in my string prompt before I send it over?"))

if __name__ == "__main__":
    asyncio.run(main())
