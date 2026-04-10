import asyncio
import os
from typing import List, Optional, Tuple
from openai.types.chat import ChatCompletionMessageParam
from structlog import BoundLogger
from systems.rag_interface import RunStreamingResponse
from systems.vanilla_agent.model_config import get_model_config
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.reranker_vllm import GeneralReranker, get_reranker
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult
from tools.lse_search import search_clueweb
from tools.brave_search import brave_search
from tools.brave_llm_context import brave_llm_context
from tools.jina_retriever import retrieve_urls
from tools.docs_utils import reciprocal_rank_fusion, chunk_docs


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
    config = get_model_config(model_id)
    return config.build_answer_messages(results, query, enable_think)


def build_to_context(results: list[SearchResult]) -> str:
    """Build context string from search results.

    Uses default_config as both configs have the same implementation.
    """
    from systems.vanilla_agent.model_config import default_config
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
                       preset_llm: Optional[GeneralOpenAIClient] = None) -> List[str]:
    """Generate a list of query variants"""
    if not preset_llm:
        llm, _ = await get_default_llms()
    else:
        llm = preset_llm

    config = get_model_config(llm.model_id)
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


async def rewrite_search_query(query: str, preset_llm: Optional[GeneralOpenAIClient] = None) -> str:
    """Rewrite a natural language question into a keyword-optimized search query."""
    if not preset_llm:
        llm, _ = await get_default_llms()
    else:
        llm = preset_llm

    config = get_model_config(llm.model_id)
    system_prompt = config.SEARCH_QUERY_REWRITE_PROMPT()
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response, _ = await llm.complete_chat(messages)
    if response:
        return response.strip().strip('"\'')
    return query


async def reformulate_query(query: str, preset_llm: Optional[GeneralOpenAIClient] = None) -> str:
    """Reformulate the query to improve search results"""
    if not preset_llm:
        llm, _ = await get_default_llms()
    else:
        llm = preset_llm

    config = get_model_config(llm.model_id)
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
                      search_engine: str = "clueweb22b",
                      preset_llm: Optional[GeneralOpenAIClient] = None,
                      chunk_max_words: int = 300,
                      chunk_overlap_words: int = 50) -> Tuple[List[str], List[SearchResult]]:
    """Search with query variants and merge results using Reciprocal Rank Fusion.

    Args:
        query: The original search query
        num_qvs: Number of query variants to generate
        enable_think: Whether to enable thinking mode for LLM
        logger: Bound logger instance
        cw22_a: Use ClueWeb22-A instead of B (only for clueweb22b engine)
        search_engine: Search engine to use. Options:
            - "clueweb22b" (default): Uses LSE search with ClueWeb22
            - "brave_jina": Uses Brave search + Jina AI Reader for content retrieval
            - "brave_llm_context": Uses Brave LLM Context API (pre-extracted content, no Jina needed)
        preset_llm: Optional pre-configured LLM client
        chunk_max_words: Maximum words per chunk (default 300)
        chunk_overlap_words: Overlapping words between chunks (default 50)

    Returns:
        Tuple of (list of query variants used, list of SearchResults)
    """
    if search_engine == "brave_llm_context":
        # For Brave LLM Context, use a single keyword-optimized rewrite + original query.
        # Multiple variants don't help here since results rarely overlap for RRF.
        rewritten = await rewrite_search_query(query, preset_llm=preset_llm)
        queries = set([query, rewritten])
        ranked_lists = await asyncio.gather(*[
            brave_llm_context(q, count=10) for q in queries])
    elif search_engine == "brave_jina":
        queries = await generate_qvs(query, num_qvs, enable_think, logger=logger, preset_llm=preset_llm)
        queries = set([query, *queries])
        ranked_lists = await asyncio.gather(*[
            _search_with_jina(q, k=10) for q in queries])
    else:
        queries = await generate_qvs(query, num_qvs, enable_think, logger=logger, preset_llm=preset_llm)
        queries = set([query, *queries])
        ranked_lists = await asyncio.gather(*[
            search_clueweb(query=q, k=10, cw22_a=cw22_a) for q in queries])

    # Apply Reciprocal Rank Fusion to combine rankings
    all_results = reciprocal_rank_fusion(ranked_lists)

    # Chunk large documents into smaller pieces for better reranking
    # Skip for brave_llm_context — content is already pre-extracted snippets
    if search_engine != "brave_llm_context":
        all_results = chunk_docs(all_results, max_words=chunk_max_words, overlap_words=chunk_overlap_words)

    logger.info("Search with query variants completed, merged with RRF",
                original_query=query,
                num_variants=len(queries),
                search_engine=search_engine,
                all_results=len(all_results))

    return list(queries), all_results


async def _search_with_jina(query: str, k=10) -> List[SearchResult]:
    """Search using Brave search and retrieve content with Jina.

    Args:
        query: Search query
        k: Number of results to retrieve
    Returns:
        List of SearchResult objects with content from Jina
    """
    # Step 1: Get URLs from Brave search
    search_results = await brave_search(query, count=k)
    timeout = int(os.getenv("JINA_RETRIEVER_TIMEOUT", '5'))

    if not search_results:
        return []

    # Step 2: Extract URLs and fetch content with Jina
    urls = [r["url"] for r in search_results if r.get("url")]

    # Step 3: Retrieve content using Jina AI Reader
    results = await retrieve_urls(
        urls=urls,
        max_concurrent=10,
        timeout=timeout,
        search_metadata=search_results,
    )

    return results


global_llm_client: GeneralOpenAIClient | None = None
global_reranker: GeneralReranker | None = None


async def get_default_llms():
    global global_llm_client, global_reranker
    if not global_llm_client:
        llm_mgr = get_llm_mgr(VllmConfig())
        global_llm_client = await llm_mgr.get_openai_client(
            max_tokens=4_096,
        )
    if not global_reranker:
        global_reranker = await get_reranker()
    return global_llm_client, global_reranker


async def main():
    print(await reformulate_query("I'm using vllm to run Qwen/Qwen3-4B model, now I'm sending it a string prompt, how can I use python libraries calculate how many tokens in my string prompt before I send it over?"))

if __name__ == "__main__":
    asyncio.run(main())
