import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from openai.types.chat import ChatCompletionMessageParam
from structlog import BoundLogger
from systems.rag_interface import RunStreamingResponse
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.reranker_vllm import GeneralReranker, get_reranker
from tools.str_utils import extract_tag_val
from tools.web_search import SearchResult, search_clueweb


def system_message(enable_think: bool) -> str:
    # Create a simple RAG prompt
    return f"""You are a knowledgeable AI search assistant built by the RMIT IR team.

Your search engine has returned a list of relevant webpages based on the user's question, listed below in <search-results> tags. These webpages are your knowledge.

The next user message is the full user question, and you need to explain and answer the question based on the search results. Do not make up answers that are not supported by the search results. If the search results do not have the necessary information for you to answer the search question, say you don't have enough information for the question.

Try to provide a balanced view for controversial topics.

Tailor the complexity of your response to the user question, use simpler bullet points for simple questions, and sections for more detailed explanations for complex topics or rich content.

Do not answer to greetings or chat with the user, always reply in English.

You should refer to the search results in your final response as much as possible, append [ID] after each sentence to point to the specific search result. e.g., "This sentence is referring to information in search result 1 [1].".

Current time at UTC+00:00 timezone: {datetime.now(timezone.utc)}
Search results knowledge cutoff: December 2024
{'/nothink' if enable_think is not True else ''}

"""


def build_llm_messages(results: str | list[SearchResult], query: str, enable_think: bool) -> List[ChatCompletionMessageParam]:
    context = ""
    if isinstance(results, list):
        if len(results) > 0 and isinstance(results[0], SearchResult):
            context = build_to_context(results)
        else:
            context = "<search-results>Empty results</search-results>"
    elif isinstance(results, str):
        context = results

    sys_msg = system_message(enable_think) + context
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]


def build_to_context(results: list[SearchResult]) -> str:
    context = "<search-results>"
    context += "\n".join([f"""
Webpage ID=[{r.sid}] URL=[{r.url}] Date=[{r.date}]:

{r.text}""" for r in results])
    context += "</search-results>"
    return context


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
    system_prompt = f"""You will receive a question from a user and you need interpret what the question is actually asking about and come up with 2 to {num_qvs} Google search queries to answer that question.

Try express the same question in different ways, use different techniques, query expansion, query relaxation, query segmentation, use different synonyms, use reasonable guess and different keywords to reach different aspects.

Try to provide a balanced view for controversial topics.

To comply with the format, put your query variants enclosed in queries xml markup:

<queries>
query variant 1
query variant 2
...
</queries>

Put each query in a line, do not add any prefix on each query, only provide the query themselves.
{'' if enable_think else '/nothink'}"""
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


async def reformulate_query(query: str, preset_llm: Optional[GeneralOpenAIClient] = None) -> str:
    """Reformulate the query to improve search results"""
    if not preset_llm:
        llm, _ = await get_default_llms()
    else:
        llm = preset_llm

    system_prompt = """You will receive a question from a user and you need interpret what the question is actually asking about and come up with a better Google search query to answer that question. Only provide the reformulated query, do not add any prefix or suffix."""
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
    """Search with query variants and merge results"""
    queries = await generate_qvs(query, num_qvs, enable_think, logger=logger, preset_llm=preset_llm)
    queries = set([query, *queries])
    queries_docs = await asyncio.gather(*[
        search_clueweb(query=q, k=10, cw22_a=cw22_a) for q in queries])

    # Deduplicate and put into a list
    all_results: List[SearchResult] = []
    all_docs_id_set = set()
    for docs in queries_docs:
        for r in docs:
            if isinstance(r, SearchResult) and r.id not in all_docs_id_set:
                all_results.append(r)
                all_docs_id_set.add(r.id)

    logger.info("Search with query variants completed",
                original_query=query,
                num_variants=len(queries),
                all_results=len(all_results))

    return list(queries), all_results


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
