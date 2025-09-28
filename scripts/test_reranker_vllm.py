# Modified from vLLM official Github repo
# uv run scripts/test_reranker_vllm.py -i data/past_topics/runs/topics.rag24.test.retrieval.jsonl
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import argparse
from typing import Any, Dict, List, Sequence, TypedDict, Union
import jsonlines

from tools.logging_utils import get_logger
from tools.reranker_vllm import get_reranker
from tools.web_search import SearchResult

logger = get_logger('test_reranker_vllm')


def dict_to_search_result(doc: Dict[str, Any]) -> SearchResult:
    """Convert a dictionary document to SearchResult namedtuple."""
    return SearchResult(
        text=doc.get('text', ''),
        id=doc.get('id', ''),
        sid=doc.get('sid', ''),
        dump=doc.get('dump', ''),
        url=doc.get('url', ''),
        date=doc.get('date', ''),
        file_path=doc.get('file_path', ''),
        language=doc.get('language', ''),
        language_score=doc.get('language_score', 0.0),
        token_count=doc.get('token_count', 0)
    )


class OutputRecord(TypedDict):
    iid: str
    query: str
    docs: Sequence[Dict[str, Any]]


async def main() -> None:
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--run-file',
        '-i',
        required=True,
        help='Input JSONL run file under data/past_topics/runs with docs (must have "iid", "query" and "docs" fields)'
    )

    parser.add_argument(
        '--num-docs',
        '-n',
        type=int,
        default=None,
        help='Number of documents to rerank per query (default: all)'
    )

    args = parser.parse_args()

    # Load run
    topics: List[OutputRecord] = []
    with jsonlines.open(args.run_file, 'r') as reader:
        for line_num, topic in enumerate(reader, 1):
            topics.append(OutputRecord(**topic))

    logger.info("Topics loaded successfully", topics_count=len(
        topics), run_file=args.run_file)

    reranker = get_reranker(drop_irrelevant_threshold=0.5)

    # For each topic, rerank the documents, finally save to same filename .rerank.jsonl
    for topic in topics:
        docs = topic['docs']

        # Filter out documents without text
        valid_docs = [doc for doc in docs if isinstance(
            doc, dict) and doc.get('text', '').strip()]

        if not valid_docs:
            logger.warning("No valid documents to rerank",
                           topic_id=topic['iid'])
            continue

        # Apply num_docs limit if specified
        if args.num_docs is not None:
            valid_docs = valid_docs[:args.num_docs]

        query = topic.get('query', '')
        search_results = [dict_to_search_result(doc) for doc in valid_docs]
        ranked_results = await reranker.rerank(query, search_results)

        # Convert SearchResultRanked back to dict format for output
        topic['docs'] = [dict(result) for result in ranked_results]
    # Save to same filename .rerank.jsonl
    output_file = args.run_file.replace('retrieval.jsonl', 'rerank.jsonl')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(topics)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
