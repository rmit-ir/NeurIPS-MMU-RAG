#!/usr/bin/env python
"""
Simple script to run remote RAG systems via API.

Usage:
    export REMOTE_API_KEY="find this from your browser https://search.chai-research.au/ login key"
    uv run scripts/run_remote.py mmu_rag_vanilla \
        --topics-file ./data/past_topics/organizers_outputs/t2t_val.jsonl \
        --output-dir ./data/past_topics/inhouse_outputs/

Input topics file must be in JSONL format with 'iid' and 'query' fields.
Output will be saved in JSONL format with evaluation results.
The system_key parameter is passed to the remote API as server_key.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List
from urllib.parse import quote_plus

import aiohttp
import jsonlines
from systems.rag_interface import EvaluateResponse, EvaluateRequest
from tools.loaders import load_topics
from tools.logging_utils import get_logger
from tools.retry_utils import retry

logger = get_logger('run_remote')

# Remote API configuration
REMOTE_API_BASE_URL = "https://ase-server-api.rankun.org/api/search/ai-overview"


@retry(max_retries=4, retry_on=(aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError))
async def make_remote_request(session: aiohttp.ClientSession, query: str,
                              server_key: str, api_key: str) -> dict:
    """
    Make a request to the remote API.

    Args:
        session: aiohttp session
        query: Search query
        server_key: System key to pass to API
        api_key: Bearer token for authorization

    Returns:
        API response as dict
    """
    # URL encode the query
    encoded_query = quote_plus(query)

    # Build URL with query parameters
    url = f"{REMOTE_API_BASE_URL}?query={encoded_query}&stream=false&server_key={server_key}"

    headers = {
        'accept': 'application/json',
        'authorization': f'Bearer {api_key}'
    }

    async with session.get(url, headers=headers) as response:
        result = await response.json()
        if response.status != 200:
            logger.error("Remote API error",
                         status=response.status, response=result)
        response.raise_for_status()
        return result


async def process_topic_remote(session: aiohttp.ClientSession, request: EvaluateRequest,
                               server_key: str, api_key: str) -> EvaluateResponse | None:
    """
    Process single topic through remote RAG API.

    Args:
        session: aiohttp session
        request: EvaluateRequest object
        server_key: System key for the remote API
        api_key: Bearer token for authorization

    Returns:
        EvaluateResponse object
    """
    try:
        # Make request to remote API
        response_data = await make_remote_request(session, request.query, server_key, api_key)

        # Extract response data from OpenAI-like format
        choices = response_data.get('choices', [])
        if choices and len(choices) > 0:
            message = choices[0].get('message', {})
            generated_response = message.get('content', '')
        else:
            generated_response = ''

        # TODO: add citations and contexts
        citations = []
        contexts = []

        return EvaluateResponse(
            query_id=request.iid,
            generated_response=generated_response,
            citations=citations,
            contexts=contexts
        )

    except Exception as e:
        logger.error("Error processing topic via remote API",
                     topic_id=request.iid, error=str(e))


def get_output_path(output_dir: str, input_file: str, server_key: str) -> Path:
    input_filename = f'output_remote_{server_key}_{Path(input_file).name}'
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if output_path.is_dir():
        output_path = output_path / input_filename

    return output_path


def append_result_to_file(result: EvaluateResponse, output_path: Path):
    try:
        with jsonlines.open(output_path, 'a') as writer:
            writer.write({
                'query_id': result.query_id,
                'generated_response': result.generated_response,
                'citations': result.citations,
                'contexts': result.contexts
            })
    except Exception as e:
        logger.error("Error appending result to file",
                     output_path=output_path,
                     query_id=result.query_id,
                     error=str(e))
        raise


async def run_remote_parallel(topics: List[EvaluateRequest], server_key: str,
                              api_key: str, parallel: int, output_path: Path) -> List[EvaluateResponse]:
    """
    Run remote RAG system on topics with parallel processing using Queue.
    Results are saved incrementally as each query completes.

    Args:
        topics: List of EvaluateRequest objects
        server_key: System key for the remote API
        api_key: Bearer token for authorization
        parallel: Number of parallel requests
        output_path: Path to output file for incremental saving

    Returns:
        List of EvaluateResponse objects
    """
    logger.info(
        "Starting parallel processing with remote API",
        topics_count=len(topics),
        parallel_workers=parallel,
        server_key=server_key,
        output_path=output_path
    )

    # Clear/create target file at start
    try:
        with jsonlines.open(output_path, 'w') as writer:
            pass  # Just create/clear the file
        logger.info("Target file cleared/created", output_path=output_path)
    except Exception as e:
        logger.error("Error creating/clearing target file",
                     output_path=output_path, error=str(e))
        raise

    # Setup queues and results
    work_queue = asyncio.Queue()
    results = []
    total_num = len(topics)
    results_lock = asyncio.Lock()

    # Add all work to queue
    for request in topics:
        await work_queue.put(request)

    # Create aiohttp session with timeout
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
    connector = aiohttp.TCPConnector(limit=parallel * 2)  # Connection pool

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def worker():
            """Process requests from queue and save results incrementally."""
            while True:
                try:
                    request = await work_queue.get()
                    result = await process_topic_remote(session, request, server_key, api_key)
                    if not result:
                        logger.error("No result returned for topic",
                                     topic_id=request.iid)
                        work_queue.task_done()
                        continue

                    # Append result immediately to file with lock protection
                    async with results_lock:
                        try:
                            append_result_to_file(result, output_path)
                            results.append(result)
                            logger.info(
                                "Topic processed and saved",
                                topic_id=request.iid,
                                finished=len(results),
                                total=total_num,
                                progress=len(results)/total_num
                            )
                        except Exception as save_error:
                            logger.error("Error saving result to file",
                                         topic_id=request.iid,
                                         error=str(save_error))
                            # Still add to memory for final count
                            results.append(result)

                    work_queue.task_done()

                except Exception as e:
                    logger.error("Worker error processing topic",
                                 topic_id=request.iid if 'request' in locals() else 'unknown',
                                 error=str(e))

                    work_queue.task_done()

        # Start workers
        workers = [asyncio.create_task(worker()) for _ in range(parallel)]

        try:
            # Wait for all work to complete
            await work_queue.join()
        except KeyboardInterrupt:
            logger.warning("Ctrl-C: Graceful shutdown initiated by user")
        finally:
            # Cancel workers
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run remote RAG systems via API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required environment variables: REMOTE_API_KEY

Examples:
  python scripts/run_remote.py mmu_rag_vanilla \\
    --topics-file data/topics.jsonl --output-dir data/runs/

  python scripts/run_remote.py perplexity_research \\
    --topics-file data/topics.jsonl --output-dir data/runs/ --parallel 5
        """
    )

    parser.add_argument(
        'server_key',
        help='System key to pass to remote API (e.g., mmu_rag_vanilla, perplexity_research)'
    )

    parser.add_argument(
        '--topics-file',
        required=True,
        help='Input JSONL file with topics (must have "iid" and "query" fields)'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for results (will create if needed)'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel requests (default: 1)'
    )

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv('REMOTE_API_KEY')
    if not api_key:
        logger.error("REMOTE_API_KEY environment variable is required")
        sys.exit(1)

    # Validate parallel parameter
    if args.parallel < 1:
        logger.error("Parallel parameter must be at least 1")
        sys.exit(1)

    logger.info("Starting remote RAG processing",
                server_key=args.server_key, parallel=args.parallel)

    # Load topics
    topics = load_topics(args.topics_file)

    # Get output path
    output_path = get_output_path(
        args.output_dir, args.topics_file, args.server_key)

    # Run remote system with incremental saving
    try:
        results = asyncio.run(
            run_remote_parallel(topics, args.server_key,
                                api_key, args.parallel, output_path)
        )

        logger.info("Processing completed successfully",
                    results_count=len(results),
                    output_path=output_path)

    except KeyboardInterrupt:
        logger.warning("Ctrl-C: Processing interrupted by user")
        logger.info("Partial results saved to file", output_path=output_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Error during execution", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
