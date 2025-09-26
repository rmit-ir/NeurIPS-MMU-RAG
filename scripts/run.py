#!/usr/bin/env python
"""
Simple script to run RAG systems using their export names.

Usage:
    python scripts/run.py systems.commercial.azure_o3_research.AzureO3ResearchRAG \
        --topics-file data/past_topics/trec_rag_2025_queries.jsonl \
        --output-dir data/runs/ \
        --parallel 3

Input topics file must be in JSONL format with 'iid' and 'query' fields.
Output will be saved in JSONL format with evaluation results.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

import jsonlines
from systems.rag_interface import EvaluateResponse, RAGInterface, EvaluateRequest
from tools.classes import load_system_class, unknown_args_to_dict
from tools.loaders import load_topics
from tools.logging_utils import get_logger

logger = get_logger('run')


async def process_topic(system: RAGInterface, request: EvaluateRequest) -> EvaluateResponse:
    """
    Process single topic through RAG system.

    Args:
        system: RAG system instance
        request: EvaluateRequest object

    Returns:
        EvaluateResponse object
    """
    try:
        # Get response from system
        response = await system.evaluate(request)
        return response

    except Exception as e:
        logger.error("Error processing topic",
                     topic_id=request.iid, error=str(e))
        return EvaluateResponse(
            query_id=request.iid,
            generated_response=f"Error: {str(e)}",
            citations=[]
        )


async def run_system_parallel(system: RAGInterface, topics: List[EvaluateRequest],
                              parallel: int) -> List[EvaluateResponse]:
    """
    Run RAG system on topics with parallel processing using Queue.

    Args:
        system: RAG system instance
        topics: List of EvaluateRequest objects
        parallel: Number of parallel requests

    Returns:
        List of EvaluateResponse objects
    """
    logger.info(
        "Starting parallel processing",
        topics_count=len(topics),
        parallel_workers=parallel
    )

    # Setup queues
    work_queue = asyncio.Queue()
    results = []
    total_num = len(topics)

    # Add all work to queue
    for request in topics:
        await work_queue.put(request)

    async def worker():
        """Process requests from queue."""
        while True:
            try:
                request = await work_queue.get()
                result = await process_topic(system, request)
                results.append(result)
                logger.info(
                    "Topic processed successfully",
                    topic_id=request.iid,
                    finished=len(results),
                    total=total_num,
                    progress=len(results)/total_num
                )
                work_queue.task_done()
            except Exception as e:
                logger.error("Worker error processing topic",
                             topic_id=request.iid, error=str(e))
                # Add error result to maintain count
                error_result = EvaluateResponse(
                    query_id=request.iid,
                    generated_response=f"Error: {str(e)}",
                    citations=[]
                )
                results.append(error_result)
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


def save_results(results: List[EvaluateResponse], output_dir: str, input_file: str):
    """
    Save results to JSONL file using jsonlines library.
    """
    # Create output directory if needed
    input_filename = 'output_' + Path(input_file).name
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if output_path.is_dir():
        output_path = output_path / input_filename

    try:
        # Write results using jsonlines
        with jsonlines.open(output_path, 'w') as writer:
            for result in results:
                writer.write({
                    'query_id': result.query_id,
                    'generated_response': result.generated_response,
                    'citations': result.citations,
                    'contexts': result.contexts
                })

        logger.info("Results saved successfully",
                    output_path=output_path, results_count=len(results))

    except Exception as e:
        logger.error("Error saving results",
                     output_path=output_path, error=str(e))
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run RAG systems using their export names",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py AzureO3ResearchRAG \\
    --topics-file data/topics.jsonl --output-dir data/runs/

  python scripts/run.py PerplexityResearchRAG \\
    --topics-file data/topics.jsonl --output-dir data/runs/ --parallel 5 --model sonar-pro

  python scripts/run.py PerplexityResearchRAG \\
    --topics-file data/topics.jsonl --output-dir data/runs/ --model sonar-deep-research
        """
    )

    parser.add_argument(
        'system_class',
        help='RAG system class name (e.g., AzureO3ResearchRAG)'
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

    # Parse known and unknown arguments
    args, unknown_args = parser.parse_known_args()
    system_kwargs_dict = unknown_args_to_dict(unknown_args)

    # Load system
    logger.info("Loading RAG system", system_class=args.system_class)
    system = load_system_class(args.system_class, system_kwargs_dict)
    logger.info("System loaded successfully", system_name=system.name)

    # Load topics
    topics = load_topics(args.topics_file)

    # Run system
    try:
        results = asyncio.run(
            run_system_parallel(system, topics, args.parallel)
        )

        # Save results
        save_results(results, args.output_dir, args.topics_file)

        logger.info("Processing completed successfully",
                    results_count=len(results))

    except KeyboardInterrupt:
        logger.warning("Ctrl-C: Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Error during execution", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
