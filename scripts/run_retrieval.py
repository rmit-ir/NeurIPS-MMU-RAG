#!/usr/bin/env python
"""
Script to run web search retrieval on JSONL topic files.

Usage:
    uv run scripts/run_retrieval.py --topics-file data/past_topics/processed/topics.rag24.test.n50.jsonl --num-docs 200

Input topics file must be in JSONL format with 'iid' and 'query' fields.
Output will be saved in JSONL format with search results in a 'docs' field.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, TypedDict

import jsonlines
from tools.logging_utils import get_logger
from tools.web_search import search_fineweb_sync, SearchResult
from systems.rag_interface import EvaluateRequest

logger = get_logger('run_retrieval')


class OutputRecord(TypedDict):
    iid: str
    query: str
    docs: List[Dict[str, Any]]


def run_search_for_topic(topic: EvaluateRequest, num_docs: int) -> OutputRecord:
    """
    Run web search for a single topic.

    Args:
        topic: Dictionary with 'iid' and 'query' fields
        num_docs: Number of documents to retrieve

    Returns:
        Dictionary with topic info and search results
    """
    try:
        # Run search
        search_results = search_fineweb_sync(topic.query, k=num_docs)

        # Convert search results to serializable format
        docs = [r._asdict()
                for r in search_results if isinstance(r, SearchResult)]
        output_record = OutputRecord(
            iid=topic.iid, query=topic.query, docs=docs)

        logger.info("Search completed successfully",
                    topic_id=topic.iid, docs_count=len(docs))
        return output_record

    except Exception as e:
        logger.error("Error processing topic",
                     topic_id=topic.iid, error=str(e))
        return {
            'iid': topic.iid,
            'query': topic.query,
            'docs': [{'error': f"Search failed: {str(e)}", 'value': None}]
        }


def save_results(results: List[OutputRecord], output_file: str):
    """
    Save results to JSONL file using jsonlines library.
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(output_path, 'w') as writer:
            for result in results:
                writer.write(result)

        logger.info("Results saved successfully",
                    output_path=output_path, results_count=len(results))

    except Exception as e:
        logger.error("Error saving results",
                     output_path=output_file, error=str(e))
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run web search retrieval on JSONL topic files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_retrieval.py --topics-file data/past_topics/processed/topics.rag24.test.n50.jsonl
  
  python scripts/run_retrieval.py --topics-file data/past_topics/processed/topics.rag24.test.n50.jsonl --num-docs 100
        """
    )

    parser.add_argument(
        '--topics-file',
        '-i',
        required=True,
        help='Input JSONL file with topics (must have "iid" and "query" fields)'
    )

    parser.add_argument(
        '--num-docs',
        '-n',
        type=int,
        default=200,
        help='Number of documents to retrieve per query (default: 200)'
    )

    args = parser.parse_args()

    # Load topics
    topics: List[EvaluateRequest] = []
    with jsonlines.open(args.topics_file, 'r') as reader:
        for line_num, topic in enumerate(reader, 1):
            topics.append(EvaluateRequest(**topic))

    logger.info("Topics loaded successfully", topics_count=len(
        topics), topics_file=args.topics_file)

    # Process topics
    results = []
    for i, topic in enumerate(topics, 1):
        logger.info("Processing topic", topic_id=topic.iid,
                    progress=f"{i}/{len(topics)}")
        result = run_search_for_topic(topic, args.num_docs)
        results.append(result)

    # Determine output file path
    input_path = Path(args.topics_file)
    output_dir = input_path.parent.parent / "runs"
    output_file = output_dir / f"{input_path.stem}.retrieval.jsonl"

    # Save results
    save_results(results, str(output_file))

    logger.info("Processing completed successfully",
                results_count=len(results))


if __name__ == "__main__":
    main()
