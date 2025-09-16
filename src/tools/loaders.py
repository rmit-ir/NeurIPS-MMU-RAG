import sys
from typing import List

import jsonlines
from systems.rag_interface import EvaluateRequest
from tools.logging_utils import get_logger

logger = get_logger('loaders')


def load_topics(topics_file: str) -> List[EvaluateRequest]:
    """
    Load topics from JSONL file using jsonlines library.
    """
    topics: List[EvaluateRequest] = []

    try:
        with jsonlines.open(topics_file, 'r') as reader:
            for line_num, topic in enumerate(reader, 1):
                # Check required fields
                if 'iid' not in topic or 'query' not in topic:
                    logger.warning(
                        "Line missing required fields, skipping",
                        line_num=line_num,
                        missing_fields=[field for field in [
                            'iid', 'query'] if field not in topic]
                    )
                    continue

                topics.append(EvaluateRequest(
                    iid=topic['iid'],
                    query=topic['query']
                ))

    except FileNotFoundError:
        logger.error("Topics file not found", topics_file=topics_file)
        sys.exit(1)
    except Exception as e:
        logger.error("Error reading topics file",
                     topics_file=topics_file, error=str(e))
        sys.exit(1)

    if not topics:
        logger.error("No valid topics found in file", topics_file=topics_file)
        sys.exit(1)

    logger.info("Topics loaded successfully",
                topics_count=len(topics), topics_file=topics_file)
    return topics
