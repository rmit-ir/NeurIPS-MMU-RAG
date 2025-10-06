from typing import List
from tools.logging_utils import get_logger
from tools.web_search import SearchResult

logger = get_logger("doc_truncation")


def truncate_docs(docs: List[SearchResult], words_threshold: int = 5000) -> List[SearchResult]:
    """
    Truncate a list of SearchResult documents based on a word count threshold.

    Args:
        docs: List of SearchResult objects
        words_threshold: Maximum number of words to include (default: 5000)

    Returns:
        Truncated list of SearchResult objects
    """
    if not docs:
        return []

    truncated_docs = []
    total_words = 0

    for doc in docs:
        # Count words in the document text
        word_count = len(doc.text.split())

        # Check if adding this document would exceed the threshold
        if total_words + word_count > words_threshold:
            logger.debug("Word threshold reached",
                         total_words=total_words,
                         threshold=words_threshold,
                         docs_included=len(truncated_docs),
                         docs_total=len(docs))
            break

        # Add document and update word count
        truncated_docs.append(doc)
        total_words += word_count

    logger.info("Documents truncated",
                original_count=len(docs),
                truncated_count=len(truncated_docs),
                total_words=total_words)
    return truncated_docs


def update_docs_sids(docs: List[SearchResult]) -> List[SearchResult]:
    """
    Update the 'sid' attribute of each SearchResult document to ensure uniqueness.

    Args:
        docs: List of SearchResult objects

    Returns:
        List of SearchResult objects with updated 'sid' attributes
    """
    if not docs:
        return []

    for idx, doc in enumerate(docs):
        docs[idx] = doc._replace(sid=str(idx + 1))

    logger.info("Document SIDs updated", total_docs=len(docs))
    return docs
