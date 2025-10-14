import asyncio
from typing import List

from transformers import AutoTokenizer
from tools.logging_utils import get_logger
from tools.web_search import SearchResult

logger = get_logger("doc_truncation")

global_qwen3_tokenizer = None


def calc_tokens_str(text: str) -> int:
    global global_qwen3_tokenizer
    if not global_qwen3_tokenizer:
        global_qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    tokens = global_qwen3_tokenizer.encode(text)
    return len(tokens)


def calc_tokens(doc: SearchResult) -> int:
    return calc_tokens_str(doc.text)


def truncate_docs(docs: List[SearchResult], tokens_threshold: int) -> List[SearchResult]:
    """
    Truncate a list of SearchResult documents based on a token count threshold.

    Args:
        docs: List of SearchResult objects
        tokens_threshold: Maximum number of tokens to include
    Returns:
        Truncated list of SearchResult objects
    """
    if not docs:
        return []

    truncated_docs = []
    total_words = 0

    for doc in docs:
        # Count words in the document text
        word_count = calc_tokens(doc)

        # Check if adding this document would exceed the threshold
        if total_words + word_count > tokens_threshold:
            logger.debug("Token threshold reached",
                         total_words=total_words,
                         threshold=tokens_threshold,
                         docs_included=len(truncated_docs),
                         docs_total=len(docs))
            break

        # Add document and update word count
        truncated_docs.append(doc)
        total_words += word_count

    if len(truncated_docs) == 0 and len(docs) > 0:
        # Ensure at least one document is included
        doc_0 = docs[0]
        doc_0_txt_truncated_list = doc_0.text.split()[:tokens_threshold]
        doc_0_txt_truncated = " ".join(doc_0_txt_truncated_list)
        doc_0 = doc_0._replace(text=doc_0_txt_truncated)
        truncated_docs.append(doc_0)
        total_words = len(doc_0_txt_truncated_list)
        logger.debug("No documents fit within the threshold; truncating the first document",
                     total_words=total_words,
                     threshold=tokens_threshold)

    logger.info("Documents truncated",
                original_count=len(docs),
                truncated_count=len(truncated_docs),
                total_words=total_words)
    return truncated_docs


async def atruncate_docs(docs: List[SearchResult], tokens_threshold: int) -> List[SearchResult]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, truncate_docs, docs, tokens_threshold)


def update_docs_sids(docs: List[SearchResult], base_count: int = 0) -> List[SearchResult]:
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
        docs[idx] = doc._replace(sid=str(idx + 1 + base_count))

    logger.info("Document SIDs updated", total_docs=len(docs))
    return docs
