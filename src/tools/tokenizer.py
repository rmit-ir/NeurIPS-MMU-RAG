from typing import List, Optional
import tiktoken


def tokenize(text: str, model: str = "gpt-3.5-turbo") -> List[int]:
    """
    Tokenize text using tiktoken for accurate token counting.

    Args:
        text: Input text to tokenize
        model: Model name for tokenizer (affects tokenization rules)

    Returns:
        List of token IDs
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode(text)
    except KeyError:
        # Fallback to cl100k_base encoding (used by GPT-3.5/4)
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode(text)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in text.

    Args:
        text: Input text to count tokens for
        model: Model name for tokenizer

    Returns:
        Number of tokens
    """
    return len(tokenize(text, model))


def truncate_text_to_tokens(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        model: Model name for tokenizer

    Returns:
        Truncated text that fits within token limit
    """
    if count_tokens(text, model) <= max_tokens:
        return text

    # Binary search to find the right truncation point
    text_bytes = text.encode('utf-8')
    low, high = 0, len(text_bytes)

    while low < high:
        mid = (low + high + 1) // 2
        truncated_text = text_bytes[:mid].decode('utf-8', errors='ignore')

        if count_tokens(truncated_text, model) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    return text_bytes[:low].decode('utf-8', errors='ignore')