from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.

    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks with specified overlap
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position for this chunk
        end = start + chunk_size

        # If we're not at the end, try to find a good break point
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            break_chars = ['. ', '! ', '? ', '\n\n', '\n']
            best_break = end

            for break_char in break_chars:
                last_break = text.rfind(break_char, start, end)
                if last_break != -1 and last_break > end - 100:
                    best_break = last_break + len(break_char)
                    break

            end = best_break

        # Extract the chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap

        # Ensure we don't get stuck
        if start >= len(text):
            break
        elif start <= start - chunk_size + overlap:
            # If overlap is too large, move forward by at least 1
            start = start - chunk_size + overlap + 1

    return chunks


def chunk_tokens(tokens: List[int], size: int, overlap: int) -> List[List[int]]:
    """
    Split token sequences into overlapping chunks for processing.

    Args:
        tokens: List of token IDs to chunk
        size: Maximum size of each chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of token chunks with specified overlap
    """
    if not tokens:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        # Calculate end position for this chunk
        end = min(start + size, len(tokens))

        # Extract the chunk
        chunk = tokens[start:end]
        chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap

        # Ensure we don't get stuck
        if start >= len(tokens):
            break
        elif start <= start - size + overlap:
            # If overlap is too large, move forward by at least 1
            start = start - size + overlap + 1

    return chunks