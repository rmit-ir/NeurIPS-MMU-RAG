from typing import List


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
    # TODO: Implement chunking logic
    # - Create sliding window chunks with overlap
    # - Handle edge cases for short documents
    # - Ensure chunks don't exceed size limits
    # - Preserve context across chunk boundaries
    pass