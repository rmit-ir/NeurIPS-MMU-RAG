from typing import List


def build_index(chunks: List[str], index_path: str):
    """
    Build and save a FAISS index from document chunks.
    
    Args:
        chunks: List of text chunks to index
        index_path: Path where the index should be saved
    """
    # TODO: Implement indexing logic
    # - Generate embeddings for each chunk using sentence transformers
    # - Create FAISS index with appropriate configuration
    # - Save index to specified path for later retrieval
    # - Consider index type (flat, IVF, etc.) based on corpus size
    pass