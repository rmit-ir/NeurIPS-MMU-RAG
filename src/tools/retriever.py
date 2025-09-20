from typing import List, Dict, Any, Optional, Tuple
import asyncio
import aiohttp
from tools.chunker import chunk_text
from tools.tokenizer import count_tokens, truncate_text_to_tokens
from tools.web_search import fineweb_search, clueweb_search
from tools.logging_utils import get_logger

logger = get_logger('retriever')


class DocumentRetriever:
    """Document retriever that searches web sources and returns relevant chunks with citations."""

    def __init__(self, max_context_tokens: int = 3000):
        """
        Initialize the document retriever.

        Args:
            max_context_tokens: Maximum tokens to use for context (leaving room for query and response)
        """
        self.max_context_tokens = max_context_tokens

    async def search_and_retrieve(self, query: str, max_results: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Search for relevant documents and return formatted context with citations.

        Args:
            query: Search query
            max_results: Maximum number of search results to process

        Returns:
            Tuple of (formatted_context, citations_list)
        """
        try:
            # Search using FineWeb (more comprehensive)
            search_results = await fineweb_search(query, k=max_results)

            if not search_results:
                # Fallback to ClueWeb if FineWeb fails
                logger.info("FineWeb search returned no results, trying ClueWeb")
                search_results = await clueweb_search(query, k=max_results)

            if not search_results:
                logger.warning("No search results found")
                return "", []

            # Extract and process content with citations
            context_parts = []
            citations = []
            total_tokens = 0

            for i, result in enumerate(search_results):
                content = self._extract_content_from_result(result)
                if content:
                    # Create citation info for this result
                    citation_info = self._extract_citation_info(result, i + 1)
                    citations.append(citation_info)

                    # Chunk the content
                    chunks = chunk_text(content, chunk_size=1000, overlap=100)

                    # Score and select best chunks
                    relevant_chunks = self._select_relevant_chunks(query, chunks)

                    for chunk in relevant_chunks:
                        chunk_tokens = count_tokens(chunk)

                        # Check if adding this chunk would exceed token limit
                        if total_tokens + chunk_tokens > self.max_context_tokens:
                            break

                        # Format chunk with citation reference
                        cited_chunk = f"[Source {i+1}] {chunk}"
                        context_parts.append(cited_chunk)
                        total_tokens += chunk_tokens

                    if total_tokens >= self.max_context_tokens:
                        break

            # Format the context
            if context_parts:
                context = "\n\n".join(context_parts)
                return context, citations
            else:
                return "", []

        except Exception as e:
            logger.error("Error in search_and_retrieve", error=str(e))
            return "", []

    def _extract_citation_info(self, result: Dict[str, Any], source_number: int) -> Dict[str, Any]:
        """Extract citation information from a search result."""
        citation = {
            "source_number": source_number,
            "title": None,
            "url": None,
            "domain": None,
            "date": None,
            "text_preview": None
        }

        # Extract URL
        if "url" in result and result["url"]:
            citation["url"] = result["url"]
            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(result["url"]).netloc
                if domain.startswith('www.'):
                    domain = domain[4:]
                citation["domain"] = domain
            except:
                pass

        # Extract title from text (first line or first sentence)
        if "text" in result and result["text"]:
            text = result["text"].strip()
            citation["text_preview"] = text[:200] + "..." if len(text) > 200 else text

            # Try to extract title from first line
            first_line = text.split('\n')[0].strip()
            if len(first_line) < 100 and first_line.endswith('?') or first_line.endswith('.'):
                citation["title"] = first_line
            else:
                # Use domain as fallback title
                if citation["domain"]:
                    citation["title"] = f"Article from {citation['domain']}"

        # Extract date
        if "date" in result and result["date"]:
            citation["date"] = result["date"]

        return citation

    def _extract_content_from_result(self, result: Dict[str, Any]) -> str:
        """Extract readable content from search result."""
        # Try different fields that might contain the content
        content_fields = ['text', 'content', 'body', 'description', 'snippet']

        for field in content_fields:
            if field in result and result[field]:
                content = str(result[field]).strip()
                if len(content) > 100:  # Only use substantial content
                    return content

        return ""

    def _select_relevant_chunks(self, query: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
        """Select the most relevant chunks based on query similarity."""
        if not chunks:
            return []

        # Simple relevance scoring based on keyword matching
        query_words = set(query.lower().split())
        scored_chunks = []

        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            scored_chunks.append((score, chunk))

        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:max_chunks]]


async def retrieve(query: str, index_path: str = "", top_k: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Retrieve the most relevant chunks for a given query with citations.

    Args:
        query: User query to search for
        index_path: Path to the saved FAISS index (unused for web search)
        top_k: Number of top chunks to retrieve

    Returns:
        Tuple of (chunks_list, citations_list)
    """
    retriever = DocumentRetriever()
    context, citations = await retriever.search_and_retrieve(query, max_results=top_k)

    if context:
        # Split context back into chunks for compatibility
        chunks = context.split("\n\n")
        return chunks, citations
    else:
        return [], []