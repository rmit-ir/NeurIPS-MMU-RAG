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
            logger.info("Searching FineWeb", query=query, max_results=max_results)
            search_results = await fineweb_search(query, k=max_results)
            logger.info("FineWeb search completed", result_count=len(search_results) if search_results else 0)

            if not search_results:
                # Fallback to ClueWeb if FineWeb fails
                logger.info("FineWeb search returned no results, trying ClueWeb")
                try:
                    search_results = await clueweb_search(query, k=max_results)
                    logger.info("ClueWeb search completed", result_count=len(search_results) if search_results else 0)
                except Exception as e:
                    logger.warning("ClueWeb search failed", error=str(e))
                    search_results = []

            if not search_results:
                logger.warning("No search results found")
                return "", []

            # Debug: Log what we got
            logger.info("Processing search results", total_results=len(search_results))
            for i, result in enumerate(search_results[:3]):  # Log first 3 results
                if hasattr(result, 'text'):
                    logger.info(f"Result {i+1} preview", text=result.text[:100])
                else:
                    logger.info(f"Result {i+1} type", type=type(result).__name__)

            # Extract and process content with citations

            # Extract and process content with citations
            context_parts = []
            citations = []
            total_tokens = 0

            for i, result in enumerate(search_results):
                # Skip SearchError objects
                if not hasattr(result, 'text'):
                    logger.warning(f"Skipping invalid result at index {i}", type=type(result).__name__)
                    continue

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

    def _extract_citation_info(self, result, source_number: int) -> Dict[str, Any]:
        """Extract citation information from a search result."""
        citation = {
            "source_number": source_number,
            "title": None,
            "url": None,
            "domain": None,
            "date": None,
            "text_preview": None
        }

        # Handle SearchResult named tuples
        if hasattr(result, 'url') and result.url:
            citation["url"] = result.url
            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(result.url).netloc
                if domain.startswith('www.'):
                    domain = domain[4:]
                citation["domain"] = domain
            except:
                pass

        if hasattr(result, 'text') and result.text:
            text = result.text.strip()
            citation["text_preview"] = text[:200] + "..." if len(text) > 200 else text

            # Try to extract title from first line
            first_line = text.split('\n')[0].strip()
            if len(first_line) < 100 and (first_line.endswith('?') or first_line.endswith('.')):
                citation["title"] = first_line
            else:
                # Use domain as fallback title
                if citation["domain"]:
                    citation["title"] = f"Article from {citation['domain']}"

        if hasattr(result, 'date') and result.date:
            citation["date"] = result.date

        # Fallback for dictionary-like results
        elif isinstance(result, dict):
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

    def _extract_content_from_result(self, result) -> str:
        """Extract readable content from search result."""
        # Handle SearchResult named tuples
        if hasattr(result, 'text') and result.text:
            content = str(result.text).strip()
            if len(content) > 100:  # Only use substantial content
                return content

        # Fallback for dictionary-like results
        if isinstance(result, dict):
            content_fields = ['text', 'content', 'body', 'description', 'snippet']
            for field in content_fields:
                if field in result and result[field]:
                    content = str(result[field]).strip()
                    if len(content) > 100:
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


async def test_retriever():
    """Test the DocumentRetriever with real search queries."""
    print("Testing DocumentRetriever with real search...")

    retriever = DocumentRetriever(max_context_tokens=2000)

    # Test queries
    test_queries = [
        "What is machine learning?",
        "Latest developments in artificial intelligence 2024",
        "Climate change impacts on biodiversity"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print(f"{'='*60}")

        try:
            # Test search_and_retrieve method
            print("\n--- Testing search_and_retrieve ---")
            context, citations = await retriever.search_and_retrieve(query, max_results=3)

            if context:
                print(f"Retrieved context ({len(context)} characters):")
                print(context[:500] + "..." if len(context) > 500 else context)

                print(f"\nCitations ({len(citations)}):")
                for citation in citations:
                    print(f"  [{citation['source_number']}] {citation.get('title', 'No title')} - {citation.get('url', 'No URL')}")
            else:
                print("No context retrieved")

            # Test retrieve function
            print("\n--- Testing retrieve function ---")
            chunks, citations = await retrieve(query, top_k=3)

            print(f"Retrieved {len(chunks)} chunks:")
            for j, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"  Chunk {j+1}: {chunk[:200]}..." if len(chunk) > 200 else f"  Chunk {j+1}: {chunk}")

            print(f"Citations: {len(citations)} items")

        except Exception as e:
            print(f"Error testing query '{query}': {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Retriever testing completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(test_retriever())