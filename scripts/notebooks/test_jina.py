# /// script
# dependencies = [
#   "aiohttp",
#   "python-dotenv",
# ]
# ///

"""
Test Jina AI Reader API with Brave Search

This script:
1. Uses Brave Search API to find URLs for a query
2. Batch requests all URLs using Jina AI Reader API in Speed First mode
3. Gets Markdown format content
4. Saves results to ~/Downloads as JSONL with URL and timing metrics
"""

import asyncio
import aiohttp
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, TypedDict, Optional
from dotenv import load_dotenv


class BraveSearchResult(TypedDict):
    """Result structure for Brave Search API response

    Example:
    {
        "url": "https://example.com/article",
        "title": "Article Title",
        "description": "Brief description of the article content"
    }
    """
    url: str  # URL of the search result
    title: Optional[str]  # Title of the page or None
    description: Optional[str]  # Description snippet or None


class JinaResult(TypedDict):
    """Result structure for Jina AI Reader API response

    Example:
    {
        "url": "https://example.com/article",
        "status": 200,
        "success": true,
        "content": "# Article Title\n\nArticle content in markdown...",
        "title": "Article Title",
        "content_url": "https://example.com/article",
        "content_length": 1234,
        "error": null,
        "elapsed_seconds": 2.45,
        "response_headers": {
            "X-Cache-Status": "hit",
            "Content-Length": "5678"
        },
        "timestamp": "2024-01-26T12:34:56.789",
        "search_title": "Article Title from Search",
        "search_description": "Brief description from search results"
    }
    """
    url: str  # Original URL requested
    status: Optional[int]  # HTTP status code (200, 404, etc.) or None if request failed
    success: bool  # True if status 2xx
    content: Optional[str]  # Markdown content from page or None
    title: Optional[str]  # Page title or None
    content_url: Optional[str]  # Final URL after redirects or None
    content_length: Optional[int]  # Length of content in bytes or None
    error: Optional[str]  # Error message if failed or None
    elapsed_seconds: Optional[float]  # Request duration in seconds or None
    response_headers: Optional[Dict[str, str]]  # Response headers or None
    timestamp: str  # ISO format timestamp when request was made
    search_title: Optional[str]  # Title from Brave search result or None
    search_description: Optional[str]  # Description from Brave search result or None

# Load environment variables from .env file
load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

# Validate API keys
if not BRAVE_API_KEY:
    raise ValueError("BRAVE_API_KEY not found in environment")
if not JINA_API_KEY:
    raise ValueError("JINA_API_KEY not found in environment")

# API endpoints
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
JINA_READER_BASE = "https://r.jina.ai/"


async def brave_search(query: str, count: int = 10) -> List[BraveSearchResult]:
    """
    Search using Brave Search API

    Args:
        query: Search query
        count: Number of results to return (max 20)

    Returns:
        List of search result URLs and metadata
    """
    print(f"Searching Brave for: {query}")

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    params = {
        "q": query,
        "count": min(count, 20),  # Brave API max is 20
    }

    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.get(BRAVE_SEARCH_URL, headers=headers, params=params) as response:
            elapsed = time.time() - start_time

            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Brave Search API error: {response.status} - {error_text}")

            data = await response.json()

    results = []
    if "web" in data and "results" in data["web"]:
        for result in data["web"]["results"]:
            results.append({
                "url": result.get("url"),
                "title": result.get("title"),
                "description": result.get("description"),
            })

    print(f"Found {len(results)} URLs in {elapsed:.2f}s")
    return results


async def fetch_with_jina(session: aiohttp.ClientSession, url: str, index: int) -> JinaResult:
    """
    Fetch a single URL using Jina AI Reader API in Speed First mode

    Args:
        session: aiohttp session
        url: URL to fetch
        index: Index of the URL in the batch

    Returns:
        Dictionary with content, metadata, and timing
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "direct",  # Use direct engine for fast crawl
        "X-Return-Format": "markdown",
        "X-Token-Budget": "200000",  # Token budget for response
    }

    jina_url = f"{JINA_READER_BASE}{url}"
    print(f"  [{index+1}] Fetching: {url[:80]}...")

    # Base result structure
    result: JinaResult = {
        "url": url,
        "status": None,
        "success": False,
        "content": None,
        "title": None,
        "content_url": None,
        "content_length": None,
        "error": None,
        "elapsed_seconds": None,
        "response_headers": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "search_title": None,
        "search_description": None,
    }

    start_time = time.time()
    try:
        async with session.get(jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            result["elapsed_seconds"] = round(time.time() - start_time, 3)
            result["status"] = response.status

            # Extract response headers for metrics
            response_headers = {}
            for header in ["X-Cache-Status", "X-Response-Time", "X-Timeout-Used", "Content-Length"]:
                if header.lower() in response.headers:
                    response_headers[header] = response.headers[header.lower()]
            result["response_headers"] = response_headers if response_headers else None

            if 200 <= response.status < 300:
                # Parse JSON response
                response_data = await response.json()
                data = response_data.get("data", {})
                content = data.get("content") or None
                result.update({
                    "success": True,
                    "content": content,
                    "title": data.get("title") or None,
                    "content_url": data.get("url") or None,
                    "content_length": len(content) if content else None,
                })

                # Format output with bytes and word count
                if result['content_length']:
                    word_count = content.count(' ') + 1 if content and content.strip() else 0
                    length_str = f"{result['content_length']:,} bytes, {word_count:,} words"
                else:
                    length_str = "no content"
                print(f"  [{index+1}] ✓ Status {response.status}, {length_str}, {result['elapsed_seconds']:.2f}s")
            else:
                error_text = await response.text()
                result["error"] = f"HTTP {response.status}: {error_text[:200]}"
                print(f"  [{index+1}] ✗ Status {response.status}, {result['elapsed_seconds']:.2f}s")

    except asyncio.TimeoutError:
        result["elapsed_seconds"] = round(time.time() - start_time, 3)
        result["error"] = "Timeout"
        print(f"  [{index+1}] ✗ Timeout after {result['elapsed_seconds']:.2f}s")
    except Exception as e:
        result["elapsed_seconds"] = round(time.time() - start_time, 3)
        result["error"] = str(e)
        print(f"  [{index+1}] ✗ Error: {str(e)}")

    return result


async def batch_fetch_with_jina(urls: List[str], max_concurrent: int = 10) -> List[JinaResult]:
    """
    Batch fetch multiple URLs using Jina AI Reader API

    Args:
        urls: List of URLs to fetch
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of results with content and timing data
    """
    print(f"\nFetching {len(urls)} URLs with Jina AI Reader (Speed First mode)")
    print(f"Max concurrent requests: {max_concurrent}")
    print()

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_with_jina(session, url, i) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_time = sum(r["elapsed_seconds"] for r in results if r["elapsed_seconds"] is not None)
    avg_time = total_time / len(results) if results else 0

    print(f"\n{'='*80}")
    print(f"Batch Results: {successful} successful, {failed} failed")
    print(f"Total time: {total_time:.2f}s, Average: {avg_time:.2f}s per request")
    print(f"{'='*80}")

    return results


def save_to_jsonl(results: List[JinaResult], filename: str) -> str:
    """
    Save results to JSONL file in ~/Downloads

    Args:
        results: List of result dictionaries
        filename: Output filename (without path)

    Returns:
        Full path to saved file
    """
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(exist_ok=True)

    output_path = downloads_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(results)} results to: {output_path}")

    # Calculate file size
    file_size = output_path.stat().st_size
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{file_size / 1024:.2f} KB"
    print(f"  File size: {size_str}")

    return str(output_path)


async def main():
    """Main function to run the test"""

    # Configure test parameters
    SEARCH_QUERY = "latest AI research papers 2024"
    NUM_RESULTS = 10
    MAX_CONCURRENT = 20

    print("=" * 80)
    print("RetinaFace PyTorch batch inference multiple backends")
    print("=" * 80)
    print(f"Query: {SEARCH_QUERY}")
    print(f"Number of results: {NUM_RESULTS}")
    print("Mode: Speed First (15s timeout)")
    print()

    # Step 1: Search with Brave
    search_start = time.time()
    search_results = await brave_search(SEARCH_QUERY, count=NUM_RESULTS)
    search_elapsed = time.time() - search_start

    if not search_results:
        print("No search results found!")
        return

    # Extract URLs
    urls = [result["url"] for result in search_results if result.get("url")]

    print(f"\nURLs to fetch ({len(urls)} total):")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")

    # Step 2: Batch fetch with Jina
    jina_start = time.time()
    jina_results = await batch_fetch_with_jina(urls, max_concurrent=MAX_CONCURRENT)
    jina_elapsed = time.time() - jina_start

    total_elapsed = search_elapsed + jina_elapsed

    # Combine search metadata with Jina results
    for jina_result, search_result in zip(jina_results, search_results):
        jina_result["search_title"] = search_result.get("title")
        jina_result["search_description"] = search_result.get("description")

    # Step 3: Save to JSONL
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"jina_test_results_{timestamp}.jsonl"
    output_path = save_to_jsonl(jina_results, filename)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    print(f"\nTotal time: {total_elapsed:.2f}s")
    print(f"  - Brave Search: {search_elapsed:.2f}s ({search_elapsed/total_elapsed*100:.1f}%)")
    print(f"  - Jina Fetch: {jina_elapsed:.2f}s ({jina_elapsed/total_elapsed*100:.1f}%)")

    successful_results = [r for r in jina_results if r["success"] and r["content"]]

    if successful_results:
        content_sizes = [r["content_length"] for r in successful_results if r["content_length"] is not None]
        if content_sizes:
            print("Content sizes:")
            print(f"  Min: {min(content_sizes):,} bytes")
            print(f"  Max: {max(content_sizes):,} bytes")
            print(f"  Avg: {sum(content_sizes)/len(content_sizes):,.0f} bytes")
            print(f"  Total: {sum(content_sizes):,} bytes")

    timings = [r["elapsed_seconds"] for r in jina_results if r["elapsed_seconds"] is not None]
    if timings:
        print("\nJina request timing:")
        print(f"  Fastest: {min(timings):.2f}s")
        print(f"  Slowest: {max(timings):.2f}s")
        print(f"  Average: {sum(timings)/len(timings):.2f}s")
        print(f"  Parallel speedup: {sum(timings)/jina_elapsed:.2f}x")

    errors = [r["error"] for r in jina_results if not r["success"]]
    if errors:
        print("\nErrors encountered:")
        for error in set(errors):
            count = errors.count(error)
            print(f"  - {error}: {count} occurrence(s)")
    else:
        print("\n✓ No errors encountered")

    print("\n" + "=" * 80)
    print(f"Done! Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
