"""
Test Bright Data Scraping Browser API: connect via Playwright CDP,
fetch URLs in parallel batches, and extract content as markdown.
"""

import asyncio
import os
import time
from typing import TypedDict

import html2text
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

SBR_WS_ENDPOINT = os.getenv("BRIGHT_DATA_SBR_WS")
if not SBR_WS_ENDPOINT:
    raise RuntimeError("BRIGHT_DATA_SBR_WS not found in environment")

BATCH_SIZE = int(os.getenv("BRIGHT_DATA_BATCH_SIZE", "5"))


class ScrapeResult(TypedDict, total=False):
    url: str
    status: int
    elapsed: float
    markdown: str | None
    length: int
    error: str


def _html_to_markdown(html: str) -> str:
    """Convert HTML to clean markdown."""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html)


async def _fetch_single(playwright, url: str) -> ScrapeResult:
    """Fetch a single URL in its own browser session (one domain per session)."""
    start = time.time()
    try:
        browser = await playwright.chromium.connect_over_cdp(SBR_WS_ENDPOINT)
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            html_content = await page.content()
            elapsed = time.time() - start
            markdown = _html_to_markdown(html_content)
            return ScrapeResult(
                url=url,
                status=200,
                elapsed=elapsed,
                markdown=markdown,
                length=len(markdown),
            )
        except Exception as e:
            elapsed = time.time() - start
            return ScrapeResult(
                url=url,
                status=-1,
                elapsed=elapsed,
                markdown=None,
                length=0,
                error=str(e),
            )
        finally:
            await page.close()
            await browser.close()
    except Exception as e:
        elapsed = time.time() - start
        return ScrapeResult(
            url=url,
            status=-1,
            elapsed=elapsed,
            markdown=None,
            length=0,
            error=str(e),
        )


async def scrap_urls(urls: list[str], batch_size: int = BATCH_SIZE) -> list[ScrapeResult]:
    """
    Scrape URLs in parallel using Bright Data Scraping Browser.

    Each URL gets its own browser session (Bright Data limits one domain per
    session). URLs are processed concurrently in batches.

    Args:
        urls: List of URLs to scrape.
        batch_size: Max concurrent sessions per batch (env: BRIGHT_DATA_BATCH_SIZE).

    Returns:
        List of ScrapeResult dicts, in same order as input URLs.
    """
    results: list[ScrapeResult] = []
    async with async_playwright() as p:
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[_fetch_single(p, url) for url in batch]
            )
            results.extend(batch_results)
    return results


async def main():
    test_urls = [
        "https://en.wikipedia.org/wiki/Melbourne",
        "https://www.python.org",
        "https://news.ycombinator.com",
        "https://docs.python.org/3/tutorial/index.html",
        "https://www.bbc.com/news/articles/c1450zj6n48o",
    ]

    print("=" * 70)
    print("Bright Data Scraping Browser — Fetch & Convert to Markdown")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 70)

    connect_start = time.time()
    results = await scrap_urls(test_urls)
    total_elapsed = time.time() - connect_start

    for r in results:
        marker = "OK" if r["status"] == 200 else f"ERR {r['status']}"
        print(f"\n  [{marker}] {r['url']}")
        print(f"         Time: {r['elapsed']:.2f}s  |  Markdown: {r['length']} chars")
        if r.get("markdown"):
            preview = r["markdown"][:200].replace("\n", " ")
            print(f"         Preview: {preview}...")
        elif r.get("error"):
            print(f"         Error: {r['error'][:200]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'URL':<50} {'Time':>8} {'Chars':>8}")
    print("  " + "-" * 66)
    for r in results:
        short_url = r["url"].replace("https://", "")[:48]
        print(f"  {short_url:<50} {r['elapsed']:>7.2f}s {r['length']:>8}")

    successful = [r for r in results if r["status"] == 200]
    if successful:
        times = [r["elapsed"] for r in successful]
        avg = sum(times) / len(times)
        print(
            f"\n  Total: {total_elapsed:.2f}s  |  Avg/page: {avg:.2f}s  |"
            f"  {len(successful)}/{len(results)} succeeded"
        )


if __name__ == "__main__":
    asyncio.run(main())
