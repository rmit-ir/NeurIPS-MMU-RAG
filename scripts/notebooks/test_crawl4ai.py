#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = ["crawl4ai-cloud-sdk"]
# ///

import asyncio
import os
import time
from typing import AsyncGenerator, NamedTuple
from crawl4ai_cloud import AsyncWebCrawler, CrawlerRunConfig

CRAWL4AI_API_KEY = os.environ.get("CRAWL4AI_API_KEY", "placeholder")

URLS = [
    "https://en.wikipedia.org/wiki/Web_crawler",
    "https://developers.google.com/crawling/docs/crawlers-fetchers/overview-google-crawlers",
    # "https://crawlee.dev/",
    # "https://store.rc4wd.com/Ready-To-Run-Trucks_c_1.html?srsltid=AfmBOop_gQZLtIAbX2-uPTg4t0gli_l71GPCnHCuYV6cfj4LVC_tDqzC",
    # "https://shop.ozobot.com/products/ozobot-crawler?srsltid=AfmBOooqx0SutYqFLDkFec-7hoUFsDM57m7y-q03N2t6oW1CMCBYoVlB",
    # "https://developer.mozilla.org/en-US/docs/Glossary/Crawler",
    # "https://www.reddit.com/r/cryptids/comments/15gr6rr/crawlers_what_are_they/",
    # "https://www.cloudflare.com/learning/bots/what-is-a-web-crawler/",
    # "https://en.wiktionary.org/wiki/crawler",
    # "https://www.nasa.gov/humans-in-space/exploration-ground-systems/the-crawlers/",
    # "https://andrewkchan.dev/posts/crawler.html",
    # "https://www.imdb.com/title/tt1471153/",
    # "https://store.rc4wd.com/Ready-To-Run-Trucks_c_1.html?srsltid=AfmBOopqCcyeNts5rMbeT0RB0Pr4meSa5rDYm8H4tTXwaEXzjjRD64Qf",
    # "https://en.ryte.com/wiki/Crawler",
]


class CrawlResult(NamedTuple):
    url: str
    success: bool
    content: str
    elapsed: float


def chunk_urls(urls: list[str], size: int):
    for i in range(0, len(urls), size):
        yield urls[i: i + size]


async def crawl_url(
    crawler: AsyncWebCrawler,
    url: str,
    timeout_ms: int = 30000,
    strategy: str = "http",
):
    start = time.perf_counter()
    config = CrawlerRunConfig(page_timeout=timeout_ms)
    try:
        result = await asyncio.wait_for(
            crawler.run(url, strategy=strategy, config=config),
            timeout=timeout_ms / 1000.0,
        )
        elapsed = time.perf_counter() - start
        if result.success:
            return CrawlResult(url, True, result.markdown.raw_markdown[:500], elapsed)
        return CrawlResult(url, False, result.error_message or "", elapsed)
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        return CrawlResult(url, False, "timed out", elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return CrawlResult(url, False, str(exc), elapsed)


async def crawl_urls_in_batches(
    urls: list[str], batch_size: int = 10, timeout_ms: int = 30000
) -> AsyncGenerator[CrawlResult, None]:
    async with AsyncWebCrawler(api_key=CRAWL4AI_API_KEY) as crawler:
        for batch in chunk_urls(urls, batch_size):
            tasks = [crawl_url(crawler, url, timeout_ms=timeout_ms)
                     for url in batch]
            for done in asyncio.as_completed(tasks):
                yield await done


async def main():
    start = time.perf_counter()
    async for result in crawl_urls_in_batches(URLS, batch_size=10, timeout_ms=10000):
        print(f"-: {result.elapsed:.2f}s\t{result.url}")
        if result.success:
            print(' ', result.content[:100].replace('\n', ' ') + "...")
        else:
            print(f" : {result.content}")
    elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {elapsed:.2f}s")


asyncio.run(main())
