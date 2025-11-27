# uv sync --extra crawl
# uv run src/tools/crawler_search/crawler.py
# open /tmp/crawl_results
import asyncio
import re
import time
from typing import AsyncGenerator, List
from aiohttp_retry import Any, Union
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlResult, CrawlerRunConfig, MemoryAdaptiveDispatcher, RateLimiter

VERBOSE = True
PAGE_TIMEOUT = 10_000  # in milliseconds
URLs = [
    "https://docs.crawl4ai.com/core/installation/",
    "https://www.kbb.com/toyota/camry/",
    # immich Enable full-size image generation
    # "https://github.com/immich-app/immich/discussions/6306",
    # "https://github.com/immich-app/immich/releases/tag/v1.131.0",
    # "https://immich.app/docs/administration/system-settings/",
    # "https://www.reddit.com/r/immich/comments/1fnmm2y/is_it_possible_to_just_load_full_images_directly/",
    # "https://svrforum.com/itnews/2246288",
    # "https://www.answeroverflow.com/m/1321475263988502568",
    # "https://github.com/immich-app/immich/issues/18906",
    # "https://www.reddit.com/r/immich/comments/1ef4s42/can_i_stop_immich_from_changing_photo_size/",
    # "https://forums.unraid.net/topic/169036-immich-docker-image-ballooning/",
    # "https://github.com/immich-app/immich/discussions/3336",
    # "https://immich.app/docs/FAQ/",
    # "https://github.com/immich-app/immich/discussions/11457",
    "https://github.com/immich-app/immich/discussions/18515",
    "https://www.xda-developers.com/heres-how-you-can-replace-google-photos-with-a-self-hosted-immich-server/",
    "https://immich.app/",
    # "https://www.reddit.com/r/immich/comments/1ibz9bt/introducing_immich_upload_optimizer_save_storage/",
    # "https://github.com/immich-app/immich/discussions/20437",
    # "https://immich-power-tools.featureos.app/",
    # "https://docs.pikapods.com/tutorials/photo/immich-1-basics/",
    # are there plugins for Emby to download lyrics for my songs?
    "https://www.reddit.com/r/emby/comments/c759qm/is_there_any_lyrics_plugins_or_plans_to_include/",
    "https://emby.media/support/articles/Music-Lyrics.html",
    "https://emby.media/community/index.php?/topic/136066-lyrics-plugin-fetch-synced-plain-text-for-testing/",
    # "https://emby.media/community/index.php?/topic/55024-lyrics-plugin-for-emby/",
    # "https://emby.freshdesk.com/support/solutions/articles/44002394743-music-lyrics",
    # "https://emby.media/community/index.php?/topic/110773-adding-lyrics-with-lyrics-finder/",
    # "https://emby.media/community/index.php?/topic/123494-emby-lyrics/",
    # "https://emby.media/community/index.php?/topic/144009-looking-for-pointers-to-get-lyrics-for-my-music-collection/",
    # "https://emby.media/community/index.php?/topic/114042-some-advice-on-downloads-and-lyrics-on-ios/",
    # "https://emby.media/community/index.php?/topic/110074-lyrics-how-to-add-them-to-the-library/",
    # "https://emby.media/community/index.php?/topic/109393-music-lyrics-on-version-472/",
    # "https://emby.media/support/articles/Plugins.html",
    # "https://emby.media/community/index.php?/topic/112780-lyrics/",
    # "https://github.com/Techsmith404/Emby-Plugins-List",
    # "https://emby.media/support/articles/Plugins-Duplicate.html",
    # "https://emby.media/community/index.php?/topic/129853-scrolling-lyrics/",
    # "https://github.com/oldkingOK/EmbyLyricEnhance",
    "https://www.reddit.com/r/emby/comments/1kbfz1h/music_player_for_emby_an_fyi/",
    "https://www.reddit.com/r/emby/comments/1emuyqj/tweaks_and_plugins/",
    # "https://emby.media/community/index.php?/topic/130323-emby-does-not-display-lrc-lyrics/"
]


async def fetch_urls(urls: list[str]):
    browser_config = BrowserConfig(headless=False)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=VERBOSE,
        page_timeout=PAGE_TIMEOUT,
        excluded_tags=["header", "nav", "footer", "svg", "style",
                       "script", "noscript", "img", "video", "audio", "canvas"],
    )
    # Dispatcher with rate limiter enabled (default behavior)
    dispatcher = MemoryAdaptiveDispatcher(
        rate_limiter=RateLimiter(
            base_delay=(0.5, 1.0), max_delay=60.0, max_retries=3
        ),
        max_session_permit=50,
    )
    start_time = time.perf_counter()
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result_container: Any = await crawler.arun_many(
            urls,
            config=config,
            dispatcher=dispatcher
        )
        results: List[CrawlResult] = []
        typed_result: Union[List[CrawlResult],
                            AsyncGenerator[CrawlResult, None]] = result_container
        if isinstance(typed_result, list):
            results = typed_result
        else:
            async for res in typed_result:
                results.append(res)
    total_time = time.perf_counter() - start_time
    return total_time, results


async def fetch_url(url: str):
    browser_config = BrowserConfig(headless=False)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=VERBOSE,
        page_timeout=PAGE_TIMEOUT,
        excluded_tags=["header", "nav", "footer", "svg", "style",
                       "script", "noscript", "img", "video", "audio", "canvas"],
    )
    # Dispatcher with rate limiter enabled (default behavior)
    dispatcher = MemoryAdaptiveDispatcher(
        rate_limiter=RateLimiter(
            base_delay=(0.1, 0.3), max_delay=60.0, max_retries=3
        ),
        max_session_permit=50,
    )
    start_time = time.perf_counter()
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result: Any = await crawler.arun(
            url,
            config=config,
            dispatcher=dispatcher
        )
        typed_result: CrawlResult = result
    total_time = time.perf_counter() - start_time
    return total_time, typed_result


async def save_crawl_results(results: List[CrawlResult], folder_path: str):
    import os
    os.makedirs(folder_path, exist_ok=True)
    for res in results:
        file_key = re.sub(r"[^a-zA-Z0-9]", "_", res.url)
        file_name = f"{file_key}.txt"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"""URL: {res.url}
Status Code: {res.status_code}
{ "Markdown" if res.markdown else "HTML"} Content:
{"-" * 80}
{res.markdown if res.markdown else res.html}
{"-" * 80}
{"="*80}
""")


async def main():
    total_time, results = await fetch_urls(URLs)
    print(f"Fetched {len(results)} pages in {total_time:.2f} seconds")
    await save_crawl_results(results, folder_path="/tmp/crawl_results")
    for res in results:
        print(
            f"URL: {res.url}, Status: {res.status_code}, Words: {len(res.markdown.split()) if res.markdown else 0}")

    # total_time, res = await fetch_url("https://docs.crawl4ai.com/core/installation/")
    # print(f"Fetched page in {total_time:.2f} seconds")
    # print(
    #     f"URL: {res.url}, Status: {res.status_code}, Words: {len(res.markdown.split()) if res.markdown else 0}")

if __name__ == "__main__":
    asyncio.run(main())
