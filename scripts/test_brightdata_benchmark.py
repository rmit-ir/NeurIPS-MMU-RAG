"""
⏺ Bash(uv run python scripts/test_brightdata_benchmark.py)
 ⎿  ======================================================================
     Bright Data SERP API Benchmark
     ======================================================================

     --- Global Searches ---
       [OK] pizza                          4.96s  (from: Thailand (TH))
       [OK] best restaurants               3.71s  (from: Uzbekistan (UZ))
       [OK] weather forecast               3.11s  (from: Egypt (EG))
       [OK] python programming             35.73s  (from: Chile (CL))
       [OK] machine learning tutorial      2.31s  (from: Italy (IT))
       [OK] Hello fresh                    1.93s  (from: Venezuela (VE))
       [OK] laptop reviews                 2.49s  (from: Mexico (MX))
       [OK] how to make sourdough          3.85s  (from: Dominican Republic (DO))
       [OK] latest news today              3.76s  (from: Colombia (CO))
       [OK] running shoes                  5.19s  (from: Egypt (EG))

     --- Australia (Victoria) Searches ---
       [OK] pizza                          3.17s  (from: Australia (AU))
       [OK] best restaurants               4.20s  (from: Australia (AU))
       [OK] weather forecast               4.41s  (from: Australia (AU))
       [OK] python programming             2.11s  (from: Australia (AU))
       [OK] machine learning tutorial      3.34s  (from: Australia (AU))
       [OK] Hello fresh                    6.12s  (from: Australia (AU))
       [OK] laptop reviews                 3.18s  (from: Australia (AU))
       [OK] how to make sourdough          4.66s  (from: Australia (AU))
       [OK] latest news today              4.94s  (from: Australia (AU))
       [OK] running shoes                  4.17s  (from: Australia (AU))

     ======================================================================
     TIMING SUMMARY
     ======================================================================
     Metric           Global   AU (VIC)
     --------------------------------
     Average           6.70s      4.03s
     Median            3.73s      4.18s
     P95              35.73s      6.12s
     ======================================================================

     ======================================================================
     RESULTS COMPARISON (Top 5 organic results)
     ======================================================================

     >>> Query: "pizza"
       Global:
         1. เดอะ พิซซ่า คอมปะนี 1112 ออนไลน์ดิลิเวอรี ซื้อกลับ
         2. PizzaHut Thailand
         3. พิซซ่าซื้อ 1 แถม 1 - เดอะ พิซซ่า คอมปะนี 1112
         4. Domino's Pizza Thailand: โดมิโน่ส์ พิซซ่า | ความอร่อยสั่งได้ สั่งเลย ...
         5. The Pizza Company 1112 | Bangkok
       AU (Victoria):
         1. Domino's | Order Pizza Delivery Online | Food Delivery ...
         2. THE 10 BEST Pizza Places in Melbourne (Updated 2026)
         3. Pizza
         4. PIZZA PIZZA PIZZA
         5. Order from Melbourne
       Overlap: 0/5 results in common

     >>> Query: "best restaurants"
       Global:
         1. THE 10 BEST Restaurants in Tashkent ... - Tripadvisor
         2. The World's 50 Best Restaurants | The best restaurants in the ...
         3. 17 BEST Restaurants in Tashkent, Uzbekistan to Try Now
         4. The Best Restaurants In Tashkent, Uzbekistan
         5. What are good restaurants and dishes in Tashkent ...
       AU (Victoria):
         1. Best Restaurants Australia: Discover the best restaurants ...
         2. 15 of The Best Restaurants in Melbourne
         3. THE 10 BEST Restaurants with a View in Melbourne
         4. Melbourne - Restaurants, Bars, Cafes & Things to Do
         5. The 10 Best Restaurants In Melbourne For 2026
       Overlap: 0/5 results in common

     >>> Query: "weather forecast"
       Global:
         1. Cairo, Cairo, Egypt Weather Forecast
         2. AccuWeather: Local, National, & Global Daily Weather Forecast
         3. Cairo - BBC Weather
         4. National and Local Weather Radar, Daily Forecast and News ...
         5. Cairo, Egypt 14 day weather forecast
       AU (Victoria):
         1. Melbourne, Victoria | The Bureau of Meteorology
         2. Today's Local Weather Forecast - Melbourne
         3. Daily Forecast West Melbourne, Victoria 3003, Australia
         4. Melbourne, Victoria, Australia Weather Forecast
         5. East Melbourne Forecast - Bureau of Meteorology
       Overlap: 0/5 results in common

     >>> Query: "python programming"
       Global:
         1. Welcome to Python.org
         2. Introduction to Python
         3. Python (programming language)
         4. Online Python Compiler (Interpreter)
         5. Python Tutorial
       AU (Victoria):
         1. Welcome to Python.org
         2. Introduction to Python
         3. Python (programming language)
         4. Best Python Courses + Tutorials
         5. Introduction to Python Programming
       Overlap: 3/5 results in common

     >>> Query: "machine learning tutorial"
       Global:
         1. Machine Learning Tutorial
         2. Machine Learning Crash Course
         3. Python Machine Learning
         4. Machine Learning (ML) Tutorial
         5. Intro to Machine Learning
       AU (Victoria):
         1. Machine Learning Crash Course
         2. Machine Learning Tutorial
         3. Python Machine Learning
         4. Machine Learning (ML) Tutorial
         5. The 2026 Guide to Machine Learning
       Overlap: 4/5 results in common

     >>> Query: "Hello fresh"
       Global:
         1. HelloFresh® Meal Kits | 10 Free Meals + Free Zwilling Four ...
         2. HelloFresh Group
         3. HelloFresh US (@hellofresh) • Instagram photos and videos
         4. HelloFresh
         5. HelloFresh: Meal Kit Delivery - Apps on Google Play
       AU (Victoria):
         1. HelloFresh: Simply Delicious Meals Delivered To Your Door
         2. https://www.hellofresh.com/my-account/deliveries/w...
         3. HelloFresh (@hellofreshau) • Instagram photos and videos
         4. HelloFresh
         5. HelloFresh (@HelloFreshAU)
       Overlap: 1/5 results in common

     >>> Query: "laptop reviews"
       Global:
         1. Best Laptops (2026): My Honest Advice Having Tested ...
         2. The 14 Best Laptops of 2026 | Reviews by Wirecutter
         3. Expert Reviews and News on Laptops, Smartphones and ...
         4. Laptop Reviews
         5. PCs: Buying Advice, Tips, and News
       AU (Victoria):
         1. Best Laptops 2026: Tested Reviews By The Ardor's Team
         2. The Best Laptops for 2026
         3. Computer reviews - Laptops
         4. Laptop Reviews
         5. The best laptops of 2026 in Australia: Top options for all ...
       Overlap: 1/5 results in common

     >>> Query: "how to make sourdough"
       Global:
         1. Sourdough Bread: A Beginner's Guide
         2. Homemade Sourdough Bread, Step by Step
         3. Easy Sourdough Bread Recipe
         4. How To Bake Simple Sourdough Bread: A Beginner's Guide
         5. Beginner's Sourdough Bread (With Video)
       AU (Victoria):
         1. Sourdough Bread: A Beginner's Guide
         2. Homemade Sourdough Bread, Step by Step
         3. Beginner's Sourdough Bread (With Video)
         4. Basic sourdough bread | Sourdough Recipe - BakeClub
         5. Sourdough starter recipe
       Overlap: 3/5 results in common

     >>> Query: "latest news today"
       Global:
         1. CNN: Breaking News, Latest News and Videos
         2. BBC Home - Breaking News, World News, US News, Sports ...
         3. Google News
         4. Breaking News, World News and Video from Al Jazeera
         5. NBC News - Breaking Headlines and Video Reports on World ...
       AU (Victoria):
         1. Just In - ABC News
         2. 9News - Latest news and headlines from Australia and the world
         3. Herald Sun | Breaking News And Headlines From Melbourne ...
         4. news.com.au — Australia's leading news site for latest headlines
         5. The Age: Latest & Breaking News Melbourne, Victoria
       Overlap: 0/5 results in common

     >>> Query: "running shoes"
       Global:
         1. Shop adidas Running Shoes Online
         2. Men's Running Shoes
         3. Running Shoes
         4. Running shoes and trainers for men
         5. The 15 Best Running Shoes of 2026
       AU (Victoria):
         1. The 15 Best Running Shoes of 2026
         2. Men's Running & Jogging Shoes Australia
         3. Women's Running Shoes - Over 50+ Brands
         4. Men's Running Shoes | Runners for Men
         5. Women's Running Shoes
       Overlap: 1/5 results in common

⏺ All 20 requests returned 200 OK. Key findings:

  Timing:

  ┌─────────┬────────┬──────────┐
  │ Metric  │ Global │ AU (VIC) │
  ├─────────┼────────┼──────────┤
  │ Average │ 6.70s  │ 4.03s    │
  ├─────────┼────────┼──────────┤
  │ Median  │ 3.73s  │ 4.18s    │
  ├─────────┼────────┼──────────┤
  │ P95     │ 35.73s │ 6.12s    │
  └─────────┴────────┴──────────┘

  Global had one outlier (35.73s for "python programming") that skewed the average and p95. Median times are comparable
   (~3.7s vs ~4.2s).

  Results:
  - AU localization works perfectly — all AU requests show country: Australia (AU) and return
  Melbourne/Victoria-specific results
  - Global searches route through random countries (Thailand, Uzbekistan, Egypt, etc.)
  - Location-dependent queries (pizza, restaurants, weather, news) show 0/5 overlap — completely different results
  - Universal queries (python, ML tutorial, sourdough) show 3-4/5 overlap — similar results regardless of location
"""
import os
import statistics
import time
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BRIGHT_DATA_API_KEY")
if not API_KEY:
    raise RuntimeError("BRIGHT_DATA_API_KEY not found in environment")

ENDPOINT = "https://api.brightdata.com/request"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
ZONE = "serp_api_test_1"

QUERIES = [
    "pizza",
    "best restaurants",
    "weather forecast",
    "python programming",
    "machine learning tutorial",
    "Hello fresh",
    "laptop reviews",
    "how to make sourdough",
    "latest news today",
    "running shoes",
]


def make_request(url: str) -> tuple[int, float, dict | None]:
    """Make a SERP API request and return (status_code, elapsed_seconds, parsed_json)."""
    data = {"zone": ZONE, "url": url, "format": "raw"}
    start = time.time()
    resp = requests.post(ENDPOINT, json=data, headers=HEADERS)
    elapsed = time.time() - start
    try:
        body = resp.json()
    except Exception:
        body = None
    return resp.status_code, elapsed, body


def extract_top_results(body: dict | None, n: int = 5) -> list[str]:
    """Extract top N organic result titles from parsed SERP response."""
    if not body:
        return []
    organic = body.get("organic", [])
    return [r.get("title", "(no title)") for r in organic[:n]]


def extract_country(body: dict | None) -> str:
    """Extract detected country from response."""
    if not body:
        return "N/A"
    general = body.get("general", {})
    country = general.get("country", "N/A")
    code = general.get("country_code", "")
    return f"{country} ({code})" if code else country


def compute_stats(times: list[float]) -> dict:
    """Compute average, median, and p95 from a list of times."""
    sorted_times = sorted(times)
    p95_idx = int(len(sorted_times) * 0.95)
    p95_idx = min(p95_idx, len(sorted_times) - 1)
    return {
        "average": statistics.mean(times),
        "median": statistics.median(times),
        "p95": sorted_times[p95_idx],
    }


def run_benchmark():
    print("=" * 70)
    print("Bright Data SERP API Benchmark")
    print("=" * 70)

    global_results = {}
    au_results = {}

    # --- Global searches ---
    print("\n--- Global Searches ---")
    global_times = []
    for query in QUERIES:
        url = f"https://www.google.com/search?q={quote_plus(query)}"
        status, elapsed, body = make_request(url)
        global_times.append(elapsed)
        global_results[query] = body
        marker = "OK" if status == 200 else f"ERR {status}"
        country = extract_country(body)
        print(f"  [{marker}] {query:30s} {elapsed:.2f}s  (from: {country})")

    # --- Australia / Victoria searches ---
    print("\n--- Australia (Victoria) Searches ---")
    au_times = []
    for query in QUERIES:
        url = (
            f"https://www.google.com/search?q={quote_plus(query)}"
            f"&gl=AU&hl=en-AU&uule=Melbourne,Victoria,Australia"
        )
        status, elapsed, body = make_request(url)
        au_times.append(elapsed)
        au_results[query] = body
        marker = "OK" if status == 200 else f"ERR {status}"
        country = extract_country(body)
        print(f"  [{marker}] {query:30s} {elapsed:.2f}s  (from: {country})")

    # --- Timing Summary ---
    global_stats = compute_stats(global_times)
    au_stats = compute_stats(au_times)

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<12} {'Global':>10} {'AU (VIC)':>10}")
    print("-" * 32)
    print(f"{'Average':<12} {global_stats['average']:>9.2f}s {au_stats['average']:>9.2f}s")
    print(f"{'Median':<12} {global_stats['median']:>9.2f}s {au_stats['median']:>9.2f}s")
    print(f"{'P95':<12} {global_stats['p95']:>9.2f}s {au_stats['p95']:>9.2f}s")
    print("=" * 70)

    # --- Results Comparison ---
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON (Top 5 organic results)")
    print("=" * 70)

    for query in QUERIES:
        print(f"\n>>> Query: \"{query}\"")

        global_top = extract_top_results(global_results.get(query))
        au_top = extract_top_results(au_results.get(query))

        print("  Global:")
        for i, title in enumerate(global_top, 1):
            print(f"    {i}. {title}")
        if not global_top:
            print("    (no organic results)")

        print("  AU (Victoria):")
        for i, title in enumerate(au_top, 1):
            print(f"    {i}. {title}")
        if not au_top:
            print("    (no organic results)")

        # Show overlap
        global_set = set(global_top)
        au_set = set(au_top)
        if global_set and au_set:
            overlap = len(global_set & au_set)
            print(f"  Overlap: {overlap}/{max(len(global_set), len(au_set))} results in common")


if __name__ == "__main__":
    run_benchmark()
