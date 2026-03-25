#!/usr/bin/env python
"""Test script for the brave_answers tool."""

import argparse
import asyncio

from dotenv import load_dotenv

from tools.brave_answers import brave_answers

load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="Test Brave Answers tool")
    parser.add_argument("query", nargs="?", default="Lunisolar calendar")
    parser.add_argument(
        "--research",
        action="store_true",
        help="Enable research mode (slower, more thorough)",
    )
    args = parser.parse_args()

    print(f"=== Query: {args.query} ===")
    if args.research:
        print("Research mode: ON")
    print()

    result = await brave_answers(
        query=args.query,
        enable_research=args.research,
    )

    print(result.content)

    if result.citations:
        print(f"\n--- Citations ({len(result.citations)}) ---")
        for c in result.citations:
            snippet_preview = f" - {c.snippet[:80]}..." if c.snippet else ""
            print(f"  [{c.number}] {c.url}{snippet_preview}")

    print(f"\n--- Usage ---")
    print(f"  Queries used: {result.queries_used}")
    print(f"  Tokens in:    {result.tokens_in}")
    print(f"  Tokens out:   {result.tokens_out}")
    print(f"  Total cost:   ${result.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
