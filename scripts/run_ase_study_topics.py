"""
Batch-run the ASE study topics through SearchAgent.

Reads:
  data/raw_responses/ase_study_topic_runs/queries.json
  topics/... (held in another repo — values inlined into queries.json beforehand)

Writes one file per (topic, question) into
  data/raw_responses/ase_study_topic_runs/topic_{tid}_q{qid}.json

Each output captures: query, final answer text, reasoning trace,
citations, per-turn metadata, plus the gold MCQ for downstream judging.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Make `src/` importable when invoked from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from systems.rag_interface import RunRequest  # noqa: E402
from systems.search_agent.search_agent import SearchAgent  # noqa: E402

TOPICS_JSONL = Path(
    "/Users/kun/Projects/rmit/research/ase2.0/ase2-ai-mode/topics/ase_study_topics.jsonl"
)
RUNS_DIR = REPO_ROOT / "data" / "raw_responses" / "ase_study_topic_runs"
QUERIES_FILE = RUNS_DIR / "queries.json"


def load_topics() -> Dict[str, Dict[str, Any]]:
    topics: Dict[str, Dict[str, Any]] = {}
    with TOPICS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            topics[str(t["topic_id"])] = t
    return topics


def gold_for(topics: Dict[str, Dict[str, Any]], tid: str, qid: int) -> Dict[str, Any]:
    topic = topics[tid]
    for q in topic["questions"]:
        if int(q["id"]) == int(qid):
            return {
                "topic_title": topic["title"],
                "topic_category": topic.get("category"),
                "backstory": topic["backstory"],
                "narrative": topic["narrative"],
                "question_text": q["text"],
                "options": q["options"],
                "correct": q["correct"],
            }
    raise KeyError(f"No question {qid} for topic {tid}")


async def run_one(agent: SearchAgent, query: str) -> Dict[str, Any]:
    """Drive one streaming run end-to-end and collect everything."""
    reasoning_chunks: List[str] = []
    answer_chunks: List[str] = []
    citations: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    error: str | None = None

    started = time.time()
    stream_fn = await agent.run_streaming(RunRequest(question=query))
    async for ev in stream_fn():
        if ev.error:
            error = ev.error
        if ev.is_intermediate and ev.intermediate_steps:
            reasoning_chunks.append(ev.intermediate_steps)
        if not ev.is_intermediate and ev.final_report:
            answer_chunks.append(ev.final_report)
        if ev.citations:
            citations = [dict(c) for c in ev.citations]
        if ev.metadata:
            metadata = dict(ev.metadata)
        if ev.complete:
            break
    wall = round(time.time() - started, 3)

    return {
        "answer": "".join(answer_chunks).strip(),
        "reasoning": "".join(reasoning_chunks).strip(),
        "citations": citations,
        "metadata": metadata,
        "wall_seconds": wall,
        "error": error,
    }


async def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    topics = load_topics()
    queries_doc = json.loads(QUERIES_FILE.read_text())
    queries = queries_doc["queries"]

    agent = SearchAgent(
        api_base=os.environ["ALT_LLM_API_BASE_GPT5_MINI"],
        api_key=os.environ["ALT_LLM_API_KEY_GPT5_MINI"],
        model=os.environ.get("ALT_LLM_MODEL_GPT5_MINI", "gpt-5.4-mini"),
        reasoning_effort="medium",
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(queries)
    concurrency = int(os.environ.get("ASE_STUDY_CONCURRENCY", "10"))
    sem = asyncio.Semaphore(concurrency)
    successes = 0
    failures: List[str] = []
    completed = 0
    started_at = time.time()

    async def process(i: int, entry: Dict[str, Any]) -> None:
        nonlocal completed, successes
        tid = str(entry["topic_id"])
        qid = int(entry["question_id"])
        query = entry["query"]

        out_path = RUNS_DIR / f"topic_{tid.zfill(2)}_q{qid}.json"
        if out_path.exists():
            completed += 1
            print(f"[{completed:02d}/{total}] SKIP (exists) topic={tid} q={qid}")
            successes += 1
            return

        async with sem:
            print(f"[start  i={i:02d}] topic={tid} q={qid} :: {query[:100]}")
            try:
                result = await run_one(agent, query)
            except Exception as e:  # noqa: BLE001
                print(f"[fail   i={i:02d}] topic={tid} q={qid} {type(e).__name__}: {e}")
                failures.append(f"topic={tid} q={qid}: {e}")
                result = {
                    "answer": "",
                    "reasoning": "",
                    "citations": [],
                    "metadata": {},
                    "wall_seconds": None,
                    "error": f"{type(e).__name__}: {e}",
                }

            gold = gold_for(topics, tid, qid)
            out_path.write_text(json.dumps({
                "topic_id": tid,
                "question_id": qid,
                "query": query,
                "gold": gold,
                "run": result,
            }, indent=2, ensure_ascii=False))

            completed += 1
            if result.get("error"):
                print(
                    f"[done   {completed:02d}/{total}] topic={tid} q={qid} "
                    f"ERROR {str(result['error'])[:100]}"
                )
            else:
                successes += 1
                print(
                    f"[done   {completed:02d}/{total}] topic={tid} q={qid} "
                    f"chars={len(result['answer'])} "
                    f"cits={len(result['citations'])} "
                    f"wall={result['wall_seconds']}s"
                )

    await asyncio.gather(*(process(i, e) for i, e in enumerate(queries, start=1)))

    elapsed = round(time.time() - started_at, 1)
    print()
    print(f"Done. {successes}/{total} runs succeeded in {elapsed}s (concurrency={concurrency}).")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    asyncio.run(main())
