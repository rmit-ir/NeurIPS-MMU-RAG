"""
SearchAgent: tool-calling agent backed by Brave LLM Context.

Uses OpenAI's Responses API so a reasoning model (e.g. gpt-5.4-mini) can
decide when and how to issue `web_search` calls, iterate on results, and
emit cited answers — instead of the fixed search→rerank→answer flow of
VanillaRAG or the templated review loop of VanillaAgent.

Compatible with the existing RAGInterface contract: emits CitationItems
in the same shape and streams via RunStreamingResponse.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from systems.rag_interface import (
    CitationItem,
    RAGInterface,
    RunRequest,
    RunStreamingResponse,
)
from systems.search_agent.functions import exec_web_search
from systems.search_agent.prompts import SYSTEM_PROMPT
from systems.search_agent.tools import WEB_SEARCH_TOOL
from tools.brave_llm_context import COST_PER_CALL_USD as BRAVE_COST_PER_CALL_USD
from tools.logging_utils import get_logger
from tools.posthog_client import (
    capture_ai_trace_summary,
    capture_responses_turn,
    make_trace_ids,
)


class SearchAgent(RAGInterface):
    """Reasoning-model agent that calls Brave LLM Context as a tool."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        reasoning_effort: str = "medium",
        max_tool_calls: int = 12,
        max_turns: int = 8,
        per_search_count: int = 5,
        per_search_max_tokens: int = 4096,
        request_timeout: float = 600.0,
        parallel_tool_calls: bool = True,
    ):
        """
        Args:
            api_base: OpenAI-compatible base URL. For Azure AI Foundry v1 use
                the path ending in '/openai/v1/'.
            api_key: API key (sent as bearer token).
            model: Model / deployment name (e.g. 'gpt-5.4-mini').
            reasoning_effort: 'minimal' | 'low' | 'medium' | 'high'. Avoid
                'minimal' — it disables parallel tool calls.
            max_tool_calls: Hard cap on total web_search invocations across
                the whole conversation (a single turn can issue several
                via parallel_tool_calls).
            max_turns: Hard cap on assistant rounds (loop iterations). One
                turn = one Responses.create() call, which may emit any
                number of parallel tool calls. Acts as a runaway guard.
            per_search_count: Default max URLs per search result.
            per_search_max_tokens: Approximate per-search token budget.
            parallel_tool_calls: Whether to allow the model to fan out
                multiple tool calls in one turn.
        """
        if not api_base:
            raise ValueError("api_base is required")
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")

        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tool_calls = max_tool_calls
        self.max_turns = max_turns
        self.per_search_count = per_search_count
        self.per_search_max_tokens = per_search_max_tokens
        self.parallel_tool_calls = parallel_tool_calls

        self.logger = get_logger("search_agent")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=request_timeout,
            max_retries=2,
        )
        self.logger.info(
            "Initialized SearchAgent",
            model=model,
            api_base=api_base,
            reasoning_effort=reasoning_effort,
            max_tool_calls=max_tool_calls,
            max_turns=max_turns,
        )

    @property
    def name(self) -> str:
        return f"search-agent-{self.model}"

    async def run_streaming(
        self, request: RunRequest
    ) -> Callable[[], AsyncGenerator[RunStreamingResponse, None]]:
        async def stream() -> AsyncGenerator[RunStreamingResponse, None]:
            citations_by_url: Dict[str, CitationItem] = {}
            tool_calls_used = 0
            previous_response_id: Optional[str] = None

            now = datetime.now().astimezone().strftime("%A, %Y-%m-%d %H:%M:%S %Z (UTC%z)")
            system_prompt = SYSTEM_PROMPT.format(
                max_tool_calls=self.max_tool_calls,
                now=now,
            )
            input_items: List[Dict[str, Any]] = [
                {"type": "message", "role": "system", "content": system_prompt},
                {"type": "message", "role": "user", "content": request.question},
            ]

            # PostHog: trace = one user question.
            trace_id, distinct_id = make_trace_ids(
                prefix="search-agent", seed=request.question
            )
            trace_started_at = time.time()
            # accumulate answer text for trace summary
            answer_buf: List[str] = []
            time_to_first_answer_token: Optional[float] = None
            time_to_first_think_token: Optional[float] = None

            try:
                yield _inter(f"Question: {request.question}\n\n")

                query_history: List[str] = []  # queries actually executed

                for turn in range(self.max_turns):
                    create_kwargs: Dict[str, Any] = {
                        "model": self.model,
                        "input": input_items,
                        "tools": [WEB_SEARCH_TOOL],
                        "tool_choice": "auto",
                        "reasoning": {
                            "effort": self.reasoning_effort,
                            "summary": "auto",
                        },
                        "parallel_tool_calls": self.parallel_tool_calls,
                    }
                    if previous_response_id is not None:
                        create_kwargs["previous_response_id"] = previous_response_id

                    # Collect function calls emitted in this turn so we can
                    # execute them after the stream completes.
                    pending_calls: List[Dict[str, Any]] = []
                    call_args_buf: Dict[str, List[str]] = {}
                    call_meta: Dict[str, Dict[str, str]] = {}
                    # SearchAgent emits exactly one final answer to the user;
                    # output_text in iterations that also issue function calls
                    # is treated as internal narration (or degeneration leak,
                    # e.g. Harmony format) and routed to intermediate, never
                    # into final_report.
                    iteration_has_function_call = False
                    saw_output_text = False

                    self.logger.info(
                        "Starting Responses turn",
                        turn=turn,
                        tool_calls_used=tool_calls_used,
                    )

                    turn_started_at = time.time()
                    turn_answer_text: List[str] = []
                    turn_reasoning_text: List[str] = []
                    # Captured if the stream ends with status="incomplete"
                    # (e.g. Azure content filter, max_output_tokens). The SDK
                    # only populates its internal completed_response on
                    # status="completed", so we keep our own handle.
                    incomplete_response: Any = None
                    async with self.client.responses.stream(**create_kwargs) as resp_stream:
                        async for event in resp_stream:
                            etype = getattr(event, "type", "")

                            # Reasoning summary: surface as intermediate steps.
                            if etype == "response.reasoning_summary_text.delta":
                                delta = getattr(event, "delta", "") or ""
                                if delta:
                                    if time_to_first_think_token is None:
                                        time_to_first_think_token = round(
                                            time.time() - trace_started_at, 3)
                                    turn_reasoning_text.append(delta)
                                    yield _inter(delta)

                            # New output item: track function calls so we can
                            # collect arguments as they stream in.
                            elif etype == "response.output_item.added":
                                item = getattr(event, "item", None)
                                item_type = getattr(
                                    item, "type", None) if item else None
                                if item_type == "function_call":
                                    iteration_has_function_call = True
                                    item_id = getattr(item, "id", "") or ""
                                    call_id = getattr(
                                        item, "call_id", "") or ""
                                    name = getattr(item, "name", "") or ""
                                    call_args_buf[item_id] = []
                                    call_meta[item_id] = {
                                        "call_id": call_id,
                                        "name": name,
                                    }

                            elif etype == "response.function_call_arguments.delta":
                                item_id = getattr(event, "item_id", "") or ""
                                delta = getattr(event, "delta", "") or ""
                                if item_id in call_args_buf and delta:
                                    call_args_buf[item_id].append(delta)

                            elif etype == "response.function_call_arguments.done":
                                item_id = getattr(event, "item_id", "") or ""
                                full = getattr(event, "arguments", None)
                                if full is None and item_id in call_args_buf:
                                    full = "".join(call_args_buf[item_id])
                                meta = call_meta.get(item_id, {})
                                pending_calls.append(
                                    {
                                        "item_id": item_id,
                                        "call_id": meta.get("call_id", ""),
                                        "name": meta.get("name", ""),
                                        "arguments": full or "",
                                    }
                                )
                                try:
                                    parsed_preview = json.loads(full or "{}")
                                    q_preview = parsed_preview.get("query", "")
                                except Exception:  # noqa: BLE001
                                    q_preview = (full or "")[:120]
                                yield _inter(f"\n\nSearching: {q_preview}\n\n")

                            # Assistant message text. Only the LAST iteration
                            # (no function calls) produces the user-facing
                            # final answer. Text in tool-calling iterations
                            # is model narration or degeneration (e.g. the
                            # gpt-5.x Harmony "to=functions.X" leak) and must
                            # not pollute the final_report stream.
                            elif etype == "response.output_text.delta":
                                delta = getattr(event, "delta", "") or ""
                                if not delta:
                                    pass
                                elif iteration_has_function_call:
                                    turn_answer_text.append(delta)
                                    yield _inter(delta)
                                else:
                                    if not saw_output_text:
                                        time_to_first_answer_token = round(
                                            time.time() - trace_started_at, 3
                                        )
                                    saw_output_text = True
                                    turn_answer_text.append(delta)
                                    answer_buf.append(delta)
                                    yield RunStreamingResponse(
                                        final_report=delta,
                                        is_intermediate=False,
                                        complete=False,
                                    )

                            elif etype == "response.error":
                                err = getattr(event, "error", None)
                                msg = getattr(err, "message", str(
                                    err)) if err else "unknown"
                                raise RuntimeError(
                                    f"Responses stream error: {msg}")

                            elif etype == "response.incomplete":
                                incomplete_response = getattr(
                                    event, "response", None)
                                reason = "unknown"
                                details = getattr(
                                    incomplete_response, "incomplete_details", None
                                ) if incomplete_response else None
                                if details is not None:
                                    reason = getattr(
                                        details, "reason", None) or reason
                                self.logger.warning(
                                    "Responses stream ended incomplete",
                                    turn=turn,
                                    reason=reason,
                                )
                                yield _inter(
                                    f"\n(Response ended early: {reason}.)\n"
                                )

                        try:
                            final_response = await resp_stream.get_final_response()
                        except RuntimeError:
                            if incomplete_response is None:
                                raise
                            final_response = incomplete_response

                    previous_response_id = getattr(final_response, "id", None)

                    # Served tier ("default" = Azure downgraded from priority).
                    self.logger.info("Responses turn served", turn=turn, service_tier_served=getattr(
                        final_response, "service_tier", None))

                    # Decide each call's fate first (parse args, check
                    # budget, accumulate tool_calls_used) before any
                    # awaits, so the budget accounting is deterministic
                    # regardless of how the parallel coroutines
                    # interleave. Doing it before the generation event is
                    # captured also tells us how many billable searches
                    # this turn is about to trigger.
                    tool_calls_used_before = tool_calls_used
                    plans: List[Dict[str, Any]] = []
                    for call in pending_calls:
                        plan: Dict[str, Any] = {
                            "call": call, "body": None, "args": None}
                        try:
                            plan["args"] = json.loads(call["arguments"] or "{}")
                        except json.JSONDecodeError as e:
                            plan["body"] = f"Error: invalid JSON arguments: {e}"
                        else:
                            if call["name"] != "web_search":
                                plan["body"] = f"Error: unknown tool {call['name']!r}."
                            elif tool_calls_used >= self.max_tool_calls:
                                plan["body"] = (
                                    "Search budget exhausted — this call "
                                    "was not executed. Do not call "
                                    "web_search again; produce your "
                                    "final answer using the evidence "
                                    "already collected, and note any "
                                    "remaining gaps."
                                )
                            else:
                                tool_calls_used += 1
                                query_history.append(
                                    (plan["args"].get("query")
                                     or "").strip()[:200]
                                )
                        plans.append(plan)

                    searches_this_turn = sum(
                        1 for p in plans if p["body"] is None)

                    capture_responses_turn(
                        distinct_id=distinct_id,
                        trace_id=trace_id,
                        turn=turn,
                        model=self.model,
                        provider="azure-openai",
                        started_at=turn_started_at,
                        final_response=final_response,
                        input_items=input_items,
                        answer_text="".join(turn_answer_text),
                        reasoning_text="".join(turn_reasoning_text),
                        pending_calls=pending_calls,
                        question=request.question,
                        response_id=previous_response_id,
                        web_search_count=searches_this_turn,
                        web_search_cost_usd=(
                            searches_this_turn * BRAVE_COST_PER_CALL_USD
                        ),
                        extra_properties={
                            "agent_kind": "search_agent_run",
                            "reasoning_effort": self.reasoning_effort,
                            "search_agent_tool_calls_used_before": tool_calls_used_before,
                        },
                    )

                    # If the model issued tool calls, execute them — in
                    # parallel — and feed the outputs back in the next turn.
                    if pending_calls:
                        # Execute the plans that need a real search,
                        # concurrently.
                        async def _run(plan: Dict[str, Any]) -> str:
                            if plan["body"] is not None:
                                return plan["body"]
                            return await exec_web_search(
                                plan["args"],
                                citations_by_url,
                                default_count=self.per_search_count,
                                max_tokens=self.per_search_max_tokens,
                                trace_id=trace_id,
                                distinct_id=distinct_id,
                                parent_id=previous_response_id,
                            )

                        bodies = await asyncio.gather(*[_run(p) for p in plans])

                        next_input: List[Dict[str, Any]] = []
                        for plan, body in zip(plans, bodies):
                            footer = _state_footer(
                                tool_calls_used=tool_calls_used,
                                max_tool_calls=self.max_tool_calls,
                                turns_used=turn + 1,
                                max_turns=self.max_turns,
                                citations_count=len(citations_by_url),
                                query_history=query_history,
                            )
                            next_input.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": plan["call"]["call_id"],
                                    "output": f"{body}\n\n{footer}",
                                }
                            )

                        # With previous_response_id, only send the new
                        # function_call_output items; prior reasoning is
                        # preserved server-side.
                        input_items = next_input
                        continue

                    # No further tool calls (or budget exhausted with no
                    # forced-answer turn left) — we're done with the loop.
                    if not saw_output_text:
                        # Fallback: pull text from final response if streaming
                        # didn't surface any (rare edge case).
                        text = _extract_text(final_response)
                        if text:
                            yield RunStreamingResponse(
                                final_report=text,
                                is_intermediate=False,
                                complete=False,
                            )
                    break
                else:
                    # for-loop fell through: max_turns hit while the model
                    # was still issuing tool calls. Surface a note so the
                    # user knows the answer is truncated.
                    self.logger.warning(
                        "max_turns exhausted before model produced an answer",
                        max_turns=self.max_turns,
                        tool_calls_used=tool_calls_used,
                    )
                    yield RunStreamingResponse(
                        final_report=(
                            "\n\n(Stopped after "
                            f"{self.max_turns} turns without a final answer.)"
                        ),
                        is_intermediate=False,
                        complete=False,
                    )

                # Emit the trace-root summary BEFORE the final yield. If we
                # put it after, consumers that break on complete=True (the
                # normal pattern) leave this code suspended at the yield —
                # it only runs when the generator is GC'd, which is too late
                # for PostHog to anchor the trace name to the question.
                wall_latency = round(time.time() - trace_started_at, 3)
                capture_ai_trace_summary(
                    distinct_id=distinct_id,
                    trace_id=trace_id,
                    span_name=_truncate_for_name(request.question),
                    started_at=trace_started_at,
                    question=request.question,
                    answer="".join(answer_buf),
                    extra_properties={
                        "agent_kind": "search_agent_run",
                        "reasoning_effort": self.reasoning_effort,
                        "search_agent_model": self.model,
                        "search_agent_turns_used": turn + 1,
                        "search_agent_tool_calls_used": tool_calls_used,
                        "search_agent_citations_count": len(citations_by_url),
                        "search_agent_answer": "".join(answer_buf),
                        "search_agent_queries": query_history,
                        "search_agent_reasoning_effort": self.reasoning_effort,
                        "search_agent_is_error": False,
                        # Explicit timings — independent of PostHog's
                        # totalLatency, which is computed from server-side
                        # event-receive timestamps and includes flush delay.
                        "wall_latency_seconds": wall_latency,
                        "time_to_first_think_token_seconds": time_to_first_think_token,
                        "time_to_first_answer_token_seconds": time_to_first_answer_token,
                    },
                )

                yield RunStreamingResponse(
                    citations=list(citations_by_url.values()),
                    is_intermediate=False,
                    complete=True,
                    metadata={
                        "answer_model_id": self.model,
                        "tool_calls_used": tool_calls_used,
                        "turns_used": turn + 1,
                        "wall_latency_seconds": wall_latency,
                        "time_to_first_think_token_seconds": time_to_first_think_token,
                        "time_to_first_answer_token_seconds": time_to_first_answer_token,
                    },
                )

            except Exception as e:  # noqa: BLE001
                self.logger.exception("Error in SearchAgent.run_streaming")
                # Same ordering rule on the error path: trace summary first,
                # then the final yield.
                capture_ai_trace_summary(
                    distinct_id=distinct_id,
                    trace_id=trace_id,
                    span_name=_truncate_for_name(request.question),
                    started_at=trace_started_at,
                    question=request.question,
                    answer="".join(answer_buf),
                    extra_properties={
                        "agent_kind": "search_agent_run",
                        "reasoning_effort": self.reasoning_effort,
                        "search_agent_model": self.model,
                        "search_agent_tool_calls_used": tool_calls_used,
                        "search_agent_citations_count": len(citations_by_url),
                        "search_agent_queries": query_history,
                        "search_agent_is_error": True,
                        "search_agent_error": str(e),
                        "wall_latency_seconds": round(time.time() - trace_started_at, 3),
                        "time_to_first_think_token_seconds": time_to_first_think_token,
                        "time_to_first_answer_token_seconds": time_to_first_answer_token,
                    },
                )
                yield RunStreamingResponse(
                    final_report=f"Error processing question: {e}",
                    citations=list(citations_by_url.values()),
                    is_intermediate=False,
                    complete=True,
                    error=str(e),
                )

        return stream


def _truncate_for_name(text: str, limit: int = 80) -> str:
    """Shorten arbitrary text for use as a PostHog span_name (trace label)."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _state_footer(
    tool_calls_used: int,
    max_tool_calls: int,
    turns_used: int,
    max_turns: int,
    citations_count: int,
    query_history: List[str],
) -> str:
    """Render the agent's current state at the bottom of every tool output.

    This is the model's continuous-awareness signal: how much budget it has
    spent, what's been collected, what queries have already been tried. We
    surface this here (in tool output) rather than mutating prior turns or
    yanking the tool away — the model can decide when to stop based on
    actual state, the way a person would.
    """
    recent = query_history[-5:]
    queries_block = (
        "; ".join(f"{i + 1}. {q}" for i, q in enumerate(recent))
        if recent
        else "(none yet)"
    )
    return (
        "<agent_state>\n"
        f"searches_used: {tool_calls_used}/{max_tool_calls}\n"
        f"turns_used: {turns_used}/{max_turns} "
        "(one turn = one model round; reaching the cap ends the run "
        "without a final answer)\n"
        f"unique_sources_collected: {citations_count}\n"
        f"recent_queries: {queries_block}\n"
        "</agent_state>"
    )


def _inter(text: str) -> RunStreamingResponse:
    return RunStreamingResponse(
        intermediate_steps=text,
        is_intermediate=True,
        complete=False,
    )


def _extract_text(response: Any) -> str:
    """Pull assistant text out of a Responses API final response."""
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                parts.append(getattr(c, "text", "") or "")
    return "".join(parts)


if __name__ == "__main__":
    import asyncio
    import sys

    DEFAULT_QUESTION = "Who won the 2025 Nobel Prize in Physics and for what work?"

    async def _main(question: str) -> None:
        from dotenv import load_dotenv

        load_dotenv()

        agent = SearchAgent(
            api_base=os.environ["ALT_LLM_API_BASE_GPT5_MINI"],
            api_key=os.environ["ALT_LLM_API_KEY_GPT5_MINI"],
            model=os.environ.get("ALT_LLM_MODEL_GPT5_MINI", "gpt-5.4-mini"),
            reasoning_effort="medium",
        )
        stream_fn = await agent.run_streaming(RunRequest(question=question))

        print(f"── QUESTION ──\n{question}")
        mode = None  # "think" or "answer"
        async for ev in stream_fn():
            if ev.is_intermediate and ev.intermediate_steps:
                if mode != "think":
                    print("\n\n── THINKING ──")
                    mode = "think"
                print(ev.intermediate_steps, end="", flush=True)
            elif not ev.is_intermediate and ev.final_report:
                if mode != "answer":
                    print("\n\n── ANSWER ──")
                    mode = "answer"
                print(ev.final_report, end="", flush=True)
            if ev.citations and ev.complete:
                print("\n\n── CITATIONS ──")
                for c in ev.citations:
                    print(f"  [{c.get('sid')}] {c.get('url')}")
                if ev.metadata:
                    print(f"\n── METADATA ──\n{ev.metadata}")

    q = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    asyncio.run(_main(q))
