"""
PostHog client for LLM observability.

Captures $ai_generation events per LLM call so we can analyze model behavior,
latency, token usage, and per-turn agent activity in the PostHog UI.

Configuration:
- POSTHOG_API_KEY: required to enable capture; if unset, all calls become no-ops.
- POSTHOG_HOST: defaults to PostHog Cloud US (`https://us.i.posthog.com`).
  Use `https://eu.i.posthog.com` for EU cloud or your self-hosted URL.
- POSTHOG_PRIVACY_MODE: "1"/"true" to drop $ai_input / $ai_output_choices
  contents and keep only metadata. Defaults to off.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from posthog import Client as PostHogClient

from tools.logging_utils import get_logger


logger = get_logger("posthog_client")

_client: Optional[PostHogClient] = None
_initialized = False


def get_posthog_client() -> Optional[PostHogClient]:
    """Return a singleton PostHog client, or None if not configured.

    The first call initializes the client from env vars; subsequent calls
    reuse the same instance. If POSTHOG_API_KEY is unset, returns None
    so call sites can short-circuit without raising.
    """
    global _client, _initialized
    if _initialized:
        return _client

    _initialized = True
    api_key = os.getenv("POSTHOG_API_KEY")
    if not api_key:
        logger.info(
            "POSTHOG_API_KEY not set; LLM observability disabled (calls will be no-ops)."
        )
        return None

    host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
    privacy_mode = os.getenv("POSTHOG_PRIVACY_MODE", "").lower() in ("1", "true", "yes")
    _client = PostHogClient(
        project_api_key=api_key,
        host=host,
        privacy_mode=privacy_mode,
        # Flush small batches frequently so events show up quickly in the UI
        # for interactive debugging; bump these for high-throughput prod.
        flush_at=10,
        flush_interval=2.0,
    )
    logger.info(
        "PostHog LLM observability enabled",
        host=host,
        privacy_mode=privacy_mode,
    )

    # Flush+shutdown the client on interpreter exit so trailing events
    # (e.g. the trace-summary emitted right before process exit) don't get
    # dropped from the in-memory buffer.
    def _shutdown() -> None:
        try:
            assert _client is not None
            _client.flush()
            _client.shutdown()
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_shutdown)
    return _client


def capture_ai_generation(
    *,
    distinct_id: str,
    trace_id: str,
    span_name: str,
    model: str,
    provider: str,
    started_at: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    is_error: bool = False,
    error: Optional[str] = None,
    http_status: Optional[int] = 200,
    input_messages: Optional[Any] = None,
    output_choices: Optional[Any] = None,
    parent_id: Optional[str] = None,
    span_id: Optional[str] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a single $ai_generation event for one LLM call.

    Safe to call when PostHog is disabled — becomes a no-op.
    """
    client = get_posthog_client()
    if client is None:
        return

    latency = time.time() - started_at
    properties: Dict[str, Any] = {
        "$ai_trace_id": trace_id,
        "$ai_span_name": span_name,
        "$ai_model": model,
        "$ai_provider": provider,
        "$ai_latency": latency,
        "$ai_http_status": http_status,
        "$ai_is_error": is_error,
    }
    if parent_id is not None:
        properties["$ai_parent_id"] = parent_id
    if span_id is not None:
        properties["$ai_span_id"] = span_id
    if input_tokens is not None:
        properties["$ai_input_tokens"] = input_tokens
    if output_tokens is not None:
        properties["$ai_output_tokens"] = output_tokens
    if reasoning_tokens is not None:
        properties["$ai_reasoning_tokens"] = reasoning_tokens
    if cached_tokens is not None:
        properties["$ai_cache_read_input_tokens"] = cached_tokens
    if error:
        properties["$ai_error"] = error
    if input_messages is not None:
        properties["$ai_input"] = input_messages
    if output_choices is not None:
        properties["$ai_output_choices"] = output_choices
    if extra_properties:
        properties.update(extra_properties)

    try:
        client.capture(
            distinct_id=distinct_id,
            event="$ai_generation",
            properties=properties,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("PostHog capture failed", error=str(e))


def capture_ai_span(
    *,
    distinct_id: str,
    trace_id: str,
    span_name: str,
    started_at: float,
    parent_id: Optional[str] = None,
    span_id: Optional[str] = None,
    is_error: bool = False,
    error: Optional[str] = None,
    input_state: Optional[Any] = None,
    output_state: Optional[Any] = None,
    cost_usd: Optional[float] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a $ai_span event for non-LLM work inside a trace.

    Use this to surface things like external HTTP calls (Brave Context),
    DB queries, or any bounded operation we want to time and inspect.
    PostHog renders these as siblings of $ai_generation events in the
    trace tree when parent_id links them to a generation's span_id.
    """
    client = get_posthog_client()
    if client is None:
        return

    properties: Dict[str, Any] = {
        "$ai_trace_id": trace_id,
        "$ai_span_name": span_name,
        "$ai_latency": time.time() - started_at,
        "$ai_is_error": is_error,
    }
    if parent_id is not None:
        properties["$ai_parent_id"] = parent_id
    if span_id is not None:
        properties["$ai_span_id"] = span_id
    if error:
        properties["$ai_error"] = error
    if input_state is not None:
        properties["$ai_input_state"] = input_state
    if output_state is not None:
        properties["$ai_output_state"] = output_state
    if cost_usd is not None:
        # PostHog rolls per-span $ai_total_cost_usd up into the trace's
        # webSearchCost (for web_search-style spans) and totalCost.
        properties["$ai_total_cost_usd"] = cost_usd
    if extra_properties:
        properties.update(extra_properties)

    try:
        client.capture(
            distinct_id=distinct_id,
            event="$ai_span",
            properties=properties,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("PostHog span capture failed", error=str(e))


def make_trace_ids(prefix: str, seed: str) -> tuple[str, str]:
    """Return (trace_id, distinct_id) for a single agent run.

    trace_id is fresh per run; distinct_id is derived from `seed` so
    repeated runs of the same input (e.g. same question) group in the UI.
    """
    trace_id = str(uuid.uuid4())
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    distinct_id = f"{prefix}-{digest}"
    return trace_id, distinct_id


def capture_responses_turn(
    *,
    distinct_id: str,
    trace_id: str,
    turn: int,
    model: str,
    provider: str,
    started_at: float,
    final_response: Any,
    input_items: Any,
    answer_text: str,
    reasoning_text: str,
    pending_calls: List[Dict[str, Any]],
    question: Optional[str] = None,
    response_id: Optional[str] = None,
    web_search_count: int = 0,
    web_search_cost_usd: float = 0.0,
    extra_properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a $ai_generation event for one OpenAI Responses-API turn.

    Pulls token usage out of `final_response.usage` (best-effort — every
    field optional) and packages the answer/reasoning text plus any
    pending tool calls into the `$ai_output_choices` payload.

    `input_items` is sent verbatim as `$ai_input` (full system + user +
    function_call_outputs) so debugging has the complete picture. The
    user's question is additionally surfaced as a top-level `question`
    property to drive listing columns / filters without depending on
    PostHog's heuristic for picking a preview from $ai_input.

    `web_search_count` / `web_search_cost_usd` describe the searches this
    turn's tool calls will trigger. PostHog only rolls $ai_generation and
    $ai_embedding costs up into a trace total — $ai_span costs are ignored
    — so search spend has to ride on the generation that requested it or it
    never reaches the trace's headline cost.
    """

    # Reshape our internal pending_calls into OpenAI's standard tool_call
    # format so PostHog's AI observability UI renders them as tool calls
    # rather than opaque JSON.
    tool_calls_std = [
        {
            "id": c.get("call_id", ""),
            "type": "function",
            "function": {
                "name": c.get("name", ""),
                "arguments": c.get("arguments", ""),
            },
        }
        for c in pending_calls
    ]

    usage = getattr(final_response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    output_details = (
        getattr(usage, "output_tokens_details", None) if usage else None
    )
    reasoning_tokens = (
        getattr(output_details, "reasoning_tokens", None)
        if output_details
        else None
    )
    input_details = (
        getattr(usage, "input_tokens_details", None) if usage else None
    )
    cached_tokens = (
        getattr(input_details, "cached_tokens", None)
        if input_details
        else None
    )

    props: Dict[str, Any] = {
        "turn": turn,
        "response_id": response_id,
        "pending_tool_calls": len(pending_calls),
        "pending_tool_names": [c.get("name", "") for c in pending_calls],
    }
    if question is not None:
        props["question"] = question
    if web_search_count:
        # Ingestion preserves a pre-set $ai_web_search_cost_usd and folds it
        # into $ai_total_cost_usd alongside the token costs. We send the cost
        # directly rather than relying on $ai_web_search_count alone, which
        # only resolves for models PostHog has a web-search price for.
        props["$ai_web_search_count"] = web_search_count
        props["$ai_web_search_cost_usd"] = web_search_cost_usd
    if extra_properties:
        props.update(extra_properties)

    capture_ai_generation(
        distinct_id=distinct_id,
        trace_id=trace_id,
        span_name=f"turn-{turn}",
        model=model,
        provider=provider,
        started_at=started_at,
        # Use the API-provided response_id (when available) as the span_id
        # so child $ai_span events for tool calls issued in this turn can
        # link via parent_id and render as nested children.
        span_id=response_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        input_messages=input_items,
        output_choices=[
            {
                "role": "assistant",
                # `None` is OpenAI-standard when the assistant message is a
                # pure tool call. Sending "" makes PostHog render an empty
                # text bubble.
                "content": answer_text if answer_text else None,
                "reasoning": reasoning_text,
                "tool_calls": tool_calls_std,
            }
        ],
        extra_properties=props,
    )


def capture_ai_trace_summary(
    *,
    distinct_id: str,
    trace_id: str,
    span_name: str,
    started_at: float,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    input_state: Optional[Any] = None,
    output_state: Optional[Any] = None,
    extra_properties: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a trace-level summary event for the full agent run.

    Uses event name '$ai_trace' so it shows up in PostHog's AI observability
    Traces view as the root of the trace tree. PostHog requires
    $ai_span_id, $ai_input_state, and $ai_output_state to recognise the
    event as a valid trace root (matching what their LangChain callback
    emits); without them the Conversation tab shows "No top-level trace
    event". If `question` / `answer` are provided they auto-populate
    input/output states.
    """
    client = get_posthog_client()
    if client is None:
        return

    effective_input = input_state
    if effective_input is None and question is not None:
        effective_input = {"question": question}

    effective_output = output_state
    if effective_output is None and answer is not None:
        effective_output = {"answer": answer}

    properties: Dict[str, Any] = {
        "$ai_trace_id": trace_id,
        # PostHog's trace-root convention: the trace event's span_id
        # equals the trace_id, marking it as the root of the tree.
        "$ai_span_id": trace_id,
        "$ai_span_name": span_name,
        "$ai_latency": time.time() - started_at,
    }
    if effective_input is not None:
        properties["$ai_input_state"] = effective_input
    if effective_output is not None:
        properties["$ai_output_state"] = effective_output
    if question is not None:
        properties["question"] = question
    if extra_properties:
        properties.update(extra_properties)

    try:
        client.capture(
            distinct_id=distinct_id,
            event="$ai_trace",
            properties=properties,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("PostHog trace capture failed", error=str(e))
