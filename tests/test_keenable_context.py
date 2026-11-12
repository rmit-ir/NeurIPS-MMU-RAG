import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
from keenable import AsyncKeenable, KeenableAPIError

from tools import keenable_context as service
from systems.search_agent import functions


class KeenableTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_parameters_mapping_and_error_policy(self):
        for statuses in ([200], [429, 200], [429, 429], [401], [402]):
            with self.subTest(statuses=statuses):
                requests = []
                def receive(request):
                    requests.append(request)
                    self.assertEqual(request.headers["X-API-Key"], "synthetic-keenable")
                    body = json.loads(request.content)
                    self.assertEqual(body["max_results"], 5)
                    self.assertEqual(body["snippet_max_length"], 3276)
                    self.assertEqual(body["mode"], "pro")
                    self.assertEqual(body["published_after"], (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat())
                    return httpx.Response(statuses[len(requests)-1], json={"results": [
                        {"url": "https://example.com/page", "title": "Page", "snippet": "Page content", "published_at": "2026-11-01"},
                        {"url": "https://cards.keenable.ai/w/1", "title": "Card", "snippet": "expires"},
                        {"url": "javascript:alert(1)", "title": "Unsafe", "snippet": "unsafe"},
                    ]})
                async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as client:
                    with patch.dict(os.environ, {"KEENABLE_API_KEY": "synthetic-keenable"}), patch.object(service, "AsyncKeenable", side_effect=lambda **kw: AsyncKeenable(client=client, **kw)), patch.object(service.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                        if statuses[-1] == 200:
                            results = await service.keenable_context("example", count=5, max_tokens=4096, max_urls=5, freshness="pw")
                            self.assertEqual(len(results), 1)
                            self.assertEqual(results[0].type, "keenable")
                            self.assertEqual(results[0].sid, "1")
                            self.assertEqual(results[0].date, "2026-11-01")
                            self.assertEqual(results[0].metadata["provider"], "keenable")
                        else:
                            with self.assertRaises(KeenableAPIError) as error:
                                await service.keenable_context("example", count=5, max_tokens=4096, max_urls=5, freshness="pw")
                            self.assertEqual(error.exception.status_code, statuses[-1])
                        self.assertEqual(len(requests), len(statuses))
                        self.assertEqual(sleep.await_count, int(statuses[0] == 429))
                        if statuses[0] == 429:
                            sleep.assert_awaited_once_with(1)

    async def test_agent_uses_keenable_and_keeps_stable_citations(self):
        result = service.SearchResult(type="keenable", text="Page content", id="keenable_1", sid="1", url="https://example.com", token_count=2, metadata={"title": "Page"}, dump=None, date=None, file_path=None, language=None, language_score=None, score=None)
        citations = {}
        with patch.object(functions, "keenable_context", new_callable=AsyncMock, return_value=[result]) as search, patch.object(functions, "capture_ai_span") as capture:
            first = await functions.exec_web_search({"query": "example"}, citations, trace_id="synthetic", distinct_id="synthetic")
            second = await functions.exec_web_search({"query": "example"}, citations)
            self.assertEqual(first, second)
            self.assertEqual(len(citations), 1)
            self.assertEqual(citations["https://example.com"]["sid"], "1")
            self.assertEqual(search.await_args.kwargs["count"], 5)
            self.assertEqual(search.await_args.kwargs["max_tokens"], 4096)
            self.assertEqual(capture.call_args.kwargs["span_name"], "keenable_search")
            self.assertIsNone(capture.call_args.kwargs["cost_usd"])

    async def test_auth_and_billing_stop_the_agent(self):
        for status in (401, 402):
            with patch.object(functions, "keenable_context", new_callable=AsyncMock, side_effect=KeenableAPIError("synthetic", status)) as search:
                with self.assertRaisesRegex(RuntimeError, str(status)):
                    await functions.exec_web_search({"query": "example"}, {})
                search.assert_awaited_once()

    async def test_missing_key_is_not_a_keyless_fallback(self):
        with patch.dict(os.environ, {"KEENABLE_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "KEENABLE_API_KEY"):
                await service.keenable_context("example")
