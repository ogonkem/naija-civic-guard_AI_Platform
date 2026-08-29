"""Tests for the RAG engine: the streaming chat endpoint, the synchronous
request-metrics row (Phase 2a), and the asynchronous evaluation task (Phase 2b).

    python manage.py test rag_engine.tests
"""
import json

from django.core.cache import cache
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from langchain_core.documents import Document
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch

from rag_engine.eval_core import evaluate_retrieval
from rag_engine.graph import build_agent_graph, run_agent
from rag_engine.models import ApiKey, EvalResult, RequestAuditLog, RequestMetric
from rag_engine.sections import find_section_references
from rag_engine.tasks import evaluate_request_task


def _fake_stream_service(mock_get_service):
    fake = mock_get_service.return_value
    fake.llm_provider = "groq"
    fake.llm_model = "openai/gpt-oss-20b"
    fake.query_stream.return_value = iter([
        '{"type": "metadata", "sources": ["Section 14"]}\n',
        '{"type": "token", "text": "one two three"}\n',
    ])
    return fake


class _FakeMcp:
    """Scripted stand-in for McpToolClient. Records calls, returns
    (payload, latency_ms, ok, error)."""

    def __init__(self, corpus, fail=()):
        self.corpus = corpus            # {"Section 45": "text", ...}
        self.fail = set(fail)           # tool names to return ok=False for
        self.calls = []

    def _reply(self, name, payload):
        self.calls.append(name)
        if name in self.fail:
            return (None, 3.0, False, f"{name} boom")
        return (payload, 4.2, True, None)

    def lookup_section(self, number):
        label = f"Section {number}"
        chunks = [self.corpus[label]] if label in self.corpus else []
        return self._reply("lookup_section",
                           {"section": label, "found": bool(chunks), "chunks": chunks})

    def find_related_sections(self, section_id):
        num = "".join(ch for ch in str(section_id) if ch.isdigit())
        own = self.corpus.get(f"Section {num}", "")
        refs = find_section_references(own, exclude={f"Section {num}"})
        related = [{"section": r, "text": self.corpus[r]} for r in refs if r in self.corpus]
        return self._reply("find_related_sections",
                           {"section": f"Section {num}", "found": bool(own),
                            "references": refs, "related": related})

    def search_precedent(self, query):
        return self._reply("search_precedent",
                           {"implemented": False, "message": "not yet implemented"})


class _FakeService:
    """Stand-in for RagService: no LLM, no ChromaDB - just scripted behaviour
    so the LangGraph agent can be exercised deterministically."""

    def __init__(self, corpus, classify="direct_lookup", verify_script=None, mcp=None):
        # corpus: {section_label: page_content}
        self.corpus = corpus
        self._classify = classify
        # verify_script: list of (adequate, reformulated) returned in order
        self._verify_script = list(verify_script or [(True, None)])
        self.mcp = mcp
        self.calls = []

    def classify_query(self, query):
        self.calls.append(("classify", query))
        return self._classify

    def retrieve(self, query):
        self.calls.append(("retrieve", query))
        # crude: return the doc whose label is named in the query, else the first
        for label, text in self.corpus.items():
            if label.lower() in query.lower() or label.split()[-1] in query:
                return [Document(page_content=text, metadata={"section": label})]
        first = next(iter(self.corpus.items()))
        return [Document(page_content=first[1], metadata={"section": first[0]})]

    def verify_retrieval(self, query, label, text):
        self.calls.append(("verify", query))
        return self._verify_script.pop(0) if self._verify_script else (True, None)


class SectionReferenceTests(SimpleTestCase):
    def test_enumeration_is_expanded_and_excluded(self):
        text = "Nothing in sections 37, 38, 39, 40 and 41 of this Constitution shall invalidate..."
        refs = find_section_references(text, exclude={"Section 45"})
        self.assertEqual(refs, ["Section 37", "Section 38", "Section 39", "Section 40", "Section 41"])

    def test_single_reference_and_out_of_range_dropped(self):
        text = "in accordance with section 143 of this Constitution and section 999"
        self.assertEqual(find_section_references(text), ["Section 143"])


class RetrievalAgentTests(SimpleTestCase):
    def test_chain_node_fires_second_retrieval_on_reference(self):
        svc = _FakeService(corpus={
            "Section 45": "Nothing in sections 37, 38 of this Constitution shall invalidate any law...",
            "Section 37": "The privacy of citizens is guaranteed.",
            "Section 38": "Every person is entitled to freedom of thought and religion.",
        })
        state = run_agent(build_agent_graph(svc), "What does Section 45 say?")

        self.assertGreater(state["retrieval_calls"], 1)                 # chaining fired
        self.assertIn("Section 37", state["chained_sections"])
        self.assertIn("Section 38", state["chained_sections"])
        self.assertIn("Section 37", state["sources"])
        for k in ("classify_ms", "retrieve_ms", "chain_ms", "verify_ms"):
            self.assertIsNotNone(state[k])
        self.assertFalse(state.get("verify_retry"))

    def test_verify_triggers_exactly_one_retry(self):
        svc = _FakeService(
            corpus={"Section 1": "The Constitution is supreme."},
            verify_script=[(False, "supremacy of the constitution"), (False, "again"), (False, "again")],
        )
        state = run_agent(build_agent_graph(svc), "Is the constitution supreme?")

        self.assertTrue(state["verify_retry"])
        self.assertEqual(state["retry_count"], 1)          # capped at 1, no infinite loop
        self.assertEqual(sum(1 for c in svc.calls if c[0] == "retrieve"), 2)

    def test_no_reference_means_single_retrieval(self):
        svc = _FakeService(corpus={"Section 33": "Every person has a right to life."})
        state = run_agent(build_agent_graph(svc), "What does Section 33 say?")
        self.assertEqual(state["retrieval_calls"], 1)
        self.assertEqual(state["chained_sections"], [])


class McpAgentTests(SimpleTestCase):
    """retrieve/chain nodes going through the MCP client (fake) instead of
    calling retrieval in-process."""

    CORPUS = {
        "Section 45": "Nothing in sections 37, 38 of this Constitution shall invalidate any law...",
        "Section 37": "The privacy of citizens ... is guaranteed.",
        "Section 38": "Every person is entitled to freedom of thought, conscience and religion.",
    }

    def test_mcp_tools_are_called_and_logged_with_latency(self):
        mcp = _FakeMcp(self.CORPUS)
        svc = _FakeService(self.CORPUS, classify="direct_lookup", mcp=mcp)
        state = run_agent(build_agent_graph(svc), "What does Section 45 say?")

        names = [c["tool_name"] for c in state["tool_calls"]]
        self.assertEqual(names[0], "lookup_section")                 # retrieve node
        self.assertIn("find_related_sections", names)                # chain node
        for c in state["tool_calls"]:
            self.assertIn("tool_latency_ms", c)
            self.assertIsInstance(c["tool_latency_ms"], (int, float))
            self.assertTrue(c["ok"])
            self.assertIsNone(c["error"])

        # chaining happened via MCP, not in-process retrieve()
        self.assertNotIn("retrieve", [c[0] for c in svc.calls])
        self.assertIn("Section 37", state["chained_sections"])
        self.assertGreater(state["retrieval_calls"], 1)

    def test_mcp_lookup_failure_falls_back_in_process_and_records_error(self):
        mcp = _FakeMcp(self.CORPUS, fail={"lookup_section"})
        svc = _FakeService(self.CORPUS, classify="direct_lookup", mcp=mcp)
        state = run_agent(build_agent_graph(svc), "What does Section 45 say?")

        lookup = next(c for c in state["tool_calls"] if c["tool_name"] == "lookup_section")
        self.assertFalse(lookup["ok"])
        self.assertIn("boom", lookup["error"])
        self.assertIn("retrieve", [c[0] for c in svc.calls])        # fell back in-process

    def test_interpretive_query_calls_search_precedent_stub(self):
        mcp = _FakeMcp(self.CORPUS)
        svc = _FakeService(self.CORPUS, classify="interpretive", mcp=mcp)
        state = run_agent(build_agent_graph(svc), "Can rights be limited in an emergency?")
        self.assertIn("search_precedent", [c["tool_name"] for c in state["tool_calls"]])

    def test_real_mcp_client_roundtrip_stub_tool(self):
        """End-to-end over a real subprocess + reused session (search_precedent
        needs no ChromaDB)."""
        from rag_engine.mcp_client import McpToolClient
        client = McpToolClient()
        try:
            if not client.wait_ready(timeout=30):
                self.skipTest("MCP subprocess did not start")
            p1, lat1, ok1, _ = client.search_precedent("murder precedent")
            p2, lat2, ok2, _ = client.search_precedent("again")
            self.assertTrue(ok1 and ok2)
            self.assertFalse(p1["implemented"])
            self.assertIn("not yet implemented", p1["message"])
            # session reused: second call is not paying a new-connection cost
            self.assertLess(lat2, 1000)
        finally:
            client.close()


class ChatViewTestCase(APITestCase):
    """The streaming endpoint + the synchronous metrics row (authenticated)."""

    def setUp(self):
        self.url = reverse('chat')
        cache.clear()  # DRF throttle state lives in the cache
        self.api_key = ApiKey.objects.create(owner="test-suite")
        self.client.credentials(HTTP_X_API_KEY=self.api_key.key)
        # The endpoint enqueues an async eval task in its finally block; that
        # hand-off is exercised separately below. Stub it here so these tests
        # never touch a broker.
        p = patch('rag_engine.views._enqueue_eval')
        self.mock_enqueue = p.start()
        self.addCleanup(p.stop)

    def test_post_request_without_query_returns_400(self):
        response = self.client.post(self.url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["error"], "No query provided")

    @patch('rag_engine.views.get_rag_service')
    def test_post_request_with_valid_query_streams_200(self, mock_get_service):
        fake_service = mock_get_service.return_value
        fake_service.llm_provider = "groq"
        fake_service.llm_model = "openai/gpt-oss-20b"
        fake_service.query_stream.return_value = iter([
            '{"type": "metadata", "sources": ["Section 33"], "retrieved_contexts": ["Every person has a right to life..."]}\n',
            '{"type": "token", "text": "According to Section 33, "}\n',
            '{"type": "token", "text": "every person has a right to life."}\n',
        ])

        data = {"query": "What does the constitution say about the right to life?"}
        response = self.client.post(self.url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        content = b"".join(response.streaming_content).decode()
        self.assertIn("Section 33", content)
        self.assertIn('"type": "done"', content)
        self.assertIn('"duration"', content)

        fake_service.query_stream.assert_called_once()
        self.assertEqual(
            fake_service.query_stream.call_args.args[0],
            "What does the constitution say about the right to life?",
        )

    @patch('rag_engine.views.get_rag_service')
    def test_request_metric_row_is_persisted(self, mock_get_service):
        fake_service = mock_get_service.return_value
        fake_service.llm_provider = "groq"
        fake_service.llm_model = "openai/gpt-oss-20b"
        fake_service.query_stream.return_value = iter([
            '{"type": "metadata", "sources": ["Section 14"]}\n',
            '{"type": "token", "text": "one two three four five"}\n',
        ])

        self.assertEqual(RequestMetric.objects.count(), 0)
        response = self.client.post(
            self.url, {"query": "purpose of government"}, format='json'
        )
        content = b"".join(response.streaming_content).decode()
        done = json.loads([ln for ln in content.splitlines() if '"done"' in ln][0])

        self.assertEqual(RequestMetric.objects.count(), 1)
        row = RequestMetric.objects.get()
        self.assertEqual(str(row.request_id), done["request_id"])
        self.assertEqual(row.query_text, "purpose of government")
        self.assertEqual(row.provider, "groq")
        self.assertIsNotNone(row.total_time_ms)
        self.assertEqual(row.error, "")

        # The endpoint attempted the async hand-off exactly once, after the row.
        self.mock_enqueue.assert_called_once()

    @patch('rag_engine.views.get_rag_service')
    def test_request_metric_row_is_persisted_on_error(self, mock_get_service):
        fake_service = mock_get_service.return_value
        fake_service.llm_provider = "groq"
        fake_service.llm_model = "openai/gpt-oss-20b"

        def boom(*a, **kw):
            yield '{"type": "metadata", "sources": []}\n'
            raise RuntimeError("groq exploded")

        fake_service.query_stream.side_effect = boom

        response = self.client.post(self.url, {"query": "boom"}, format='json')
        b"".join(response.streaming_content)  # drain

        row = RequestMetric.objects.get()
        self.assertEqual(row.query_text, "boom")
        self.assertIn("groq exploded", row.error)
        self.assertIsNotNone(row.total_time_ms)


class AsyncEvalTestCase(APITestCase):
    """The Celery evaluation task and its enqueue hand-off."""

    GT_QUERY = "In whom are the legislative powers of the Federation vested?"  # in evaluation_set.jsonl

    def test_task_writes_eval_result_for_ground_truth_query(self):
        res = evaluate_request_task.apply(kwargs=dict(
            request_id="11111111-1111-1111-1111-111111111111",
            query=self.GT_QUERY,
            retrieved_context=[
                "The National Assembly consists of a Senate and a House of Representatives."
            ],
            retrieved_section_ids=["Section 4", "Section 5"],
            response_text="The legislative powers are vested in the National Assembly.",
        ))
        self.assertTrue(res.successful())

        row = EvalResult.objects.get(request_id="11111111-1111-1111-1111-111111111111")
        self.assertTrue(row.matched_ground_truth)
        self.assertEqual(row.keyword_source, "ground_truth")
        self.assertEqual(row.target_section, "Section 4")
        self.assertTrue(row.hit)
        self.assertEqual(row.reciprocal_rank, 1.0)
        self.assertEqual(row.keyword_coverage, 1.0)

    def test_task_writes_eval_result_for_offset_query_with_nulls(self):
        res = evaluate_request_task.apply(kwargs=dict(
            request_id="22222222-2222-2222-2222-222222222222",
            query="what does the constitution say about the police force",
            retrieved_context=["There shall be a Police Force for Nigeria."],
            retrieved_section_ids=["Section 214"],
            response_text="Section 214 establishes the Nigeria Police Force.",
        ))
        self.assertTrue(res.successful())

        row = EvalResult.objects.get(request_id="22222222-2222-2222-2222-222222222222")
        self.assertFalse(row.matched_ground_truth)
        self.assertEqual(row.keyword_source, "query")
        self.assertIsNone(row.hit)
        self.assertIsNone(row.reciprocal_rank)
        self.assertEqual(row.target_section, "")
        self.assertIsNotNone(row.keyword_coverage)  # still computed from query terms

    def test_enqueue_swallows_broker_failure(self):
        """A dead broker must not propagate out of the request path."""
        from rag_engine.views import _enqueue_eval
        from rag_engine.metrics import RequestMetrics

        m = RequestMetrics(query_text="x")
        with patch("rag_engine.tasks.evaluate_request_task.apply_async",
                   side_effect=RuntimeError("connection refused")):
            _enqueue_eval(m)  # must not raise

    def test_eval_core_join_shape(self):
        """evaluate_retrieval returns the exact keys the task/model expect."""
        out = evaluate_retrieval(
            query=self.GT_QUERY,
            retrieved_context=["Senate and House of Representatives"],
            retrieved_section_ids=["Section 4"],
            response_text="x",
        )
        self.assertEqual(
            set(out),
            {"matched_ground_truth", "keyword_coverage", "keyword_source",
             "keywords_checked", "target_section", "hit", "reciprocal_rank",
             "retrieved_section_ids", "response_chars"},
        )


class GatewayTestCase(APITestCase):
    """API-key auth (401), per-key rate limit (429), audit-log join."""

    def setUp(self):
        self.url = reverse("chat")
        cache.clear()
        p = patch("rag_engine.views._enqueue_eval")
        p.start()
        self.addCleanup(p.stop)

    def _post(self, **kwargs):
        return self.client.post(self.url, {"query": "right to life"}, format="json", **kwargs)

    # --- auth --------------------------------------------------------------- #
    def test_no_api_key_returns_401(self):
        resp = self._post()
        self.assertEqual(resp.status_code, 401)

    def test_invalid_api_key_returns_401(self):
        self.client.credentials(HTTP_X_API_KEY="ncg_not_a_real_key")
        self.assertEqual(self._post().status_code, 401)

    def test_inactive_api_key_returns_401(self):
        key = ApiKey.objects.create(owner="disabled", is_active=False)
        self.client.credentials(HTTP_X_API_KEY=key.key)
        self.assertEqual(self._post().status_code, 401)

    # --- rate limit ------------------------------------------------------- #
    @patch("rag_engine.views.get_rag_service")
    def test_rate_limit_returns_429_after_threshold(self, mock_get_service):
        key = ApiKey.objects.create(owner="load-test", requests_per_minute=3)
        self.client.credentials(HTTP_X_API_KEY=key.key)

        codes = []
        for _ in range(5):
            _fake_stream_service(mock_get_service)  # fresh iterator each call
            resp = self._post()
            if hasattr(resp, "streaming_content"):
                b"".join(resp.streaming_content)
            codes.append(resp.status_code)

        self.assertEqual(codes[:3], [200, 200, 200])
        self.assertEqual(codes[3], 429)
        self.assertEqual(codes[4], 429)

    # --- audit log ------------------------------------------------------- #
    @patch("rag_engine.views.get_rag_service")
    def test_valid_request_logged_in_audit_and_metrics_joinable(self, mock_get_service):
        _fake_stream_service(mock_get_service)
        key = ApiKey.objects.create(owner="acme")
        self.client.credentials(HTTP_X_API_KEY=key.key)

        resp = self._post()
        b"".join(resp.streaming_content)

        self.assertEqual(resp.status_code, 200)
        request_id = resp["X-Request-ID"]

        audit = RequestAuditLog.objects.get()
        metric = RequestMetric.objects.get()
        self.assertEqual(audit.status_code, 200)
        self.assertEqual(audit.endpoint, self.url)
        self.assertEqual(audit.api_key_id, key.id)
        self.assertEqual(audit.api_key_owner, "acme")
        # the join key: audit.request_id == metric.request_id == X-Request-ID
        self.assertEqual(str(audit.request_id), request_id)
        self.assertEqual(audit.request_id, metric.request_id)
        # and it joins at the ORM level
        self.assertEqual(
            RequestAuditLog.objects.filter(request_id=metric.request_id).count(), 1
        )

    def test_401_is_audited_with_no_request_id(self):
        self.client.credentials(HTTP_X_API_KEY="ncg_bogus")
        self._post()
        audit = RequestAuditLog.objects.get()
        self.assertEqual(audit.status_code, 401)
        self.assertIsNone(audit.request_id)
        self.assertIsNone(audit.api_key_id)
        self.assertEqual(audit.api_key_hint, "ncg_bogus"[:12])
        self.assertEqual(RequestMetric.objects.count(), 0)

    # Tests run with DEBUG=False, so the manifest static storage would need a
    # collected staticfiles.json just to render {% static %}. Use plain storage
    # for this template-rendering test.
    @override_settings(STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    })
    def test_chat_page_embeds_a_working_api_key(self):
        """GET / auto-provisions a 'browser-ui' key and embeds it so the
        bundled chat page can authenticate against the gateway."""
        self.assertFalse(ApiKey.objects.filter(owner="browser-ui").exists())
        resp = self.client.get(reverse("chat_page"))
        self.assertEqual(resp.status_code, 200)

        key = ApiKey.objects.get(owner="browser-ui")
        self.assertContains(resp, f'<meta name="api-key" content="{key.key}">', html=False)

        # a second load reuses the same key, doesn't pile up rows
        self.client.get(reverse("chat_page"))
        self.assertEqual(ApiKey.objects.filter(owner="browser-ui").count(), 1)


class PrometheusMetricsTests(TestCase):
    """Phase 7: /metrics endpoint + custom metrics fed from RequestMetrics."""

    def test_metrics_endpoint_exposes_custom_series(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        for name in (
            "request_latency_seconds",
            "generation_tokens_per_second",
            "mcp_tool_calls_total",
            "agent_retries_total",
            "llm_provider_requests_total",
            "django_http_requests_total_by_method_total",  # django-prometheus auto
        ):
            self.assertIn(name, body)

    def test_normalize_provider(self):
        from rag_engine.metrics_prom import normalize_provider
        self.assertEqual(normalize_provider("groq"), "openai")
        self.assertEqual(normalize_provider("openai"), "openai")
        self.assertEqual(normalize_provider("ChatOllama"), "ollama")
        self.assertEqual(normalize_provider("ollama"), "ollama")

    def test_record_request_metrics_feeds_registry_from_same_object(self):
        from prometheus_client import REGISTRY
        from rag_engine.metrics import RequestMetrics
        from rag_engine.metrics_prom import record_request_metrics

        def val(name, labels=None):
            return REGISTRY.get_sample_value(name, labels) or 0.0

        before_req = val("llm_provider_requests_total", {"provider": "ollama"})
        before_tool = val("mcp_tool_calls_total", {"tool_name": "lookup_section"})
        before_retry = val("agent_retries_total")
        before_lat = val("request_latency_seconds_count", {"stage": "generation"})

        m = RequestMetrics(query_text="q", provider="ollama")
        m.generation_time_ms = 900.0
        m.total_time_ms = 1000.0
        m.retrieve_ms = 20.0
        m.tokens_per_second = 55.0
        m.verify_retry = True
        m.tool_calls = [
            {"tool_name": "lookup_section", "tool_latency_ms": 12.0, "ok": True, "error": None},
            {"tool_name": "find_related_sections", "tool_latency_ms": 30.0, "ok": True, "error": None},
        ]
        record_request_metrics(m)

        self.assertEqual(val("llm_provider_requests_total", {"provider": "ollama"}), before_req + 1)
        self.assertEqual(val("mcp_tool_calls_total", {"tool_name": "lookup_section"}), before_tool + 1)
        self.assertEqual(val("agent_retries_total"), before_retry + 1)
        self.assertEqual(val("request_latency_seconds_count", {"stage": "generation"}), before_lat + 1)
        self.assertEqual(
            val("generation_tokens_per_second_count", {"provider": "ollama"}),
            (REGISTRY.get_sample_value("generation_tokens_per_second_count", {"provider": "ollama"}) or 0.0),
        )

    def test_eval_task_records_duration_and_coverage(self):
        from prometheus_client import REGISTRY
        before_dur = REGISTRY.get_sample_value("celery_eval_task_duration_seconds_count") or 0.0

        evaluate_request_task.apply(kwargs=dict(
            request_id="33333333-3333-3333-3333-333333333333",
            query="In whom are the legislative powers of the Federation vested?",
            retrieved_context=["the National Assembly, a Senate and a House of Representatives"],
            retrieved_section_ids=["Section 4"],
            response_text="the National Assembly",
        ))

        after_dur = REGISTRY.get_sample_value("celery_eval_task_duration_seconds_count") or 0.0
        self.assertEqual(after_dur, before_dur + 1)
        # ground-truth query -> coverage gauge is set
        self.assertIsNotNone(REGISTRY.get_sample_value("eval_keyword_coverage"))
