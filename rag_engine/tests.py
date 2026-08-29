"""Tests for the RAG engine: the streaming chat endpoint, the synchronous
request-metrics row (Phase 2a), and the asynchronous evaluation task (Phase 2b).

    python manage.py test rag_engine.tests
"""
import json

from django.test import SimpleTestCase
from django.urls import reverse
from langchain_core.documents import Document
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch

from rag_engine.eval_core import evaluate_retrieval
from rag_engine.graph import build_agent_graph, run_agent
from rag_engine.models import EvalResult, RequestMetric
from rag_engine.sections import find_section_references
from rag_engine.tasks import evaluate_request_task


class _FakeService:
    """Stand-in for RagService: no LLM, no ChromaDB - just scripted behaviour
    so the LangGraph agent can be exercised deterministically."""

    def __init__(self, corpus, classify="direct_lookup", verify_script=None):
        # corpus: {section_label: page_content}
        self.corpus = corpus
        self._classify = classify
        # verify_script: list of (adequate, reformulated) returned in order
        self._verify_script = list(verify_script or [(True, None)])
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


class ChatViewTestCase(APITestCase):
    """The streaming endpoint + the synchronous metrics row."""

    def setUp(self):
        self.url = reverse('chat')
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
