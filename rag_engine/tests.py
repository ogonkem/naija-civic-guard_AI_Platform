"""Tests for the RAG engine: the streaming chat endpoint, the synchronous
request-metrics row (Phase 2a), and the asynchronous evaluation task (Phase 2b).

    python manage.py test rag_engine.tests
"""
import json

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch

from rag_engine.eval_core import evaluate_retrieval
from rag_engine.models import EvalResult, RequestMetric
from rag_engine.tasks import evaluate_request_task


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
