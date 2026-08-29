"""tests.py - Contains unit tests for the RAG engine's API endpoints.
This test suite uses Django's APITestCase to verify the functionality of the ChatView API endpoint.
The tests cover:
1. Handling of POST requests without a query (should return 400 Bad Request).
2. Handling of POST requests with a valid query (should return 200 OK and include an answer and duration).
Make sure to run these tests using Django's test runner to ensure the integrity of your RAG engine's API before deployment.
Example command to run tests:  
python manage.py test rag_engine.tests
docker compose exec web python manage.py test rag_engine.tests
"""
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from unittest.mock import patch

class ChatViewTestCase(APITestCase):
    """Test suite for the ChatView API endpoint."""

    def setUp(self):
        # Define the API endpoint URL (Update 'chat-api' to match your urls.py name pattern)
        self.url = reverse('chat') 

    def test_post_request_without_query_returns_400(self):
        """Verifies that sending an empty payload returns a Bad Request status."""
        data = {}
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)
        self.assertEqual(response.data["error"], "No query provided")

    @patch('rag_engine.views.get_rag_service')
    def test_post_request_with_valid_query_streams_200(self, mock_get_service):
        """Verifies the endpoint streams metadata + tokens + a final duration line."""
        # 1. Stub the service so no model / network is loaded.
        fake_service = mock_get_service.return_value
        fake_service.query_stream.return_value = iter([
            '{"type": "metadata", "sources": ["Section 33"], "retrieved_contexts": ["Every person has a right to life..."]}\n',
            '{"type": "token", "text": "According to Section 33, "}\n',
            '{"type": "token", "text": "every person has a right to life."}\n',
        ])

        # 2. Fire the POST request.
        data = {"query": "What does the constitution say about the right to life?"}
        response = self.client.post(self.url, data, format='json')

        # 3. Assertions.
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        content = b"".join(response.streaming_content).decode()
        self.assertIn("Section 33", content)
        self.assertIn('"type": "done"', content)
        self.assertIn('"duration"', content)

        # Verify the service method was hit exactly once with the user query.
        fake_service.query_stream.assert_called_once_with(
            "What does the constitution say about the right to life?"
        )
