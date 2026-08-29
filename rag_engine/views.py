import json
import time

from django.http import StreamingHttpResponse
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

# Lazily-created singleton. Building RagService imports torch / sentence-
# transformers and touches the network, so we must NOT do it at import time -
# and we defer the `services` import itself so manage.py commands, migrations
# and tests never pay that ~15s import cost.
_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        from .services import RagService
        _rag_service = RagService()
    return _rag_service


def chat_page(request):
    """Renders the chat interface for user interaction with the RAG engine."""
    return render(request, 'chat.html')


class ChatView(APIView):
    """Streams RAG answers as newline-delimited JSON.

    Emits one ``{"type": "metadata", ...}`` line, then ``{"type": "token"}``
    lines as the LLM generates, then a final ``{"type": "done", "duration": s}``.
    Streaming means the user sees the first token in ~1s instead of staring at
    a spinner until the whole answer is generated.
    """

    def post(self, request):
        user_query = request.data.get("query")

        if not user_query:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)

        service = get_rag_service()

        def event_stream():
            start_time = time.time()
            for line in service.query_stream(user_query):
                yield line
            duration = round(time.time() - start_time, 4)
            yield json.dumps({"type": "done", "duration": duration}) + "\n"

        response = StreamingHttpResponse(event_stream(), content_type="application/x-ndjson")
        # Defeat proxy/browser buffering so tokens arrive as they are produced.
        response["X-Accel-Buffering"] = "no"
        response["Cache-Control"] = "no-cache"
        return response
