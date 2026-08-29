"""Audit-log middleware for the gateway.

Plain Django middleware (not DRF) - it sits outside the view and sees the
final status code for every /api/ request, including the 401/429 that never
reach the view. One RequestAuditLog row per request, via the ORM.

The row's `request_id` is read from the `X-Request-ID` response header that
ChatView sets from its RequestMetrics; that's what joins the audit log to the
metrics log. 401/429 requests have no metrics row and no request_id.
"""

import logging

from .models import ApiKey, RequestAuditLog

logger = logging.getLogger(__name__)

AUDITED_PREFIX = "/api/"


class AuditLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if request.path.startswith(AUDITED_PREFIX):
            try:
                self._write(request, response)
            except Exception:  # auditing must never break the response
                logger.exception("audit log write failed for %s %s", request.method, request.path)
        return response

    @staticmethod
    def _write(request, response):
        raw_key = request.headers.get("X-API-Key", "")
        api_key = ApiKey.objects.filter(key=raw_key).first() if raw_key else None

        RequestAuditLog.objects.create(
            api_key=api_key,
            api_key_owner=(api_key.owner if api_key else ""),
            api_key_hint=("" if api_key or not raw_key else raw_key[:12]),
            method=request.method,
            endpoint=request.path,
            status_code=response.status_code,
            # Set by ChatView from its RequestMetrics; absent for 401/429.
            request_id=response.get("X-Request-ID") or None,
        )
