"""Rate limiting for the gateway, keyed on the API key (not a Django user).

Uses DRF's SimpleRateThrottle machinery. The default rate comes from
REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"]["api_key"]; an ApiKey row with
`requests_per_minute` set overrides it for that key.
"""

from rest_framework.throttling import SimpleRateThrottle

from .models import ApiKey


class ApiKeyRateThrottle(SimpleRateThrottle):
    scope = "api_key"

    def get_cache_key(self, request, view):
        api_key = getattr(request, "auth", None)
        if not isinstance(api_key, ApiKey):
            return None  # unauthenticated -> auth layer already rejected it
        return f"throttle_api_key_{api_key.pk}"

    def allow_request(self, request, view):
        api_key = getattr(request, "auth", None)
        if isinstance(api_key, ApiKey) and api_key.requests_per_minute:
            self.rate = f"{api_key.requests_per_minute}/min"
            self.num_requests, self.duration = self.parse_rate(self.rate)
        return super().allow_request(request, view)
