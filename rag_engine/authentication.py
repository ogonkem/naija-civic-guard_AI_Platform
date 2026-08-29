"""API-key authentication for the DRF gateway.

Callers pass `X-API-Key: <key>`. A missing / unknown / inactive key raises
AuthenticationFailed, which - because authenticate_header() returns a value -
DRF renders as 401 (not 403).
"""

from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from .models import ApiKey

HEADER = "X-API-Key"


class ApiKeyUser:
    """Minimal principal so DRF's IsAuthenticated is satisfied without a
    django.contrib.auth User. `request.auth` holds the ApiKey instance."""

    is_authenticated = True
    is_active = True
    is_anonymous = False
    is_staff = False

    def __init__(self, api_key: ApiKey):
        self.api_key = api_key
        self.pk = api_key.pk
        self.id = api_key.pk
        self.username = api_key.owner

    def __str__(self):
        return f"apikey:{self.api_key.owner}"


class ApiKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        raw = request.META.get("HTTP_X_API_KEY") or request.headers.get(HEADER)
        if not raw:
            raise AuthenticationFailed(f"No API key. Send the {HEADER} header.")
        try:
            api_key = ApiKey.objects.get(key=raw)
        except ApiKey.DoesNotExist:
            raise AuthenticationFailed("Invalid API key.")
        if not api_key.is_active:
            raise AuthenticationFailed("API key is inactive.")
        return (ApiKeyUser(api_key), api_key)

    def authenticate_header(self, request):
        # Non-None -> DRF uses 401 for auth failures on this view.
        return HEADER
