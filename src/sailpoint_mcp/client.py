"""Authenticated SailPoint SDK client, shared by every tool.

The SDK's `Configuration` object fetches an OAuth access token from
`{SAIL_BASE_URL}/oauth/token` the moment it is constructed, using the
client-credentials grant, and then holds that token for its lifetime. It does not
refresh it. Since an MCP server is long-lived, this module:

  * builds the client lazily, on the first tool call (so the server still starts
    and can report a clear error when credentials are wrong),
  * caches it across calls (one token fetch, not one per tool call),
  * rebuilds it when the token ages out or when the API answers 401.

Tool authors should not construct `Configuration` themselves -- call
`call_sailpoint()` and you get auth, caching, and retry for free.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
import time
from typing import Callable, TypeVar

from sailpoint import ApiClient
from sailpoint.configuration import Configuration, ConfigurationParams
from sailpoint.exceptions import ApiException, UnauthorizedException

from .config import ConfigError, SailPointSettings, load_settings

log = logging.getLogger(__name__)

T = TypeVar("T")

# ISC access tokens are long-lived, but re-authenticating periodically keeps a
# server that runs for days from wedging on an expired token.
_TOKEN_MAX_AGE_SECONDS = 30 * 60

_lock = threading.Lock()
_api_client: ApiClient | None = None
_created_at: float = 0.0


class SailPointAuthError(RuntimeError):
    """Raised when the tenant will not issue an access token."""


def _build_client(settings: SailPointSettings) -> ApiClient:
    params = ConfigurationParams()
    params.base_url = settings.base_url
    params.client_id = settings.client_id
    params.client_secret = settings.client_secret

    # Constructing Configuration performs the token request. When that fails the
    # SDK prints the reason to stdout and leaves access_token as None, so every
    # later call dies with an unrelated TypeError deep in the request layer. Two
    # things are needed here: capture that stdout (an MCP server must never write
    # to stdout -- it is the protocol channel) and turn the silent failure into a
    # real exception.
    sdk_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(sdk_output):
            configuration = Configuration(params)
    except Exception as exc:
        raise SailPointAuthError(
            f"Could not reach {settings.base_url} to authenticate: {exc}"
        ) from exc

    # Some ISC APIs the tools need (e.g. access-request-administration) are
    # flagged experimental, and the SDK refuses to call them unless this is set.
    # It only unlocks those endpoints; it changes nothing about the ones below.
    configuration.experimental = True
    configuration.suppress_experimental_warnings = True

    detail = sdk_output.getvalue().strip()
    if not configuration.access_token:
        raise SailPointAuthError(
            f"SailPoint would not issue an access token for tenant "
            f"'{settings.tenant}'. Check SAIL_CLIENT_ID / SAIL_CLIENT_SECRET and "
            f"that SAIL_BASE_URL is correct. "
            + (f"Tenant said: {detail}" if detail else "")
        )
    if detail:
        log.warning("SailPoint SDK reported: %s", detail)

    log.info("Authenticated to SailPoint tenant %s", settings.tenant)
    return ApiClient(configuration)


def get_api_client(*, force_refresh: bool = False) -> ApiClient:
    """Return the shared, authenticated `ApiClient`, (re)authenticating as needed."""
    global _api_client, _created_at

    with _lock:
        expired = time.monotonic() - _created_at > _TOKEN_MAX_AGE_SECONDS
        if _api_client is None or force_refresh or expired:
            _api_client = _build_client(load_settings())
            _created_at = time.monotonic()
        return _api_client


def reset_client() -> None:
    """Drop the cached client so the next call re-authenticates."""
    global _api_client
    with _lock:
        _api_client = None


def call_sailpoint(operation: Callable[[ApiClient], T]) -> T:
    """Run `operation` against the shared client, retrying once on a 401.

    Example::

        from sailpoint import SearchApi

        results = call_sailpoint(
            lambda client: SearchApi(client).search_post_v1(search=search, limit=10)
        )
    """
    try:
        return operation(get_api_client())
    except UnauthorizedException:
        log.info("SailPoint token rejected; re-authenticating and retrying once")
        return operation(get_api_client(force_refresh=True))


def describe_api_error(exc: Exception) -> str:
    """Turn an SDK exception into something an LLM (and a human) can act on."""
    if isinstance(exc, (SailPointAuthError, ConfigError)):
        return str(exc)
    if isinstance(exc, ApiException):
        detail = (exc.body or exc.reason or "").strip()
        if len(detail) > 800:
            detail = detail[:800] + "..."
        hints = {
            401: "The Personal Access Token is invalid or expired.",
            403: "The token's identity lacks the required scope or user level for this API.",
            404: "The requested object does not exist in this tenant.",
            429: "SailPoint rate-limited the request. Wait a moment and retry.",
        }
        hint = hints.get(exc.status, "")
        return " ".join(part for part in (f"SailPoint API error {exc.status}.", hint, detail) if part)
    return f"{type(exc).__name__}: {exc}"
