"""`list_access_requests_by_state` -- tenant-wide access request triage.

Wraps `GET /access-request-administration`
(`AccessRequestsApi.list_administrators_access_request_status_v1`). That is the
admin view: every access request in the tenant, not just the caller's own.

Two things about this endpoint, both discovered by calling it:

  * It is flagged *experimental*. The SDK sends `X-SailPoint-Experimental: true`
    on its own (the method's `x_sail_point_experimental` parameter defaults to
    `"true"`), but it also refuses to call experimental endpoints at all unless
    `configuration.experimental` is set -- which `client._build_client()` now
    does for the whole server.
  * Its `filters` parameter supports `accountActivityItemId`, `accessRequestId`,
    `status` and `created`, but *not* `state`, and its `request_state` parameter
    accepts only `EXECUTING`. So `states` is applied after fetching, as the
    spec's fallback describes. `created_after` and `requested_for` do go to the
    API, since those it supports natively.

Same four steps as `search_identities`: build the request, run it through
`call_sailpoint()`, project the fields worth reading, return errors as data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from mcp.server import MCPServer
from sailpoint import AccessRequestsApi
from sailpoint.exceptions import ApiException

from ..client import call_sailpoint, describe_api_error
from .list_source_entitlements import _iso  # same datetime -> ISO rule

log = logging.getLogger(__name__)

# The API's own ceiling per page.
PAGE_SIZE = 250

DEFAULT_MAX_RESULTS = 1000
HARD_MAX_RESULTS = 5000

# How many requests per state to show when the caller only wants counts.
SAMPLE_PER_STATE = 5

DEFAULT_SORTERS = "-created"

# What the SDK's enum knows about. Not enforced -- a tenant may return a state
# this SDK has never heard of, and dropping it would be worse than passing it
# through -- but it is what the docstring advertises. Note there is no PENDING:
# a request awaiting approval sits in EXECUTING.
KNOWN_STATES = (
    "EXECUTING",
    "REQUEST_COMPLETED",
    "CANCELLED",
    "TERMINATED",
    "PROVISIONING_VERIFICATION_PENDING",
    "REJECTED",
    "PROVISIONING_FAILED",
    "NOT_ALL_ITEMS_PROVISIONED",
    "ERROR",
)


def build_filters(created_after: str | None) -> str | None:
    """Turn `created_after` into the API's filter expression."""
    created_after = (created_after or "").strip()
    if not created_after:
        return None
    return f'created gt "{created_after}"'


def _clean_states(states: Sequence[str] | None) -> set[str]:
    """Normalise the requested states to a comparable set. Empty means all."""
    if not states:
        return set()
    if isinstance(states, str):  # a single state passed unwrapped
        states = [states]
    return {state.strip().upper() for state in states if (state or "").strip()}


def _named(ref: Any) -> str | None:
    """Render an identity reference as `Name (id)`, with whatever is present."""
    if not isinstance(ref, dict):
        return None
    name, identifier = ref.get("name"), ref.get("id")
    if name and identifier:
        return f"{name} ({identifier})"
    return name or identifier or None


def _current_approvers(approval_details: Any) -> list[str]:
    """Names of the approvers a request is actually waiting on."""
    approvers: list[str] = []
    for approval in approval_details or []:
        if not isinstance(approval, dict):
            continue
        if str(approval.get("status") or "").upper() != "PENDING":
            continue
        owner = _named(approval.get("currentOwner") or approval.get("originalOwner"))
        if owner and owner not in approvers:
            approvers.append(owner)
    return approvers


def summarize_request(request: Any) -> dict[str, Any]:
    """Flatten one admin status record (SDK model or dict) into a compact one."""
    document = request.to_dict() if hasattr(request, "to_dict") else dict(request)

    summary = {
        "access_request_id": document.get("accessRequestId") or document.get("id"),
        "name": document.get("name"),
        "type": document.get("type"),
        "requested_for": _named(document.get("requestedFor")),
        "requester": _named(document.get("requester")),
        "created": _iso(document.get("created")),
        "modified": _iso(document.get("modified")),
        "current_approvers": _current_approvers(document.get("approvalDetails")),
        "removal_date": _iso(document.get("removeDate")),
    }
    # Omit nulls and empties -- this output is read by a model, in bulk.
    return {
        key: value
        for key, value in summary.items()
        if value is not None and value not in ([], {}, "")
    }


def state_of(request: Any) -> str:
    """Read a record's state, whatever shape it arrives in."""
    document = request.to_dict() if hasattr(request, "to_dict") else dict(request)
    state = document.get("state")
    # The SDK models `state` as an enum; `to_dict()` may leave it as one.
    return str(getattr(state, "value", state) or "UNKNOWN").upper()


def group_by_state(
    requests: Iterable[Any],
    *,
    include_items: bool = False,
) -> dict[str, dict[str, Any]]:
    """Group summaries by state. Counts are always complete; items may be a sample."""
    grouped: dict[str, dict[str, Any]] = {}
    for request in requests:
        bucket = grouped.setdefault(state_of(request), {"count": 0, "requests": []})
        bucket["count"] += 1
        if include_items or len(bucket["requests"]) < SAMPLE_PER_STATE:
            bucket["requests"].append(summarize_request(request))
    return grouped


def _fetch_page(
    offset: int,
    limit: int,
    *,
    filters: str | None,
    requested_for: str | None,
    experimental: bool = True,
) -> list[Any]:
    """One page from the admin endpoint."""
    return call_sailpoint(
        lambda client: AccessRequestsApi(
            client
        ).list_administrators_access_request_status_v1(
            limit=limit,
            offset=offset,
            sorters=DEFAULT_SORTERS,
            filters=filters,
            requested_for=requested_for,
            # Sent by default, but pinned here so a retry can be explicit.
            x_sail_point_experimental="true" if experimental else None,
        )
        or []
    )


def fetch_all(
    fetch: Callable[[int, int], list[Any]],
    max_results: int,
) -> tuple[list[Any], bool]:
    """Page until a short page arrives or `max_results` is reached.

    Returns the records and whether the result was cut short. `truncated` is
    conservative: a run that stops exactly on `max_results` with a full last
    page reports True even if the tenant had nothing more to give.
    """
    records: list[Any] = []
    truncated = False

    while len(records) < max_results:
        page_size = min(PAGE_SIZE, max_results - len(records))
        page = fetch(len(records), page_size) or []
        records.extend(page)
        if len(page) < page_size:  # short page means the end of the collection
            return records, False
        truncated = True  # a full page landed on the cap; there may be more

    return records, truncated


def _admin_rights_hint(exc: Exception) -> str | None:
    """A 401/403 here almost always means the token is not an admin token."""
    if isinstance(exc, ApiException) and exc.status in (401, 403):
        return (
            "This is a tenant-wide admin view, and the token's identity is not "
            "authorized for it. The Personal Access Token must belong to an "
            "identity with an admin user level (e.g. ORG_ADMIN) and be created "
            "with the scopes that cover access request administration. "
            f"{describe_api_error(exc)}"
        )
    return None


def collect(
    states: Sequence[str] | None = None,
    created_after: str | None = None,
    requested_for: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    include_items: bool = False,
    fetch: Callable[[int, int], list[Any]] | None = None,
) -> dict[str, Any]:
    """Fetch, filter, group. `fetch` is injectable so the paging is testable."""
    max_results = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), HARD_MAX_RESULTS))
    wanted = _clean_states(states)
    filters = build_filters(created_after)
    requested_for = (requested_for or "").strip() or None

    if fetch is None:
        def fetch(offset: int, limit: int) -> list[Any]:
            return _fetch_page(
                offset, limit, filters=filters, requested_for=requested_for
            )

    log.info(
        "list_access_requests_by_state: states=%s filters=%s requested_for=%s max=%d",
        sorted(wanted) or "ALL",
        filters,
        requested_for,
        max_results,
    )

    try:
        records, truncated = fetch_all(fetch, max_results)
    except Exception as exc:
        hint = _admin_rights_hint(exc)
        if hint is None and isinstance(exc, ApiException) and exc.status in (400, 404):
            # The endpoint is experimental; the SDK sends the header by default,
            # so this retry is belt-and-braces -- but if it fails twice, the
            # caller deserves to know the header was not the problem.
            log.info("retrying the admin endpoint with the experimental header")
            try:
                records, truncated = fetch_all(fetch, max_results)
            except Exception as retry_exc:
                return {
                    "error": (
                        "The access request administration endpoint rejected the "
                        "request even with X-SailPoint-Experimental: true. It may "
                        "not be enabled for this tenant. "
                        f"{describe_api_error(retry_exc)}"
                    ),
                    "by_state": {},
                }
        else:
            log.exception("list_access_requests_by_state failed")
            return {"error": hint or describe_api_error(exc), "by_state": {}}

    if wanted:
        records = [record for record in records if state_of(record) in wanted]

    response: dict[str, Any] = {
        "total": len(records),
        "truncated": truncated,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "by_state": group_by_state(records, include_items=include_items),
    }
    if truncated:
        response["note"] = (
            f"Stopped at max_results={max_results}; counts cover only what was "
            "fetched. Raise `max_results` or narrow with `created_after`."
        )
    elif not records:
        response["note"] = (
            "No access requests matched. Note there is no PENDING state -- a "
            "request awaiting approval is EXECUTING."
        )
    elif not include_items:
        response["note"] = (
            f"Counts are complete; at most {SAMPLE_PER_STATE} sample requests "
            "per state are shown. Pass include_items=true for all of them."
        )
    return response


def register(mcp: MCPServer) -> None:
    @mcp.tool(name="list_access_requests_by_state")
    def list_access_requests_by_state(
        states: list[str] | None = None,
        created_after: str | None = None,
        requested_for: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        include_items: bool = False,
    ) -> dict[str, Any]:
        """Count and triage access requests across the whole tenant, by state.

        This is the ADMIN view: every access request in the tenant, regardless
        of who raised it, so the Personal Access Token must belong to an admin
        identity (e.g. ORG_ADMIN). Use it for questions about the request queue
        as a whole -- "how many access requests are pending right now", "show me
        requests stuck in error", "what failed to provision this week".

        For one person's requests, pass their identity ID as `requested_for`
        rather than fetching the tenant and filtering by eye.

        States are: EXECUTING, REQUEST_COMPLETED, CANCELLED, TERMINATED,
        PROVISIONING_VERIFICATION_PENDING, REJECTED, PROVISIONING_FAILED,
        NOT_ALL_ITEMS_PROVISIONED, ERROR. There is no PENDING state -- a request
        waiting on an approver is EXECUTING, and its `current_approvers` names
        who it is waiting on.

        Args:
            states: Only include these states, e.g.
                `["EXECUTING", "PROVISIONING_FAILED"]`. Omit for all states.
            created_after: Only requests created after this ISO 8601 datetime,
                e.g. `2026-09-01T00:00:00Z`. Use it to keep a busy tenant's
                result set small.
            requested_for: Identity ID of the person the requests were made
                for. Use this for "what has Alice requested", after resolving
                her ID with `search_identities`.
            max_results: Cap on records fetched, default 1000, maximum 5000.
                Paging stops there and `truncated` comes back true.
            include_items: Return every matching request instead of at most
                5 samples per state. Counts are complete either way, so leave
                this off for "how many" questions -- it can be a lot of output.

        Returns:
            A dict with `total`, `truncated`, `fetched_at`, and `by_state`:
            each state mapping to `count` and `requests`, where a request gives
            `access_request_id`, `name`, `type`, `requested_for`, `requester`,
            `created`, `modified`, `current_approvers` and `removal_date`. Null
            fields are omitted. On failure, `error` explains why and `by_state`
            is empty.
        """
        return collect(
            states=states,
            created_after=created_after,
            requested_for=requested_for,
            max_results=max_results,
            include_items=include_items,
        )
