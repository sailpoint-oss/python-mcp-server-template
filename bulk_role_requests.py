"""Preflight and submit human role access requests, never direct assignments.

https://developer.sailpoint.com/docs/api/create-access-request-v-1
https://developer.sailpoint.com/docs/api/list-requestable-objects-v-1
https://developer.sailpoint.com/docs/api/search-post-v-1
https://documentation.sailpoint.com/saas/help/search/building-query.html#using-the-exact-keyword
https://documentation.sailpoint.com/saas/help/search/searchable-fields.html
"""

from __future__ import annotations

import re
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from sailpoint import AccessRequestsApi, RequestableObjectsApi, RolesApi, SearchApi
from sailpoint.access_requests.models.access_request import AccessRequest
from sailpoint.access_requests.models.access_request_item import AccessRequestItem
from sailpoint.access_requests.models.access_request_response import AccessRequestResponse
from sailpoint.exceptions import ApiException
from sailpoint.requestable_objects.models.requestable_object import RequestableObject
from sailpoint.rest import RESTResponse
from sailpoint.roles.models.role import Role
from sailpoint.search.models.filter import Filter
from sailpoint.search.models.query import Query
from sailpoint.search.models.query_result_filter import QueryResultFilter
from sailpoint.search.models.search import Search

from ..client import call_sailpoint, get_api_client

# A local safety cap, NOT the API's role-only maximum (which is unlimited).
# The documented 10-recipient / 25-item limits apply to entitlement requests.
BATCH_SIZE = 10
_ID = re.compile(r"[A-Za-z0-9_-]+")
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_EMAIL = re.compile(
    r"[A-Za-z0-9_%+'-]+(?:\.[A-Za-z0-9_%+'-]+)*@"
    + _DOMAIN_LABEL + r"(?:\." + _DOMAIN_LABEL + r")*"
)
WARNING = (
    "Submitted means HTTP 202 queued, NOT approved or granted. Approval and "
    "provisioning are asynchronous and may fail. Search is eventually consistent; "
    "preflight is not a reservation or guarantee of authorization. Request-on-behalf "
    "configuration and the token user's access-request segments still apply. "
    "Do not blindly retry submitted or unknown recipients: reconcile access request "
    "status in SailPoint first. No automatic approval or assignment is performed."
)


class _PreflightError(ValueError):
    """A safe, locally generated explanation for refusing submission."""


def _resolve_email(value: str) -> dict[str, Any]:
    # Use the bare quoted phrase that works in search_identities for candidates.
    # Only exact case-insensitive top-level email equality below selects a match.
    # Mailbox validation excludes other query syntax; escape + and - as well as
    # quote/backslash delimiters so permitted punctuation stays literal.
    literal = re.sub(r'([+\-\\"])', r'\\\1', value)
    search = Search(
        indices=["identities"],
        query=Query(query=f'"{literal}"'),
        query_result_filter=QueryResultFilter(includes=["id", "email"]),
        include_nested=False,
        sort=["id"],
    )
    match = None
    previous_id = None
    page_size = 250
    while True:
        results = call_sailpoint(
            lambda client: SearchApi(client).search_post_v1(
                search=search, limit=page_size, offset=0,
            )
        )
        if not isinstance(results, list):
            raise _PreflightError("Missing or malformed Search response; cannot resolve safely.")
        for document in results:
            if not isinstance(document, dict):
                raise _PreflightError("Search returned malformed identity data.")
            identity_id = document.get("id")
            if not isinstance(identity_id, str) or not _ID.fullmatch(identity_id):
                raise _PreflightError("Search returned an invalid identity/role ID.")
            if previous_id is not None and identity_id <= previous_id:
                raise _PreflightError("Search pagination did not advance in ID order; cannot resolve safely.")
            previous_id = identity_id
            actual = document.get("email")
            if actual is None:
                continue  # A username or account address is not the identity email.
            if not isinstance(actual, str):
                raise _PreflightError("Search returned malformed identity email data.")
            if actual.casefold() == value:
                if match is not None:
                    raise _PreflightError("Ambiguous identities lookup; multiple exact email matches, no selection made.")
                match = document
        if len(results) < page_size:
            break
        # SearchAfter avoids the 10K offset window; check every candidate, not
        # just the first two hits or the first page, before accepting uniqueness.
        search = search.model_copy(update={"search_after": [previous_id]})
    if match is None:
        raise _PreflightError(
            "No exact identities match on the email attribute (not username); "
            "it may be missing, absent, invisible, or not indexed yet."
        )
    return match


def _resolve(index: str, field: str, value: str) -> dict[str, Any]:
    if index == "identities" and field == "email":
        return _resolve_email(value)
    # name.exact is the case-sensitive keyword field.
    # User input is a typed term, never interpolated into executable query syntax.
    search_field = "name.exact" if index == "roles" else field
    search = Search(
        indices=[index],
        query=Query(query="*"),
        filters={search_field: Filter(type="TERMS", terms=[value])},
        query_result_filter=QueryResultFilter(includes=["id", field]),
        include_nested=False,
        sort=["id"],
    )
    results = call_sailpoint(
        lambda client: SearchApi(client).search_post_v1(search=search, limit=2, offset=0)
    )
    if not isinstance(results, list):
        raise _PreflightError("Missing or malformed Search response; cannot resolve safely.")
    if not results:
        raise _PreflightError(f"No exact {index} match; it may be absent, invisible, or not indexed yet.")
    # Two hits suffice to refuse ambiguity; never truncate and choose the first.
    if len(results) != 1:
        raise _PreflightError(f"Ambiguous {index} lookup; multiple matches, no selection made.")
    document = results[0]
    if not isinstance(document, dict) or not isinstance(document.get(field), str):
        raise _PreflightError("Search returned malformed identity/role data.")
    actual = document[field].casefold() if field == "email" else document[field]
    if actual != value:
        raise _PreflightError(f"Search did not return an exact {field} match.")
    if not isinstance(document.get("id"), str) or not _ID.fullmatch(document["id"]):
        raise _PreflightError("Search returned an invalid identity/role ID.")
    return document


def _error(exc: Exception) -> str:
    # Do not expose arbitrary API response bodies, URLs, tokens, or auth errors.
    if isinstance(exc, ApiException):
        return f"SailPoint API error {exc.status}; inspect authorization and request status in SailPoint."
    return f"Operation failed ({type(exc).__name__}); no automatic retry."


def _preflight_response(client: Any, response: Any) -> Any:
    # Avoid SDK 2.1.56's cross-partition Role lookup and broken requestStatus
    # from_dict. Keep SDK HTTP error handling and validate explicit models below.
    response_data = RESTResponse(response)
    response_data.read()
    return client.response_deserialize(
        response_data=response_data, response_types_map={"200": "object"},
    ).data


def _role_preflight(role_id: str, role_name: str) -> None:
    data = call_sailpoint(lambda client: _preflight_response(
        client, RolesApi(client).get_role_v1_without_preload_content(id=role_id),
    ))
    role = Role.model_validate(data)
    if role.id != role_id or role.name != role_name:
        raise _PreflightError("Role changed since Search indexing; resolve again before requesting.")
    if role.enabled is not True or role.requestable is not True:
        raise _PreflightError("Role is not enabled and requestable.")
    config = role.access_request_config
    if role.dimensional or (config and (
        config.comments_required or config.require_end_date or config.form_definition_id
        or config.dimension_schema
    )):
        raise _PreflightError(
            "Role requires comments, an end date, a form, or dimension selection. "
            "This simple bulk tool cannot supply them; use the SailPoint request workflow."
        )


def _recipient_preflight(identity_id: str, role_id: str, role_name: str) -> None:
    items = call_sailpoint(
        lambda client: _preflight_response(
            client, RequestableObjectsApi(client).list_requestable_objects_v1_without_preload_content(
                identity_id=identity_id, types="ROLE", filters=f'id eq "{role_id}"',
                limit=2, offset=0,
            ),
        )
    )
    if not isinstance(items, list) or len(items) != 1:
        raise _PreflightError("Role is unavailable or requestability is ambiguous for this identity.")
    item = RequestableObject.model_validate(items[0])
    if item.id != role_id or item.name != role_name or item.type != "ROLE":
        raise _PreflightError("Requestable role does not exactly match the resolved role.")
    if item.request_status != "AVAILABLE":
        raise _PreflightError("Role is already ASSIGNED, PENDING, or not confirmed AVAILABLE; nothing submitted.")
    if item.request_comments_required:
        raise _PreflightError("Role requires a request comment; use the SailPoint request workflow.")


def register(mcp: MCPServer) -> None:
    @mcp.tool(
        name="bulk_role_requests",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True,
            idempotent_hint=False, open_world_hint=True,
        ),
    )
    def bulk_role_requests(
        role_name: str, emails: list[str], dry_run: bool = True,
    ) -> dict[str, Any]:
        """Preview or submit GRANT_ACCESS requests for one role and human users.

        The two-argument call is a read-only preview. Set dry_run=False only when
        the user authorizes submission after reviewing the resolved role/users.
        Matches the trimmed role name exactly (case-sensitive), not ownership.
        Emails are trimmed, case-insensitive, unquoted ASCII mailboxes; duplicate
        emails and identity IDs are submitted only once. Never guesses between
        matches. All lookups and AVAILABLE checks must pass before any writes.

        Requires permission to read role details and requestable objects for each
        recipient (other identities require admin access), plus authorization to
        request on their behalf. Required comments/forms/end dates/dimensions are
        not supported and fail preflight. Does not support machine identities.

        Requests use local batches of 10, with no truncation. Role-only requests
        have no documented API recipient maximum; entitlement limits do not apply.
        Returns per-user and per-batch preview/submitted/failed/unknown/not_submitted
        status. submitted means queued, not granted. Stops on the first failure or
        uncertain result; never retries submissions, even on 401. Earlier batches
        cannot be rolled back. Reconcile submitted/unknown users in SailPoint before
        retrying; repeated invocations are NOT idempotent. Search can be stale and
        eligibility can change between preflight and asynchronous processing.
        """
        role_name = role_name.strip()
        response: dict[str, Any] = {
            "role_name": role_name, "dry_run": dry_run, "status": "preflight_failed",
            "preflight_complete": False, "users": [], "batches": [],
            "batch_size": BATCH_SIZE, "warning": WARNING,
        }
        errors: list[dict[str, str]] = []
        if not role_name or len(role_name) > 128 or any(ord(c) < 32 for c in role_name):
            errors.append({"error": "role_name must be a nonempty name of at most 128 characters without controls."})
        if not emails:
            errors.append({"error": "emails must contain at least one address."})
        users: dict[str, dict[str, Any]] = {}
        for raw in emails:
            email = raw.strip().casefold()
            if (not raw.strip().isascii() or len(email) > 254
                    or len(email.split("@", 1)[0]) > 64 or not _EMAIL.fullmatch(email)):
                errors.append({"email": email, "error": "Invalid email; use a simple unquoted ASCII mailbox, not search syntax."})
            if email in users:
                continue
            users[email] = {"email": email, "status": "not_submitted"}
        response["users"] = list(users.values())
        response["duplicate_emails_removed"] = len(emails) - len(users)
        if errors:
            response["errors"] = errors
            return response

        try:
            role_id = _resolve("roles", "name", role_name)["id"]
            response["role_id"] = role_id
            _role_preflight(role_id, role_name)
        except Exception as exc:
            response["errors"] = [{"role_name": role_name, "error": str(exc) if isinstance(exc, _PreflightError) else _error(exc)}]
            return response

        recipients: dict[str, list[dict[str, Any]]] = {}
        for email, user in users.items():
            try:
                identity_id = _resolve("identities", "email", email)["id"]
                user["identity_id"] = identity_id
                recipients.setdefault(identity_id, []).append(user)
            except Exception as exc:
                errors.append({"email": email, "error": str(exc) if isinstance(exc, _PreflightError) else _error(exc)})
        for identity_id in recipients:
            try:
                _recipient_preflight(identity_id, role_id, role_name)
            except Exception as exc:
                errors.append({"identity_id": identity_id, "error": str(exc) if isinstance(exc, _PreflightError) else _error(exc)})
        response["unique_identity_count"] = len(recipients)
        response["duplicate_identities_removed"] = sum(len(group) - 1 for group in recipients.values())
        if errors:
            response["errors"] = errors
            return response

        identity_ids = list(recipients)
        # Build every payload before the first non-idempotent operation.
        payloads = [
            AccessRequest(
                requested_for=identity_ids[start:start + BATCH_SIZE],
                request_type="GRANT_ACCESS",
                requested_items=[AccessRequestItem(type="ROLE", id=role_id)],
            )
            for start in range(0, len(identity_ids), BATCH_SIZE)
        ]
        response["preflight_complete"] = True
        response["batches"] = [
            {"batch": number, "identity_ids": payload.requested_for, "status": "not_submitted"}
            for number, payload in enumerate(payloads, 1)
        ]
        response["status"] = "preview" if dry_run else "submitted"
        for batch, payload in zip(response["batches"], payloads):
            if dry_run:
                batch["status"] = "preview"
            else:
                try:
                    # Reuse shared auth, but NOT call_sailpoint's automatic 401 retry.
                    api = AccessRequestsApi(get_api_client())
                except Exception as exc:
                    batch["error"] = _error(exc)
                    response["status"] = "stopped"
                    break
                try:
                    result = api.create_access_request_v1_with_http_info(
                        access_request=payload, _request_timeout=(10.0, 60.0),
                    )
                    if result.status_code != 202:
                        batch["status"] = "unknown"
                        batch["error"] = "Unexpected response; submission outcome is uncertain."
                    else:
                        batch["status"] = "submitted"
                        if isinstance(result.data, AccessRequestResponse):
                            batch["tracking"] = result.data.to_dict()
                except Exception as exc:
                    # Only documented rejection responses establish failure. A 5xx,
                    # timeout, deserialization error, etc. may follow a queued write.
                    batch["status"] = "failed" if isinstance(exc, ApiException) and exc.status in (400, 401, 403, 429) else "unknown"
                    batch["error"] = _error(exc)
            for identity_id in batch["identity_ids"]:
                for user in recipients[identity_id]:
                    user["status"] = batch["status"]
                    user["batch"] = batch["batch"]
            if batch["status"] in ("failed", "unknown"):
                response["status"] = "stopped"
                break
        response["user_counts"] = {
            status: sum(user["status"] == status for user in users.values())
            for status in ("preview", "submitted", "failed", "unknown", "not_submitted")
        }
        return response
