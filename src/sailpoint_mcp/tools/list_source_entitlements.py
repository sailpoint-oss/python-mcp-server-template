"""`list_source_entitlements` -- the entitlements defined on one source.

Wraps the same endpoint as::

    GET /v2025/entitlements?filters=source.id eq "{sourceId}" and requestable eq true&limit=250

`EntitlementsApi.list_entitlements_v1` is that endpoint in the SDK (the SDK's
`_v1` suffix is its method naming, not an older API version), so the filter and
sorter syntax in the API docs applies verbatim.

Exposed as an MCP *resource* (`sailpoint://sources/{source_id}/entitlements`),
not a tool. Same four steps as `search_identities` otherwise: build the
request, run it through `call_sailpoint()`, project the fields worth reading,
return errors as data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from mcp.server import MCPServer
from sailpoint import EntitlementsApi

from ..client import call_sailpoint, describe_api_error
from .search_identities import total_count

log = logging.getLogger(__name__)

# The API's own ceiling for this collection.
MAX_LIMIT = 250

# Entitlement sets run to thousands on a directory source, and each summary
# costs context, so the default is small. Raise it when the user wants the list.
DEFAULT_LIMIT = 50

# `offset` paging is only stable against a deterministic order, and the API does
# not promise one. Name first reads well; id breaks ties.
DEFAULT_SORTERS = "name,id"

# Descriptions are free text and occasionally hold a paragraph of policy prose.
MAX_DESCRIPTION = 300


def _iso(value: Any) -> Any:
    """SDK models hand back `datetime` for timestamps; a tool result has to
    survive JSON serialization, so render those as ISO strings."""
    return value.isoformat() if isinstance(value, datetime) else value


def _quote(value: str) -> str:
    """Quote a value for an ISC filter expression."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_filters(
    source_id: str,
    *,
    requestable: bool | None = None,
    name: str | None = None,
    extra: str | None = None,
) -> str:
    """Build the `filters` expression, always scoped to one source.

    `source.id eq "..."` is non-negotiable -- without it this endpoint returns
    every entitlement in the tenant, which is never what the caller meant.
    `extra` is parenthesised so an `or` inside it cannot widen that scope.
    """
    source_id = (source_id or "").strip()
    if not source_id:
        raise ValueError(
            "source_id is required. Pass the source's ID (a 32-character hex "
            "string), which `search_identities` reports as an account source."
        )

    clauses = [f"source.id eq {_quote(source_id)}"]
    if requestable is not None:
        clauses.append(f"requestable eq {str(requestable).lower()}")
    if name and name.strip():
        clauses.append(f"name sw {_quote(name.strip())}")
    if extra and extra.strip():
        clauses.append(f"({extra.strip()})")
    return " and ".join(clauses)


def summarize_entitlement(entitlement: Any) -> dict[str, Any]:
    """Flatten one entitlement (SDK model or dict) into a compact record."""
    document = (
        entitlement.to_dict() if hasattr(entitlement, "to_dict") else dict(entitlement)
    )
    source = document.get("source") or {}
    owner = document.get("owner") or {}
    privilege = document.get("privilegeLevel") or {}

    description = document.get("description")
    if isinstance(description, str) and len(description) > MAX_DESCRIPTION:
        description = description[:MAX_DESCRIPTION] + "..."

    summary = {
        "id": document.get("id"),
        "name": document.get("name"),
        "type": document.get("sourceSchemaObjectType"),
        "attribute": document.get("attribute"),
        "value": document.get("value"),
        "description": description,
        "requestable": document.get("requestable"),
        "cloud_governed": document.get("cloudGoverned"),
        "privilege_level": privilege.get("direct") if isinstance(privilege, dict) else None,
        "tags": document.get("tags"),
        "owner": owner.get("name") if isinstance(owner, dict) else None,
        "source": source.get("name") if isinstance(source, dict) else None,
        "source_id": source.get("id") if isinstance(source, dict) else None,
        "created": _iso(document.get("created")),
        "modified": _iso(document.get("modified")),
    }
    # Drop empties so the model isn't reading a wall of nulls. `False` is a real
    # answer for `requestable`/`cloud_governed`, so it has to survive this.
    return {
        key: value
        for key, value in summary.items()
        if value is not None and value not in ([], {}, "")
    }


def fetch_entitlements(
    source_id: str,
    *,
    requestable: bool | None = None,
    name: str | None = None,
    filters: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    sorters: str | None = None,
    count: bool = False,
) -> dict[str, Any]:
    """Do the work. Shared by the tool and the resource template below."""
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    sorters = (sorters or "").strip() or DEFAULT_SORTERS

    try:
        filter_expression = build_filters(
            source_id, requestable=requestable, name=name, extra=filters
        )
    except ValueError as exc:
        return {"error": str(exc), "entitlements": []}

    log.info(
        "list_source_entitlements: %s (limit=%d offset=%d count=%s)",
        filter_expression,
        limit,
        offset,
        count,
    )

    try:
        if count:
            # Only the *_with_http_info variant exposes X-Total-Count.
            api_response = call_sailpoint(
                lambda client: EntitlementsApi(client).list_entitlements_v1_with_http_info(
                    filters=filter_expression,
                    sorters=sorters,
                    limit=limit,
                    offset=offset,
                    count=True,
                )
            )
            results = api_response.data
            matched = total_count(api_response.headers)
        else:
            results = call_sailpoint(
                lambda client: EntitlementsApi(client).list_entitlements_v1(
                    filters=filter_expression,
                    sorters=sorters,
                    limit=limit,
                    offset=offset,
                )
            )
            matched = None
    except Exception as exc:  # surfaced to the model as tool output, not a crash
        log.exception("list_source_entitlements failed")
        return {
            "source_id": source_id,
            "filters": filter_expression,
            "error": describe_api_error(exc),
            "entitlements": [],
        }

    entitlements = [summarize_entitlement(item) for item in results or []]

    response: dict[str, Any] = {
        "source_id": source_id,
        "filters": filter_expression,
        "returned": len(entitlements),
        "offset": offset,
        "limit": limit,
        "entitlements": entitlements,
    }
    if matched is not None:
        response["total_count"] = matched
    if not entitlements:
        response["note"] = (
            "No entitlements matched. Check the source ID is right, and note "
            "that `requestable=True` hides entitlements not open to requests."
        )
    elif len(entitlements) == limit:
        response["note"] = (
            f"Returned the first {limit}; there may be more. Page with "
            "`offset`, or pass `count=True` to see the total."
        )
    return response


def register(mcp: MCPServer) -> None:
    """Registered as a resource, not a tool: the caller reads a source's
    entitlements by URI rather than calling a function. That means no filter
    arguments -- a resource is addressed only by what is in its URI -- so the
    read returns the source's requestable entitlements, up to the API's 250.
    `fetch_entitlements()` still takes the full filter set for any caller that
    needs it.
    """

    @mcp.resource(
        "sailpoint://sources/{source_id}/entitlements",
        name="list_source_entitlements",
        description=(
            "Requestable entitlements on one SailPoint source. Read "
            "sailpoint://sources/<sourceId>/entitlements, where <sourceId> is "
            "the source's 32-character hex ID. Returns each entitlement's "
            "name, type, attribute/value, description, owner and source."
        ),
        mime_type="application/json",
    )
    def list_source_entitlements(source_id: str) -> dict[str, Any]:
        return fetch_entitlements(source_id, requestable=True, limit=MAX_LIMIT)
