"""`request_access_profile` -- submit an access request for one access profile.

This is a *write* tool: it files a real access request in the tenant, which
starts the approval workflow and notifies approvers. The read tools in this
package are safe to retry; this one is not.

Wraps `POST /access-requests` (`AccessRequestsApi.create_access_request_v1`),
which takes::

    {"requestedFor": ["2c91808568c529c60168cca6f90c1313"],
     "requestType": "GRANT_ACCESS",
     "requestedItems": [{"type": "ACCESS_PROFILE",
                         "id": "2c9180835d2e5168015d32f890ca1581",
                         "comment": "Joining the finance team"}]}

The tool asks for the two IDs and builds that body. Two details worth knowing,
both pinned by tests: `requestedFor` is a list of bare identity ID strings, and
`comment` sits on the requested item rather than on the envelope.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server import MCPServer
from sailpoint import AccessRequestsApi
from sailpoint.access_requests.models.access_request import AccessRequest
from sailpoint.access_requests.models.access_request_item import AccessRequestItem

from ..client import call_sailpoint, describe_api_error

log = logging.getLogger(__name__)

ITEM_TYPE = "ACCESS_PROFILE"
REQUEST_TYPE = "GRANT_ACCESS"


def _model(cls: Any, **kwargs: Any) -> Any:
    """Build an SDK model without explicit `None`s.

    The generated `to_dict` serializes a nullable field that was *set* to None
    as a literal `null`, so passing `comment=None` would put `"comment": null`
    on the wire. Omitting the keyword leaves it out instead.
    """
    return cls(**{key: value for key, value in kwargs.items() if value is not None})


def _require_id(value: str | None, label: str) -> str:
    """Validate one caller-supplied ID. Nothing reaches the API unchecked."""
    value = value.strip() if isinstance(value, str) else ""
    if not value:
        raise ValueError(
            f"{label} is required and must be an ID, not a name -- a "
            "32-character hex string such as "
            "`2c9180835d2e5168015d32f890ca1581`."
        )
    return value


def build_access_request(
    access_profile_id: str,
    identity_id: str,
    comment: str | None = None,
) -> AccessRequest:
    """Assemble the request body. Pure, so the mapping is unit-testable."""
    access_profile_id = _require_id(access_profile_id, "access_profile_id")
    identity_id = _require_id(identity_id, "identity_id")
    comment = comment.strip() if isinstance(comment, str) else ""

    return AccessRequest(
        requested_for=[identity_id],
        request_type=REQUEST_TYPE,
        requested_items=[
            _model(
                AccessRequestItem,
                type=ITEM_TYPE,
                id=access_profile_id,
                comment=comment or None,
            )
        ],
    )


def _tracking_ids(trackings: Any) -> list[str]:
    """Pull the access request IDs out of a `newRequests`/`existingRequests` list."""
    ids: list[str] = []
    for tracking in trackings or []:
        document = (
            tracking.to_dict() if hasattr(tracking, "to_dict") else dict(tracking)
        )
        ids.extend(document.get("accessRequestIds") or [])
    return ids


def summarize_response(response: Any) -> dict[str, Any]:
    """Reduce an `AccessRequestResponse` to what the caller needs to report back.

    `existingRequests` is the interesting case: ISC answers 200 and files
    nothing when the same access is already pending for that identity, so a
    naive "submitted!" would be a lie.
    """
    document = (
        response.to_dict() if hasattr(response, "to_dict") else dict(response or {})
    )
    new_ids = _tracking_ids(document.get("newRequests"))
    existing_ids = _tracking_ids(document.get("existingRequests"))

    summary: dict[str, Any] = {
        "submitted": bool(new_ids) or not existing_ids,
        "access_request_ids": new_ids or existing_ids,
    }
    if existing_ids and not new_ids:
        summary["submitted"] = False
        summary["already_requested"] = True
        summary["note"] = (
            "This access is already requested for that identity, so no new "
            "request was created. The IDs returned are the existing request(s)."
        )
    return summary


def submit(
    access_profile_id: str,
    identity_id: str,
    comment: str | None = None,
) -> dict[str, Any]:
    """Validate, submit, summarize. Errors come back as data, never as a crash."""
    try:
        access_request = build_access_request(access_profile_id, identity_id, comment)
    except ValueError as exc:
        return {"submitted": False, "error": str(exc)}

    body = access_request.to_dict()
    log.info("request_access_profile: submitting %s", body)

    try:
        response = call_sailpoint(
            lambda client: AccessRequestsApi(client).create_access_request_v1(
                access_request=access_request
            )
        )
    except Exception as exc:
        log.exception("request_access_profile failed")
        return {
            "submitted": False,
            "identity_id": identity_id,
            "access_profile_id": access_profile_id,
            "request_body": body,
            "error": describe_api_error(exc),
        }

    result: dict[str, Any] = {
        "identity_id": identity_id,
        "access_profile_id": access_profile_id,
        "request_type": REQUEST_TYPE,
        **summarize_response(response),
    }
    if result.get("submitted") and not result.get("access_request_ids"):
        # ISC accepted it but returned no tracking IDs -- say so rather than
        # inventing an ID the caller could go looking for.
        result["note"] = (
            "Accepted by SailPoint, but no access request ID was returned. "
            "Check the identity's request history to confirm."
        )
    return result


def register(mcp: MCPServer) -> None:
    @mcp.tool(name="request_access_profile")
    def request_access_profile(
        access_profile_id: str,
        identity_id: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Submit an access request granting one access profile to one identity.

        This WRITES to SailPoint: it files a real access request and starts the
        approval workflow, so only call it when the user has asked for that
        access to be requested for a specific person. Do not call it to explore
        or to check what access exists -- use `search_identities` for a person's
        access and `list_source_entitlements` for what a source offers.

        Both IDs are required, and both must be IDs, not names. If the user
        gives you a person's name, resolve it with `search_identities` first and
        confirm you have the right person before submitting.

        Args:
            access_profile_id: The access profile to request, a 32-character hex
                ID, e.g. `2c9180835d2e5168015d32f890ca1581`.
            identity_id: The identity the access is for -- the person who will
                receive it, not the requester. A 32-character hex ID, e.g.
                `2c91808568c529c60168cca6f90c1313`.
            comment: Business justification recorded on the request and shown to
                approvers, e.g. "Joining the finance team". Include the user's
                stated reason when they gave one.

        Returns:
            A dict with `submitted`, `identity_id`, `access_profile_id`,
            `request_type`, and `access_request_ids` for tracking. When the same
            access is already pending, `submitted` is False and
            `already_requested` is True -- report that rather than claiming a
            new request was filed. On failure, `submitted` is False and `error`
            explains why.
        """
        return submit(access_profile_id, identity_id, comment)
