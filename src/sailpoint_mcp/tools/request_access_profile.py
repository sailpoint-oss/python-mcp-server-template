"""`request_access_profile` -- submit an access request, by name.

This is a *write* tool: it files a real access request in the tenant, which
starts the approval workflow and notifies approvers. The read tools in this
package are safe to retry; this one is not.

Callers give display names -- "Accounts Payable - AD" for "Aaron.Nichols" --
and this module resolves each to its ID before submitting. Resolution is
deliberately strict: a name that matches nothing, or matches more than one
object, stops the request and reports the candidates. Guessing which Aaron to
grant access to is exactly the kind of help nobody wants from a write tool.

Wraps `POST /access-requests` (`AccessRequestsApi.create_access_request_v1`),
which takes::

    {"requestedFor": ["2c91808568c529c60168cca6f90c1313"],
     "requestType": "GRANT_ACCESS",
     "requestedItems": [{"type": "ACCESS_PROFILE",
                         "id": "2c9180835d2e5168015d32f890ca1581",
                         "comment": "Joining the finance team"}]}

Note `requestedFor` is a list of bare identity ID strings, and `comment` sits
on the requested item rather than on the envelope.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Sequence

from mcp.server import MCPServer
from sailpoint import AccessProfilesApi, AccessRequestsApi, SearchApi
from sailpoint.access_requests.models.access_request import AccessRequest
from sailpoint.access_requests.models.access_request_item import AccessRequestItem

from ..client import call_sailpoint, describe_api_error
from .list_source_entitlements import _quote
from .search_identities import build_query_string, build_search

log = logging.getLogger(__name__)

ITEM_TYPE = "ACCESS_PROFILE"
REQUEST_TYPE = "GRANT_ACCESS"

# ISC object IDs. A caller who already has one should not be forced through a
# name lookup that could fail to match it.
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)

# How many candidates to look up, and how many to name back when ambiguous.
LOOKUP_LIMIT = 25
CANDIDATES_SHOWN = 10


class ResolutionError(ValueError):
    """Raised when a name matches no object, or more than one."""


def _model(cls: Any, **kwargs: Any) -> Any:
    """Build an SDK model without explicit `None`s.

    The generated `to_dict` serializes a nullable field that was *set* to None
    as a literal `null`, so passing `comment=None` would put `"comment": null`
    on the wire. Omitting the keyword leaves it out instead.
    """
    return cls(**{key: value for key, value in kwargs.items() if value is not None})


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def find_access_profiles(name: str) -> list[dict[str, str]]:
    """Access profiles matching `name`: exact first, then starts-with."""
    exact = call_sailpoint(
        lambda client: AccessProfilesApi(client).list_access_profiles_v1(
            filters=f"name eq {_quote(name)}", limit=LOOKUP_LIMIT
        )
    )
    matches = exact or call_sailpoint(
        lambda client: AccessProfilesApi(client).list_access_profiles_v1(
            filters=f"name sw {_quote(name)}", limit=LOOKUP_LIMIT, sorters="name"
        )
    )
    return [
        {
            "id": profile.id,
            "name": profile.name,
            # Carried so a non-requestable profile can be refused before the
            # write, rather than by the API after it.
            "requestable": bool(getattr(profile, "requestable", True)),
        }
        for profile in matches or []
        if getattr(profile, "id", None)
    ]


def find_identities(name: str) -> list[dict[str, str]]:
    """Identities matching `name`, via the same search the read tool uses."""
    search = build_search(
        build_query_string(name),
        attributes=["id", "displayName", "name", "email"],
    )
    hits = call_sailpoint(
        lambda client: SearchApi(client).search_post_v1(search=search, limit=LOOKUP_LIMIT)
    )
    return [
        {
            "id": hit.get("id"),
            "name": hit.get("displayName") or hit.get("name"),
            "email": hit.get("email"),
        }
        for hit in hits or []
        if isinstance(hit, dict) and hit.get("id")
    ]


def _describe(candidate: dict[str, Any]) -> str:
    label = candidate.get("name") or "(unnamed)"
    email = candidate.get("email")
    if email:
        label = f"{label} <{email}>"
    if candidate.get("requestable") is False:
        label = f"{label} (not requestable)"
    return f"{label} [{candidate['id']}]"


def resolve(
    value: str | None,
    label: str,
    finder: Callable[[str], Sequence[dict[str, str]]],
) -> dict[str, str]:
    """Turn one display name into exactly one object, or refuse to continue."""
    value = _text(value)
    if not value:
        raise ResolutionError(
            f"{label} is required -- the display name, e.g. "
            '"Accounts Payable - AD" or "Aaron.Nichols".'
        )

    # Already an ID: take it as given rather than round-tripping the name.
    if _ID_PATTERN.match(value):
        return {"id": value, "name": value, "resolved_from": "id"}

    candidates = list(finder(value) or [])

    if not candidates:
        raise ResolutionError(
            f"No {label} matched {value!r}. Check the spelling, or pass the "
            "32-character ID directly."
        )

    if len(candidates) > 1:
        # An exact, case-insensitive name match settles it; otherwise stop.
        exact = [c for c in candidates if _text(c.get("name")).lower() == value.lower()]
        if len(exact) != 1:
            shown = ", ".join(_describe(c) for c in candidates[:CANDIDATES_SHOWN])
            more = (
                f" (+{len(candidates) - CANDIDATES_SHOWN} more)"
                if len(candidates) > CANDIDATES_SHOWN
                else ""
            )
            raise ResolutionError(
                f"{len(candidates)} matches for {label} {value!r}, so nothing "
                f"was submitted. Pass the exact name or the ID. Candidates: "
                f"{shown}{more}"
            )
        candidates = exact

    chosen = candidates[0]
    # Spread the candidate so fields the caller cares about -- `requestable`,
    # say -- survive resolution instead of being dropped here.
    return {
        **chosen,
        "id": chosen["id"],
        "name": chosen.get("name") or value,
        "resolved_from": "name",
    }


def build_access_request(
    access_profile_id: str,
    identity_id: str,
    comment: str | None = None,
) -> AccessRequest:
    """Assemble the request body. Pure, so the mapping is unit-testable."""
    return AccessRequest(
        requested_for=[identity_id],
        request_type=REQUEST_TYPE,
        requested_items=[
            _model(
                AccessRequestItem,
                type=ITEM_TYPE,
                id=access_profile_id,
                comment=_text(comment) or None,
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
    access_profile_name: str,
    identity_name: str,
    comment: str | None = None,
    *,
    find_profile: Callable[[str], Sequence[dict[str, str]]] | None = None,
    find_identity: Callable[[str], Sequence[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Resolve both names, submit, summarize. Errors come back as data.

    The two finders are injectable so the resolution rules can be tested
    without a tenant.
    """
    try:
        profile = resolve(
            access_profile_name, "access_profile_name", find_profile or find_access_profiles
        )
        identity = resolve(
            identity_name, "identity_name", find_identity or find_identities
        )
    except ResolutionError as exc:
        return {"submitted": False, "error": str(exc)}
    except Exception as exc:  # the lookup itself failed (auth, network, ...)
        log.exception("request_access_profile lookup failed")
        return {"submitted": False, "error": describe_api_error(exc)}

    if profile.get("requestable") is False:
        # ISC rejects this with a 400 anyway; catching it here keeps a pointless
        # write attempt off the tenant and explains the cause in one sentence.
        return {
            "submitted": False,
            "access_profile_name": profile["name"],
            "access_profile_id": profile["id"],
            "error": (
                f"{profile['name']!r} is not requestable, so it cannot be "
                "requested through Access Request. Someone with admin rights "
                "has to mark the access profile requestable in ISC, or pick a "
                "profile that already is."
            ),
        }

    resolved = {
        "access_profile_name": profile["name"],
        "access_profile_id": profile["id"],
        "identity_name": identity["name"],
        "identity_id": identity["id"],
    }
    access_request = build_access_request(profile["id"], identity["id"], comment)
    log.info(
        "request_access_profile: granting %s to %s", profile["name"], identity["name"]
    )

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
            **resolved,
            "request_body": access_request.to_dict(),
            "error": describe_api_error(exc),
        }

    result: dict[str, Any] = {
        **resolved,
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
        access_profile_name: str,
        identity_name: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Submit an access request granting one access profile to one person.

        This WRITES to SailPoint: it files a real access request and starts the
        approval workflow, so only call it when the user has asked for that
        access to be requested for a specific person. Do not call it to explore
        or to check what access exists -- use `search_identities` for a person's
        access and the entitlements resource for what a source offers.

        Both inputs are display names, not IDs: the tool looks each one up and
        submits the request with the IDs it finds. If a name matches nothing, or
        matches several objects, nothing is submitted and the error names the
        candidates -- pass a more exact name, or the 32-character ID, which is
        also accepted in either field.

        Args:
            access_profile_name: The access profile to request, by name, e.g.
                `Accounts Receivable - AD`. An exact name wins; otherwise a
                starts-with match is tried. The profile must be marked
                requestable in ISC -- one that is not is refused here, before
                anything is submitted.
            identity_name: The person the access is for -- who will receive it,
                not the requester. Their display name or username, e.g.
                `Aaron.Nichols`.
            comment: Business justification recorded on the request and shown to
                approvers, e.g. "Joining the finance team". Include the user's
                stated reason when they gave one.

        Returns:
            A dict with `submitted`, the resolved `access_profile_name` /
            `access_profile_id` / `identity_name` / `identity_id` so you can see
            exactly who got what, `request_type`, and `access_request_ids` for
            tracking. When the same access is already pending, `submitted` is
            False and `already_requested` is True -- report that rather than
            claiming a new request was filed. On failure, `submitted` is False
            and `error` explains why.
        """
        return submit(access_profile_name, identity_name, comment)
