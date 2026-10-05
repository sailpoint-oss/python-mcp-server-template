"""Unit tests for name resolution and the request body -- the API is faked."""

import asyncio
import json

import pytest
from mcp.server import MCPServer

from sailpoint_mcp import tools
from sailpoint_mcp.tools.request_access_profile import (
    ResolutionError,
    build_access_request,
    resolve,
    submit,
    summarize_response,
)

IDENTITY_ID = "2c91808568c529c60168cca6f90c1313"
PROFILE_ID = "2c9180835d2e5168015d32f890ca1581"
COMMENT = "Joining the finance team"

PROFILES = [{"id": PROFILE_ID, "name": "Accounts Payable - AD", "requestable": True}]
IDENTITIES = [{"id": IDENTITY_ID, "name": "Aaron.Nichols", "email": "aaron@example.com"}]


def finder(results):
    """A fake lookup that records what it was asked for."""
    asked = []

    def find(name):
        asked.append(name)
        return results

    return find, asked


def test_names_resolve_to_ids_and_build_the_documented_payload(monkeypatch):
    sent = {}

    def fake_call(operation):
        # Only the create call reaches here; the finders are injected below.
        sent["request"] = operation.__closure__
        return {"newRequests": [{"accessRequestIds": ["ar1"]}]}

    monkeypatch.setattr(
        "sailpoint_mcp.tools.request_access_profile.call_sailpoint", fake_call
    )
    find_profile, profile_asked = finder(PROFILES)
    find_identity, identity_asked = finder(IDENTITIES)

    result = submit(
        "Accounts Payable - AD",
        "Aaron.Nichols",
        COMMENT,
        find_profile=find_profile,
        find_identity=find_identity,
    )

    assert profile_asked == ["Accounts Payable - AD"]
    assert identity_asked == ["Aaron.Nichols"]
    assert result["submitted"] is True
    assert result["access_profile_id"] == PROFILE_ID
    assert result["identity_id"] == IDENTITY_ID
    assert result["access_profile_name"] == "Accounts Payable - AD"
    assert result["identity_name"] == "Aaron.Nichols"
    assert result["access_request_ids"] == ["ar1"]


def test_request_body_matches_the_documented_payload():
    assert json.loads(build_access_request(PROFILE_ID, IDENTITY_ID, COMMENT).to_json()) == {
        "requestedFor": [IDENTITY_ID],
        "requestType": "GRANT_ACCESS",
        "requestedItems": [
            {"type": "ACCESS_PROFILE", "id": PROFILE_ID, "comment": COMMENT}
        ],
    }


def test_a_blank_comment_is_omitted_rather_than_sent_as_null():
    for comment in (None, "", "   "):
        body = json.loads(build_access_request(PROFILE_ID, IDENTITY_ID, comment).to_json())
        assert body["requestedItems"] == [{"type": "ACCESS_PROFILE", "id": PROFILE_ID}]


def test_an_exact_name_wins_over_other_partial_matches():
    find, _ = finder(
        [
            {"id": "a" * 32, "name": "Accounts Payable - AD"},
            {"id": "b" * 32, "name": "Accounts Payable - Workday"},
        ]
    )
    assert resolve("Accounts Payable - AD", "access_profile_name", find)["id"] == "a" * 32


def test_case_differences_still_count_as_an_exact_match():
    find, _ = finder(
        [
            {"id": "a" * 32, "name": "Accounts Payable - AD"},
            {"id": "b" * 32, "name": "Accounts Payable - Workday"},
        ]
    )
    assert resolve("accounts payable - ad", "access_profile_name", find)["id"] == "a" * 32


def test_an_ambiguous_name_names_the_candidates_and_submits_nothing():
    find_profile, _ = finder(
        [
            {"id": "a" * 32, "name": "Accounts Payable - AD"},
            {"id": "b" * 32, "name": "Accounts Receivable - AD"},
        ]
    )
    find_identity, identity_asked = finder(IDENTITIES)

    result = submit(
        "Accounts", "Aaron.Nichols", find_profile=find_profile, find_identity=find_identity
    )

    assert result["submitted"] is False
    assert "2 matches" in result["error"]
    assert "Accounts Payable - AD" in result["error"]
    assert "Accounts Receivable - AD" in result["error"]
    assert identity_asked == []  # stopped before the second lookup, and before submitting


def test_two_people_with_the_same_name_stop_the_request():
    find_identity, _ = finder(
        [
            {"id": "a" * 32, "name": "Aaron Nichols", "email": "aaron.n@example.com"},
            {"id": "b" * 32, "name": "Aaron Nichols", "email": "a.nichols@example.com"},
        ]
    )

    result = submit(
        "Accounts Payable - AD",
        "Aaron Nichols",
        find_profile=finder(PROFILES)[0],
        find_identity=find_identity,
    )

    assert result["submitted"] is False
    # The emails are what tells the two apart, so they have to be in the message.
    assert "aaron.n@example.com" in result["error"]
    assert "a.nichols@example.com" in result["error"]


def test_a_non_requestable_profile_is_refused_before_submitting(monkeypatch):
    def must_not_be_called(operation):
        raise AssertionError("submitted a request for a non-requestable profile")

    monkeypatch.setattr(
        "sailpoint_mcp.tools.request_access_profile.call_sailpoint", must_not_be_called
    )
    find_profile, _ = finder(
        [{"id": PROFILE_ID, "name": "Buyer - AD", "requestable": False}]
    )

    result = submit(
        "Buyer - AD",
        "Alan.Duffy",
        find_profile=find_profile,
        find_identity=finder(IDENTITIES)[0],
    )

    assert result["submitted"] is False
    assert "not requestable" in result["error"]
    assert result["access_profile_name"] == "Buyer - AD"
    assert result["access_profile_id"] == PROFILE_ID


def test_resolution_keeps_the_requestable_flag():
    find, _ = finder([{"id": PROFILE_ID, "name": "Buyer - AD", "requestable": False}])
    assert resolve("Buyer - AD", "access_profile_name", find)["requestable"] is False


def test_ambiguous_candidates_are_marked_when_not_requestable():
    find_profile, _ = finder(
        [
            {"id": "a" * 32, "name": "Accounts Payable - AD", "requestable": False},
            {"id": "b" * 32, "name": "Accounts Receivable - AD", "requestable": True},
        ]
    )

    error = submit(
        "Accounts", "Alan.Duffy", find_profile=find_profile, find_identity=finder(IDENTITIES)[0]
    )["error"]

    assert "Accounts Payable - AD (not requestable)" in error
    assert "Accounts Receivable - AD [" in error  # the usable one carries no marker


def test_a_name_that_matches_nothing_is_reported():
    result = submit(
        "No Such Profile",
        "Aaron.Nichols",
        find_profile=finder([])[0],
        find_identity=finder(IDENTITIES)[0],
    )

    assert result["submitted"] is False
    assert "No access_profile_name matched 'No Such Profile'" in result["error"]


def test_a_raw_id_skips_the_lookup_entirely():
    find, asked = finder(PROFILES)
    resolved = resolve(PROFILE_ID, "access_profile_name", find)

    assert resolved == {"id": PROFILE_ID, "name": PROFILE_ID, "resolved_from": "id"}
    assert asked == []  # no round trip


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_name_is_rejected(blank):
    with pytest.raises(ResolutionError, match="is required"):
        resolve(blank, "identity_name", finder(IDENTITIES)[0])


def test_a_failing_lookup_is_reported_as_data():
    def boom(name):
        raise RuntimeError("search exploded")

    result = submit("x", "y", find_profile=boom, find_identity=finder(IDENTITIES)[0])
    assert result["submitted"] is False
    assert "search exploded" in result["error"]


def test_new_request_is_reported_as_submitted():
    summary = summarize_response(
        {"newRequests": [{"requestedFor": IDENTITY_ID, "accessRequestIds": ["ar1"]}]}
    )

    assert summary["submitted"] is True
    assert summary["access_request_ids"] == ["ar1"]
    assert "already_requested" not in summary


def test_duplicate_request_is_not_reported_as_submitted():
    summary = summarize_response(
        {
            "newRequests": [],
            "existingRequests": [
                {"requestedFor": IDENTITY_ID, "accessRequestIds": ["ar0"]}
            ],
        }
    )

    assert summary["submitted"] is False
    assert summary["already_requested"] is True
    assert summary["access_request_ids"] == ["ar0"]
    assert "already requested" in summary["note"]


def test_summarize_tolerates_an_empty_response():
    assert summarize_response(None) == {"submitted": True, "access_request_ids": []}


def test_tool_asks_for_names_not_ids():
    mcp = MCPServer("test")
    assert "request_access_profile" in tools.register_all(mcp)

    listed = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    schema = listed["request_access_profile"].input_schema
    assert set(schema["required"]) == {"access_profile_name", "identity_name"}
    assert "comment" in schema["properties"]
    assert "WRITES" in listed["request_access_profile"].description
