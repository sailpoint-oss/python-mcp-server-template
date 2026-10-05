"""Unit tests for the pure helpers -- no tenant, and nothing is submitted."""

import asyncio
import json

import pytest
from mcp.server import MCPServer

from sailpoint_mcp import tools
from sailpoint_mcp.tools.request_access_profile import (
    build_access_request,
    submit,
    summarize_response,
)

IDENTITY_ID = "2c91808568c529c60168cca6f90c1313"
PROFILE_ID = "2c9180835d2e5168015d32f890ca1581"
COMMENT = "Joining the finance team"


def body_for(profile_id, identity_id, comment=None):
    return json.loads(build_access_request(profile_id, identity_id, comment).to_json())


def test_request_body_matches_the_documented_payload():
    assert body_for(PROFILE_ID, IDENTITY_ID, COMMENT) == {
        "requestedFor": [IDENTITY_ID],
        "requestType": "GRANT_ACCESS",
        "requestedItems": [
            {"type": "ACCESS_PROFILE", "id": PROFILE_ID, "comment": COMMENT}
        ],
    }


def test_a_blank_comment_is_omitted_rather_than_sent_as_null():
    for comment in (None, "", "   "):
        body = body_for(PROFILE_ID, IDENTITY_ID, comment)
        assert body["requestedItems"] == [
            {"type": "ACCESS_PROFILE", "id": PROFILE_ID}
        ]


def test_ids_are_trimmed():
    body = body_for(f"  {PROFILE_ID} ", f"\t{IDENTITY_ID}\n")
    assert body["requestedFor"] == [IDENTITY_ID]
    assert body["requestedItems"][0]["id"] == PROFILE_ID


@pytest.mark.parametrize(
    "profile_id,identity_id,expected",
    [
        ("", IDENTITY_ID, "access_profile_id is required"),
        ("   ", IDENTITY_ID, "access_profile_id is required"),
        (None, IDENTITY_ID, "access_profile_id is required"),
        (PROFILE_ID, "", "identity_id is required"),
        (PROFILE_ID, None, "identity_id is required"),
    ],
)
def test_missing_ids_are_rejected_before_any_api_call(profile_id, identity_id, expected):
    with pytest.raises(ValueError, match=expected):
        build_access_request(profile_id, identity_id)

    # And the tool path turns that into data, without reaching the tenant.
    result = submit(profile_id, identity_id)
    assert result["submitted"] is False
    assert expected in result["error"]


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


def test_tool_asks_for_the_two_ids():
    mcp = MCPServer("test")
    assert "request_access_profile" in tools.register_all(mcp)

    listed = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    schema = listed["request_access_profile"].input_schema
    assert set(schema["required"]) == {"access_profile_id", "identity_id"}
    assert "comment" in schema["properties"]
    assert "WRITES" in listed["request_access_profile"].description
