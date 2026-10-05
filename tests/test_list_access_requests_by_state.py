"""Unit tests for the pure helpers and the paging loop -- the API is faked."""

import asyncio

from mcp.server import MCPServer
from sailpoint.exceptions import ApiException

from sailpoint_mcp import tools
from sailpoint_mcp.tools.list_access_requests_by_state import (
    PAGE_SIZE,
    SAMPLE_PER_STATE,
    build_filters,
    collect,
    group_by_state,
    summarize_request,
)


def record(index: int, state: str = "EXECUTING") -> dict:
    """One admin status record, in the shape `to_dict()` produces."""
    return {
        "accessRequestId": f"ar{index}",
        "name": f"Access Profile {index}",
        "type": "ACCESS_PROFILE",
        "state": state,
        "requestedFor": {"id": "id1", "name": "Aaron.Nichols"},
        "requester": {"id": "id2", "name": "hack.day"},
    }


def pager(total: int, state: str = "EXECUTING"):
    """A fake `fetch` that serves `total` records in pages, recording calls."""
    calls: list[tuple[int, int]] = []

    def fetch(offset: int, limit: int) -> list[dict]:
        calls.append((offset, limit))
        return [record(i, state) for i in range(offset, min(offset + limit, total))]

    return fetch, calls


def test_paginates_across_three_pages():
    fetch, calls = pager(PAGE_SIZE * 2 + 7)  # 507 records -> 250, 250, 7

    result = collect(fetch=fetch, max_results=5000)

    assert [limit for _, limit in calls] == [PAGE_SIZE, PAGE_SIZE, PAGE_SIZE]
    assert [offset for offset, _ in calls] == [0, PAGE_SIZE, PAGE_SIZE * 2]
    assert result["total"] == 507
    assert result["truncated"] is False
    assert result["by_state"]["EXECUTING"]["count"] == 507


def test_a_short_first_page_ends_the_paging():
    fetch, calls = pager(3)

    assert collect(fetch=fetch)["total"] == 3
    assert len(calls) == 1  # no pointless second request


def test_groups_by_state():
    records = [record(1, "EXECUTING"), record(2, "ERROR"), record(3, "EXECUTING")]

    result = collect(fetch=lambda offset, limit: records if offset == 0 else [])

    assert result["by_state"]["EXECUTING"]["count"] == 2
    assert result["by_state"]["ERROR"]["count"] == 1
    assert result["total"] == 3


def test_states_filter_keeps_only_those_states():
    records = [
        record(1, "EXECUTING"),
        record(2, "ERROR"),
        record(3, "PROVISIONING_FAILED"),
    ]
    fetch = lambda offset, limit: records if offset == 0 else []  # noqa: E731

    result = collect(states=["error", "provisioning_failed"], fetch=fetch)

    assert set(result["by_state"]) == {"ERROR", "PROVISIONING_FAILED"}
    assert result["total"] == 2


def test_max_results_truncates_and_flags_it():
    fetch, calls = pager(1000)

    result = collect(fetch=fetch, max_results=PAGE_SIZE * 2)

    assert result["total"] == PAGE_SIZE * 2
    assert result["truncated"] is True
    assert "max_results" in result["note"]
    assert len(calls) == 2  # stopped at the cap, did not keep paging


def test_max_results_is_clamped_to_the_hard_maximum():
    fetch, calls = pager(10)
    collect(fetch=fetch, max_results=999_999)
    # First page is still a normal page, not a 999999-row ask.
    assert calls[0][1] == PAGE_SIZE


def test_samples_are_capped_unless_include_items():
    records = [record(i) for i in range(20)]
    fetch = lambda offset, limit: records if offset == 0 else []  # noqa: E731

    sampled = collect(fetch=fetch)
    assert sampled["by_state"]["EXECUTING"]["count"] == 20
    assert len(sampled["by_state"]["EXECUTING"]["requests"]) == SAMPLE_PER_STATE

    full = collect(fetch=fetch, include_items=True)
    assert len(full["by_state"]["EXECUTING"]["requests"]) == 20


def test_forbidden_explains_that_admin_rights_are_missing():
    def fetch(offset, limit):
        raise ApiException(status=403, reason="Forbidden")

    result = collect(fetch=fetch)

    assert result["by_state"] == {}
    assert "admin" in result["error"].lower()
    assert "ORG_ADMIN" in result["error"]
    assert "403" in result["error"]


def test_unauthorized_gets_the_same_treatment():
    def fetch(offset, limit):
        raise ApiException(status=401, reason="Unauthorized")

    assert "admin" in collect(fetch=fetch)["error"].lower()


def test_created_after_becomes_a_filter_expression():
    assert build_filters("2026-09-01T00:00:00Z") == 'created gt "2026-09-01T00:00:00Z"'
    assert build_filters("  ") is None
    assert build_filters(None) is None


def test_summary_is_flat_and_drops_empties():
    summary = summarize_request(
        {
            "accessRequestId": "ar1",
            "name": "Accounts Receivable - AD",
            "type": "ACCESS_PROFILE",
            "requestedFor": {"id": "id1", "name": "Aaron.Nichols"},
            "requester": {"id": "id2", "name": "hack.day"},
            "approvalDetails": [
                {"status": "PENDING", "currentOwner": {"id": "o1", "name": "Ada"}},
                {"status": "APPROVED", "currentOwner": {"id": "o2", "name": "Grace"}},
            ],
            "removeDate": None,
        }
    )

    assert summary["requested_for"] == "Aaron.Nichols (id1)"
    assert summary["requester"] == "hack.day (id2)"
    assert summary["current_approvers"] == ["Ada (o1)"]  # only who it waits on
    assert "removal_date" not in summary
    assert "created" not in summary


def test_timestamps_are_rendered_as_strings():
    from datetime import datetime, timezone

    summary = summarize_request(
        {
            "accessRequestId": "ar1",
            "created": datetime(2026, 10, 5, 17, 16, 4, tzinfo=timezone.utc),
        }
    )
    assert summary["created"] == "2026-10-05T17:16:04+00:00"


def test_state_enums_and_missing_states_are_handled():
    class Enum:
        value = "ERROR"

    grouped = group_by_state([{"state": Enum()}, {"state": None}])
    assert grouped["ERROR"]["count"] == 1
    assert grouped["UNKNOWN"]["count"] == 1


def test_tool_is_registered_alongside_search_identities():
    mcp = MCPServer("test")
    registered = tools.register_all(mcp)
    assert "list_access_requests_by_state" in registered

    listed = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    assert {"list_access_requests_by_state", "search_identities"} <= set(listed)
    schema = listed["list_access_requests_by_state"].input_schema
    assert set(schema["properties"]) == {
        "states",
        "created_after",
        "requested_for",
        "max_results",
        "include_items",
    }
    assert not schema.get("required")  # every input is optional
