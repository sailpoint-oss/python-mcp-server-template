"""Unit tests for the pure helpers -- no tenant or credentials required."""

import asyncio
import json

import pytest
from mcp.server import MCPServer

from sailpoint_mcp import tools
from sailpoint_mcp.tools.list_source_entitlements import (
    build_filters,
    summarize_entitlement,
)

SOURCE_ID = "2c9180835d191a86015d28455b4a2329"


def test_source_scope_is_always_present():
    assert build_filters(SOURCE_ID) == f'source.id eq "{SOURCE_ID}"'


def test_requestable_and_name_are_anded_in():
    assert build_filters(SOURCE_ID, requestable=True, name="Payroll") == (
        f'source.id eq "{SOURCE_ID}" and requestable eq true and name sw "Payroll"'
    )
    assert "requestable eq false" in build_filters(SOURCE_ID, requestable=False)


def test_extra_filter_is_parenthesised_so_it_cannot_widen_the_scope():
    filters = build_filters(SOURCE_ID, extra='name sw "A" or name sw "B"')
    assert filters == (
        f'source.id eq "{SOURCE_ID}" and (name sw "A" or name sw "B")'
    )


def test_quotes_in_a_value_are_escaped():
    assert build_filters(SOURCE_ID, name='say "hi"') == (
        f'source.id eq "{SOURCE_ID}" and name sw "say \\"hi\\""'
    )


def test_missing_source_id_is_rejected():
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="source_id is required"):
            build_filters(bad)


def test_summarize_flattens_and_keeps_false_flags():
    summary = summarize_entitlement(
        {
            "id": "ent1",
            "name": "CN=Admins",
            "sourceSchemaObjectType": "group",
            "attribute": "memberOf",
            "value": "CN=Admins,DC=example",
            "requestable": False,
            "cloudGoverned": False,
            "privilegeLevel": {"direct": "HIGH", "inherited": None},
            "owner": {"id": "o1", "name": "Ada Lovelace"},
            "source": {"id": SOURCE_ID, "name": "Active Directory"},
            "description": None,
            "tags": [],
        }
    )

    assert summary["name"] == "CN=Admins"
    assert summary["type"] == "group"
    assert summary["requestable"] is False  # a real answer, not an empty value
    assert summary["cloud_governed"] is False
    assert summary["privilege_level"] == "HIGH"
    assert summary["owner"] == "Ada Lovelace"
    assert summary["source_id"] == SOURCE_ID
    assert "description" not in summary  # None dropped
    assert "tags" not in summary  # [] dropped


def test_summarize_truncates_a_long_description():
    summary = summarize_entitlement({"id": "e", "description": "x" * 500})
    assert summary["description"].endswith("...")
    assert len(summary["description"]) == 303


def test_summarize_renders_timestamps_as_json_safe_strings():
    # `to_dict()` on an SDK model yields real datetimes, which json.dumps refuses.
    from datetime import datetime, timezone

    summary = summarize_entitlement(
        {"id": "e", "created": datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)}
    )
    assert summary["created"] == "2024-01-02T03:04:05+00:00"
    json.dumps(summary)


def test_summarize_tolerates_a_sparse_sdk_model():
    from sailpoint.entitlements.models.entitlement_v2 import EntitlementV2

    # The SDK model defaults both flags to False, so they ride along.
    assert summarize_entitlement(EntitlementV2.from_dict({"id": "e1"})) == {
        "id": "e1",
        "requestable": False,
        "cloud_governed": False,
    }


def test_entitlements_are_a_resource_and_not_a_tool():
    mcp = MCPServer("test")
    assert "list_source_entitlements" in tools.register_all(mcp)

    # Deliberately not a tool -- it is reached by URI.
    assert "list_source_entitlements" not in {t.name for t in asyncio.run(mcp.list_tools())}

    templates = {t.name: t for t in asyncio.run(mcp.list_resource_templates())}
    template = templates["list_source_entitlements"]
    assert str(template.uri_template) == "sailpoint://sources/{source_id}/entitlements"
    assert template.description
