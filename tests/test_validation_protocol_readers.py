from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.test_protocol import (
    ResolvedTestProtocolContract,
    TestProtocolLoad,
    compile_test_protocol,
)
from agents.tools.common import pack_tabular_rows
from agents import validation_protocol


def _s2t_payload() -> dict[str, Any]:
    rows = [
        {
            "file_id": 7,
            "sheet_name": "S2T",
            "row_num": 1,
            "source_table": "stage.orders",
            "source_field": "order_id",
            "target_table": "mart.orders",
            "target_field": "order_id",
            "transformation_rule": (
                "SELECT s.order_id FROM stage.orders AS s"
            ),
            "source_layer": "A",
            "target_layer": "B",
        }
    ]
    return {
        **pack_tabular_rows(
            rows,
            columns=list(rows[0]),
            dictionary_columns=(
                "sheet_name",
                "source_table",
                "target_table",
                "transformation_rule",
                "source_layer",
                "target_layer",
            ),
        ),
        "returned_rows": 1,
        "truncated": False,
    }


def _catalog_payload(role: str) -> dict[str, Any]:
    rows = [
        {
            "table_name": f"{role}.orders",
            "column_name": "order_id",
            "data_type": "bigint",
            "primary_key": role == "mart",
            "not_null": True,
        }
    ]
    return {"columns": list(rows[0]), "rows": rows, "truncated": False}


@dataclass
class _FakeTool:
    name: str
    payload: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, args, config=None):
        self.calls.append({"args": dict(args), "config": config})
        return self.payload


@dataclass
class _RaisingTool:
    name: str
    error: Exception
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, args, config=None):
        self.calls.append({"args": dict(args), "config": config})
        raise self.error


def _install_readers(monkeypatch):
    pair = _FakeTool("read_s2t_source_to_target", _s2t_payload())
    target = _FakeTool("read_s2t_by_target_table", _s2t_payload())
    source_catalog = _FakeTool(
        "list_source_column_catalog",
        _catalog_payload("stage"),
    )
    target_catalog = _FakeTool(
        "list_target_column_catalog",
        _catalog_payload("mart"),
    )
    monkeypatch.setattr(validation_protocol, "read_s2t_source_to_target", pair)
    monkeypatch.setattr(validation_protocol, "read_s2t_by_target_table", target)
    monkeypatch.setattr(
        validation_protocol,
        "list_source_column_catalog",
        source_catalog,
    )
    monkeypatch.setattr(
        validation_protocol,
        "list_target_column_catalog",
        target_catalog,
    )
    return pair, target, source_catalog, target_catalog


def _contract(
    checks: list[str],
    *,
    file_id: int | None = None,
    explicit_key: list[str] | None = None,
) -> ResolvedTestProtocolContract:
    return ResolvedTestProtocolContract(
        file_id=file_id,
        loads=[
            TestProtocolLoad(
                sources=["stage.orders"],
                target="mart.orders",
                checks=checks,
                explicit_key=explicit_key,
            )
        ],
        mode="explicit",
    )


def test_row_count_reads_only_directed_s2t_without_catalog(monkeypatch):
    pair, target, source_catalog, target_catalog = _install_readers(monkeypatch)

    results = validation_protocol.read_test_protocol_inputs(
        _contract(["row_count"])
    )

    assert [item["kind"] for item in results] == ["s2t_pair"]
    assert pair.calls[0]["args"] == {
        "source_table": "stage.orders",
        "target_table": "mart.orders",
    }
    assert target.calls == []
    assert source_catalog.calls == []
    assert target_catalog.calls == []


def test_missing_file_only_marks_catalog_check_unavailable(monkeypatch):
    _install_readers(monkeypatch)
    contract = _contract(["row_count", "required_null_rate"])

    results = validation_protocol.read_test_protocol_inputs(contract)
    protocol = compile_test_protocol(contract, reader_results=results)

    checks = {item.kind: item for item in protocol.targets[0].checks}
    assert checks["row_count"].status == "ready"
    assert checks["required_null_rate"].status == "unavailable"
    assert checks["required_null_rate"].missing_dependencies == [
        "target_catalog"
    ]
    assert protocol.status == "partial_protocol"


def test_schema_check_reads_each_role_catalog_with_exact_scope_once(monkeypatch):
    pair, target, source_catalog, target_catalog = _install_readers(monkeypatch)
    callbacks = [object()]

    results = validation_protocol.read_test_protocol_inputs(
        _contract(["schema_compatibility"], file_id=7),
        callbacks=callbacks,
    )

    assert [item["kind"] for item in results] == [
        "s2t_pair",
        "s2t_target",
        "target_column_catalog",
        "source_column_catalog",
    ]
    assert len(pair.calls) == len(target.calls) == 1
    assert [item["args"] for item in target_catalog.calls] == [
        {"file_id": 7, "table_name": "mart.orders"}
    ]
    assert [item["args"] for item in source_catalog.calls] == [
        {"file_id": 7, "table_name": "stage.orders"}
    ]
    assert all(call["config"] == {"callbacks": callbacks} for call in (
        pair.calls + target.calls + source_catalog.calls + target_catalog.calls
    ))


def test_explicit_key_avoids_unneeded_target_catalog(monkeypatch):
    _pair, _target, source_catalog, target_catalog = _install_readers(monkeypatch)

    validation_protocol.read_test_protocol_inputs(
        _contract(
            ["key_reconciliation"],
            file_id=7,
            explicit_key=["order_id"],
        )
    )

    assert source_catalog.calls == []
    assert target_catalog.calls == []


def test_reader_error_is_structured_and_does_not_raise_or_add_catalog(monkeypatch):
    _pair, _target, _source_catalog, target_catalog = _install_readers(monkeypatch)
    target_catalog.payload = {"error": "catalog unavailable", "rows": []}

    results = validation_protocol.read_test_protocol_inputs(
        _contract(["required_null_rate"], file_id=7)
    )

    assert results[-1] == {
        "kind": "reader_issue",
        "load_index": 1,
        "args": {"file_id": 7, "table_name": "mart.orders"},
        "tool_name": "list_target_column_catalog",
        "call_id": "test_protocol_target_catalog_1",
        "error": "catalog unavailable",
    }


def test_reader_invoke_exception_becomes_unavailable_protocol(monkeypatch):
    pair, target, _source_catalog, _target_catalog = _install_readers(monkeypatch)
    raising_pair = _RaisingTool(
        "read_s2t_source_to_target",
        RuntimeError("database is temporarily unavailable"),
    )
    monkeypatch.setattr(
        validation_protocol,
        "read_s2t_source_to_target",
        raising_pair,
    )
    contract = _contract(["row_count"])

    results = validation_protocol.read_test_protocol_inputs(contract)
    protocol = compile_test_protocol(contract, reader_results=results)

    assert pair.calls == []
    assert target.calls == []
    assert results[0]["kind"] == "reader_issue"
    assert results[0]["tool_name"] == "read_s2t_source_to_target"
    assert results[0]["error"] == (
        "RuntimeError: database is temporarily unavailable"
    )
    assert results[1]["kind"] == "s2t_pair"
    assert results[1]["payload"]["rows"] == []
    assert protocol.status == "unavailable"
    check = protocol.targets[0].checks[0]
    assert check.kind == "row_count"
    assert check.status == "unavailable"
    assert "transformation" in check.missing_dependencies
