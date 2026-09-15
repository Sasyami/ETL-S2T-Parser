"""Dependency-based exact readers for the validation-protocol pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .test_protocol import (
    ResolvedTestProtocolContract,
    check_dependencies,
)
from .tools import (
    list_source_column_catalog,
    list_target_column_catalog,
    read_s2t_by_target_table,
    read_s2t_source_to_target,
)
from .tools.saved_results import get_active_saved_result_store


def _persist_full_result(
    *,
    tool_name: str,
    call_id: str,
    payload: Dict[str, Any],
) -> None:
    store = get_active_saved_result_store()
    if store is None or payload.get("error"):
        return
    store.save_payload(
        source_tool=tool_name,
        source_tool_call_id=call_id,
        payload=payload,
    )


def _invoke(tool: Any, args: Dict[str, Any], callbacks: Sequence[Any]) -> Any:
    config = {"callbacks": list(callbacks)} if callbacks else None
    if config is not None:
        return tool.invoke(args, config=config)
    return tool.invoke(args)


def read_test_protocol_inputs(
    contract: ResolvedTestProtocolContract,
    *,
    callbacks: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    """Read only dependencies required by a resolved validation protocol.

    Exact global S2T readers establish every directed load and feed the static
    preflight. File-scoped catalogs are added only when a selected check needs
    them. A missing file selector therefore makes only catalog-dependent
    checks unavailable; it never aborts the whole protocol.
    """

    results: List[Dict[str, Any]] = []

    def append_read(
        *,
        load_index: int,
        kind: str,
        tool_name: str,
        call_id: str,
        args: Dict[str, Any],
        tool: Any,
    ) -> None:
        try:
            payload = _invoke(tool, args, callbacks)
        except Exception as exc:
            # A reader outage is an unavailable protocol dependency. It must
            # not crash /chat or switch the request to the agentic pipeline.
            payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "rows": [],
            }
        if not isinstance(payload, dict):
            payload = {
                "error": f"{tool_name} вернул результат неизвестного формата.",
                "rows": [],
            }
        if payload.get("error"):
            results.append(
                {
                    "kind": "reader_issue",
                    "load_index": load_index,
                    "args": args,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "error": str(payload.get("error")),
                }
            )
            if kind not in {"source_column_catalog", "target_column_catalog"}:
                results.append(
                    {
                        "kind": kind,
                        "load_index": load_index,
                        "args": args,
                        "tool_name": tool_name,
                        "call_id": call_id,
                        "payload": {
                            "columns": [],
                            "rows": [],
                            "returned_rows": 0,
                            "truncated": False,
                        },
                    }
                )
            return
        result = {
            "kind": kind,
            "load_index": load_index,
            "args": args,
            "tool_name": tool_name,
            "call_id": call_id,
            "payload": payload,
        }
        results.append(result)
        _persist_full_result(
            tool_name=tool_name,
            call_id=call_id,
            payload=payload,
        )

    for load_index, load in enumerate(contract.loads, start=1):
        checks = contract.checks_for_load(load)
        dependencies = set(check_dependencies(checks))
        explicit_key = contract.explicit_key_for_load(load)

        for source_index, source_table in enumerate(load.sources, start=1):
            append_read(
                load_index=load_index,
                kind="s2t_pair",
                tool_name="read_s2t_source_to_target",
                call_id=f"test_protocol_pair_{load_index}_{source_index}",
                args={
                    "source_table": source_table,
                    "target_table": load.target,
                },
                tool=read_s2t_source_to_target,
            )
        # A sole row-count check needs only the exact directed mapping. Other
        # protocols retain the complete target mapping for phase-zero coverage.
        if checks != ["row_count"]:
            append_read(
                load_index=load_index,
                kind="s2t_target",
                tool_name="read_s2t_by_target_table",
                call_id=f"test_protocol_target_{load_index}",
                args={"target_table": load.target},
                tool=read_s2t_by_target_table,
            )

        needs_target_catalog = bool(
            dependencies & {"target_catalog", "required_fields"}
        ) or ("comparison_key" in dependencies and not explicit_key)
        needs_source_catalog = "source_catalog" in dependencies
        if contract.file_id is None:
            continue
        if needs_target_catalog:
            append_read(
                load_index=load_index,
                kind="target_column_catalog",
                tool_name="list_target_column_catalog",
                call_id=f"test_protocol_target_catalog_{load_index}",
                args={
                    "file_id": contract.file_id,
                    "table_name": load.target,
                },
                tool=list_target_column_catalog,
            )
        if needs_source_catalog:
            for source_index, source_table in enumerate(load.sources, start=1):
                append_read(
                    load_index=load_index,
                    kind="source_column_catalog",
                    tool_name="list_source_column_catalog",
                    call_id=(
                        "test_protocol_source_catalog_"
                        f"{load_index}_{source_index}"
                    ),
                    args={
                        "file_id": contract.file_id,
                        "table_name": source_table,
                    },
                    tool=list_source_column_catalog,
                )

    return results


__all__ = ["read_test_protocol_inputs"]
