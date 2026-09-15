from __future__ import annotations

import pytest

from agents.sql_risk_scope_extraction import (
    MAX_SQL_RISK_SCOPE_EXTRACTION_ATTEMPTS,
    SQL_RISK_SCOPE_EXTRACTION_PROMPT,
    SqlRiskScopeExtraction,
    render_sql_risk_scope_extraction_repair,
    validate_sql_risk_scope_extraction,
)


def _payload(
    *,
    mode: str = "row_filtering",
    source_table: str = "src_orders",
    target_table: str = "tgt_orders",
    source_field: str | None = None,
    target_field: str | None = None,
    source_table_attestation: str | None = None,
    target_table_attestation: str | None = None,
    source_field_attestation: str | None = None,
    target_field_attestation: str | None = None,
    file_id: int | None = None,
    file_attestation: str | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {"table_name": source_table}
    target: dict[str, object] = {"table_name": target_table}
    source_origin: dict[str, object] = {
        "table_name": source_table_attestation or source_table,
    }
    target_origin: dict[str, object] = {
        "table_name": target_table_attestation or target_table,
    }
    if source_field is not None:
        source["field_name"] = source_field
        source_origin["field_name"] = (
            source_field_attestation or source_field
        )
    if target_field is not None:
        target["field_name"] = target_field
        target_origin["field_name"] = (
            target_field_attestation or target_field
        )
    payload: dict[str, object] = {
        "execution_mode": mode,
        "source": source,
        "target": target,
        "origin": {
            "source": source_origin,
            "target": target_origin,
        },
    }
    if file_id is not None:
        payload["file_id"] = file_id
    if file_attestation is not None:
        payload["origin"]["file_id"] = file_attestation
    return payload


@pytest.mark.parametrize(
    ("mode", "source_field", "target_field"),
    [
        ("row_filtering", None, None),
        ("conditional_cardinality", None, None),
        ("write_semantics", None, None),
        ("nullable_constraint", "order_id", "order_id"),
        ("value_changes", "order_id", "order_id"),
    ],
)
def test_native_schema_owns_all_five_closed_modes(
    mode: str,
    source_field: str | None,
    target_field: str | None,
):
    source = "src_orders"
    target = "tgt_orders"
    if source_field is not None:
        source = f"{source}.{source_field}"
        target = f"{target}.{target_field}"
    file_id = 17 if mode == "nullable_constraint" else None
    file_attestation = "17" if file_id is not None else None
    task = f"Проверь {source} → {target}"
    if file_attestation:
        task += f", {file_attestation}."

    result = validate_sql_risk_scope_extraction(
        _payload(
            mode=mode,
            source_field=source_field,
            target_field=target_field,
            file_id=file_id,
            file_attestation=file_attestation,
        ),
        original_task=task,
    )

    assert result.status == "valid"
    assert result.contract is not None
    assert result.contract.execution_mode == mode
    assert result.issues == []


def test_schema_exposes_closed_modes_and_omits_absent_optional_values():
    schema = SqlRiskScopeExtraction.model_json_schema()

    assert schema["properties"]["execution_mode"]["enum"] == [
        "row_filtering",
        "conditional_cardinality",
        "nullable_constraint",
        "value_changes",
        "write_semantics",
    ]
    assert set(schema["required"]) == {
        "execution_mode",
        "source",
        "target",
        "origin",
    }
    endpoint_schema = schema["$defs"]["SqlRiskScopeEndpoint"]
    assert set(endpoint_schema["required"]) == {"table_name"}
    origin_schema = schema["$defs"]["SqlRiskScopeOriginAttestation"]
    assert set(origin_schema["required"]) == {"source", "target"}


def test_verbatim_endpoint_locations_are_computed_without_model_offsets():
    task = "Префикс. Из src_orders данные поступают в tgt_orders."

    result = validate_sql_risk_scope_extraction(
        _payload(),
        original_task=task,
    )

    assert result.status == "valid"
    assert result.origin_locations is not None
    source_start = task.index("src_orders")
    target_start = task.index("tgt_orders")
    assert result.origin_locations.source_table_start == source_start
    assert result.origin_locations.source_table_end == (
        source_start + len("src_orders")
    )
    assert result.origin_locations.target_table_start == target_start
    assert result.origin_locations.target_table_end == (
        target_start + len("tgt_orders")
    )


def test_backtick_fields_and_unlabelled_file_id_token_are_valid():
    task = (
        "Для файла 27 перенеси `ODS.Orders.Order_ID` в "
        "`DM.Orders.Order_ID` и оцени nullable."
    )

    result = validate_sql_risk_scope_extraction(
        _payload(
            mode="nullable_constraint",
            source_table="ODS.Orders",
            source_field="Order_ID",
            target_table="DM.Orders",
            target_field="Order_ID",
            source_table_attestation="ODS.Orders",
            source_field_attestation="Order_ID",
            target_table_attestation="DM.Orders",
            target_field_attestation="Order_ID",
            file_id=27,
            file_attestation="27",
        ),
        original_task=task,
    )

    assert result.status == "valid"
    assert result.origin_locations is not None
    assert result.origin_locations.file_id_start == task.index("27")


def test_fabricated_or_normalized_endpoint_is_rejected_without_resolution():
    result = validate_sql_risk_scope_extraction(
        _payload(
            source_table="src_order",
        ),
        original_task="Проверь src_orders → tgt_orders.",
    )

    assert result.status == "invalid"
    assert result.contract is None
    assert [issue.code for issue in result.issues] == [
        "endpoint_not_bounded"
    ]


def test_separately_worded_table_and_field_components_are_valid():
    task = (
        "source table src_orders, source field customer_id; "
        "target table dim_customer, target field customer_key; file 17"
    )

    result = validate_sql_risk_scope_extraction(
        _payload(
            mode="nullable_constraint",
            source_table="src_orders",
            source_field="customer_id",
            target_table="dim_customer",
            target_field="customer_key",
            file_id=17,
            file_attestation="17",
        ),
        original_task=task,
    )

    assert result.status == "valid"
    assert result.origin_locations is not None
    assert result.origin_locations.source_table_start == task.index("src_orders")
    assert result.origin_locations.source_field_start == task.index("customer_id")
    assert result.origin_locations.target_table_start == task.index("dim_customer")
    assert result.origin_locations.target_field_start == task.index("customer_key")


@pytest.mark.parametrize(
    ("task", "source", "target"),
    [
        ("Проверь foo-bar → baz.", "bar", "baz"),
        ("Проверь foo → bar/baz.", "foo", "bar"),
        ("Проверь foo → bar::virtual.", "foo", "bar"),
    ],
)
def test_partial_endpoint_attestation_is_rejected(
    task: str,
    source: str,
    target: str,
):
    result = validate_sql_risk_scope_extraction(
        _payload(
            source_table=source,
            target_table=target,
        ),
        original_task=task,
    )

    assert result.status == "invalid"
    assert result.contract is None
    assert "endpoint_not_bounded" in {
        issue.code for issue in result.issues
    }


def test_origin_verifier_does_not_reclassify_model_owned_roles_from_wording():
    result = validate_sql_risk_scope_extraction(
        _payload(
            source_table="tgt_orders",
            target_table="src_orders",
        ),
        original_task="Проверь src_orders → tgt_orders.",
    )

    assert result.status == "valid"
    assert result.contract is not None
    assert result.contract.source.table_name == "tgt_orders"
    assert result.contract.target.table_name == "src_orders"


@pytest.mark.parametrize(
    ("task", "attestation"),
    [
        ("file_id=17.5, src.a → tgt.a", "17"),
        ("file_id=17-18, src.a → tgt.a", "17"),
        ("xfile_id=17x, src.a → tgt.a", "17"),
    ],
)
def test_partial_file_id_attestation_is_rejected(
    task: str,
    attestation: str,
):
    result = validate_sql_risk_scope_extraction(
        _payload(
            mode="nullable_constraint",
            source_table="src",
            source_field="a",
            target_table="tgt",
            target_field="a",
            file_id=17,
            file_attestation=attestation,
        ),
        original_task=task,
    )

    assert result.status == "invalid"
    assert result.contract is None
    assert [issue.code for issue in result.issues] == [
        "file_id_not_bounded"
    ]


def test_wrong_file_id_value_is_a_clear_origin_error():
    result = validate_sql_risk_scope_extraction(
        _payload(
            mode="nullable_constraint",
            source_field="id",
            target_field="id",
            file_id=18,
            file_attestation="17",
        ),
        original_task="file_id=17: src_orders.id → tgt_orders.id",
    )

    assert result.status == "invalid"
    assert result.contract is None
    assert result.issues[0].code == "file_id_mismatch"
    assert result.issues[0].location == "origin.file_id"


@pytest.mark.parametrize(
    "payload",
    [
        _payload(mode="agentic"),
        _payload(mode="row_filtering", source_field="id", target_field="id"),
        _payload(mode="value_changes"),
        _payload(
            mode="nullable_constraint",
            source_field="id",
            target_field="id",
        ),
        {
            **_payload(),
            "unexpected": "not allowed",
        },
    ],
)
def test_invalid_native_shapes_fail_closed(payload: dict[str, object]):
    result = validate_sql_risk_scope_extraction(
        payload,
        original_task="src_orders → tgt_orders",
    )

    assert result.status == "invalid"
    assert result.contract is None
    assert result.origin_locations is None
    assert result.issues
    assert {issue.code for issue in result.issues} == {"schema_error"}


def test_schema_does_not_coerce_string_file_id():
    payload = _payload(
        mode="nullable_constraint",
        source_field="id",
        target_field="id",
        file_attestation="17",
    )
    payload["file_id"] = "17"

    result = validate_sql_risk_scope_extraction(
        payload,
        original_task="file_id=17: src_orders.id → tgt_orders.id",
    )

    assert result.status == "invalid"
    assert result.contract is None
    assert result.issues[0].code == "schema_error"


def test_repair_is_caller_controlled_and_limited_to_one_retry():
    result = validate_sql_risk_scope_extraction(
        _payload(
            source_table_attestation="invented",
            target_table_attestation="pair",
        ),
        original_task="src_orders → tgt_orders",
    )
    assert result.status == "invalid"

    repair = render_sql_risk_scope_extraction_repair(result.issues)

    assert MAX_SQL_RISK_SCOPE_EXTRACTION_ATTEMPTS == 2
    assert "единственная разрешённая repair-попытка" in repair
    assert "endpoint_mismatch" in repair
    assert "agentic" not in repair


def test_prompt_assigns_mode_and_literals_to_the_model_without_sql_analysis():
    assert "Выбери один режим внутри этого pipeline" in (
        SQL_RISK_SCOPE_EXTRACTION_PROMPT
    )
    assert "Не анализируй SQL" in SQL_RISK_SCOPE_EXTRACTION_PROMPT
    assert "не требуй стрелку" in SQL_RISK_SCOPE_EXTRACTION_PROMPT
    assert "не выбирай приближённые" in SQL_RISK_SCOPE_EXTRACTION_PROMPT
    assert "без agentic fallback" in SQL_RISK_SCOPE_EXTRACTION_PROMPT
