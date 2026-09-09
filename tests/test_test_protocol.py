import json

import sqlglot

from agents.test_protocol import (
    CHECKS,
    LOAD_SCOPE_PREDICATE,
    PROTOCOL_CHECKS,
    SOURCE_SCOPE_PREDICATE,
    STANDARD_PROTOCOL_CHECKS,
    TARGET_SCOPE_PREDICATE,
    EntityResolutionMetadata,
    RawTestProtocolContract,
    RawTestProtocolLoad,
    TestProtocolContract,
    TestProtocolLoad,
    build_test_protocol_display_payloads,
    check_dependencies,
    compile_test_protocol,
    render_test_protocol_answer,
)
from agents.tools.common import pack_tabular_rows
from agents.transformation_ast import normalize_transformation
from services.sql_dialects import GREENPLUM_DIALECT


def test_protocol_contract_accepts_exact_filename_instead_of_file_id():
    contract = TestProtocolContract(
        filename="Mapping.xlsx",
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count"],
            )
        ],
    )

    assert contract.file_id is None
    assert contract.filename == "Mapping.xlsx"


def test_protocol_contract_allows_protocol_without_file_selector():
    contract = TestProtocolContract(
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count"],
            )
        ]
    )

    assert contract.file_id is None
    assert contract.filename is None


def test_raw_contract_preserves_load_roles_mode_key_and_explicit_file_id():
    contract = RawTestProtocolContract(
        file_id=17,
        mode="exhaustive",
        explicit_key="business_id",
        requested_checks=["row_count"],
        loads=[
            RawTestProtocolLoad(
                source_mentions=["src one", "src two"],
                target_mention="target one",
            ),
            RawTestProtocolLoad(
                source_mentions=["src three"],
                target_mention="target two",
            ),
        ],
    )

    assert contract.file_id == 17
    assert contract.explicit_key == ["business_id"]
    assert contract.source_mentions == ["src one", "src two", "src three"]
    assert contract.target_mentions == ["target one", "target two"]
    assert contract.loads[0].source_mentions == ["src one", "src two"]


def test_resolution_metadata_accepts_shared_resolver_methods_and_status():
    resolved = EntityResolutionMetadata(
        entity_type="source_table",
        mention="source",
        method="partial",
        status="resolved",
        canonical="stage.source_entity",
    )
    unresolved = EntityResolutionMetadata(
        entity_type="file",
        mention="unknown mapping",
        method="none",
        status="unresolved",
        error_code="unresolved_entity",
    )

    assert resolved.method == "partial"
    assert unresolved.error_code == "unresolved_entity"


def _reader_results(
    target_table="target_entity",
    *,
    load_index=1,
    source_table="source_entity",
    rule=(
        "SELECT s.entity_id, s.required_value "
        "FROM stage.source_entity AS s WHERE s.active = 1"
    ),
):
    s2t_rows = [
        {
            "source_table": source_table,
            "source_field": "entity_id",
            "target_table": target_table,
            "target_field": "entity_id",
            "transformation_rule": rule,
        },
        {
            "source_table": source_table,
            "source_field": "required_value",
            "target_table": target_table,
            "target_field": "required_value",
            "transformation_rule": rule,
        },
    ]
    s2t_payload = {
        **pack_tabular_rows(
            s2t_rows,
            columns=list(s2t_rows[0]),
            dictionary_columns=(
                "source_table",
                "target_table",
                "transformation_rule",
            ),
        ),
        "truncated": False,
    }
    catalog_rows = [
        {
            "table_name": target_table,
            "column_name": "ENTITY_ID",
            "data_type": "bigint",
            "primary_key": True,
            "not_null": True,
        },
        {
            "table_name": target_table,
            "column_name": "REQUIRED_VALUE",
            "data_type": "text",
            "primary_key": False,
            "not_null": True,
        },
    ]
    return [
        {
            "kind": "s2t_pair",
            "load_index": load_index,
            "args": {
                "source_table": source_table,
                "target_table": target_table,
            },
            "payload": s2t_payload,
        },
        {
            "kind": "s2t_target",
            "load_index": load_index,
            "args": {"target_table": target_table},
            "payload": s2t_payload,
        },
        {
            "kind": "target_column_catalog",
            "load_index": load_index,
            "args": {"file_id": 7, "table_name": target_table},
            "payload": {
                "columns": list(catalog_rows[0]),
                "rows": catalog_rows,
                "truncated": False,
            },
        },
    ]


def test_compiler_generates_four_greenplum_checks_without_execution():
    contract = TestProtocolContract(
        file_id=7,
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=list(STANDARD_PROTOCOL_CHECKS),
            )
        ],
    )

    protocol = compile_test_protocol(
        contract,
        reader_results=_reader_results(),
    )

    assert len(protocol.targets) == 1
    checks = {check.kind: check for check in protocol.targets[0].checks}
    assert set(checks) == set(STANDARD_PROTOCOL_CHECKS)
    assert all(
        LOAD_SCOPE_PREDICATE in check.sql_template for check in checks.values()
    )
    assert "expected_row_count" in checks["row_count"].sql_template
    assert 'GROUP BY "entity_id"' in checks["key_uniqueness"].sql_template
    assert '"required_value_null_count"' in checks[
        "required_null_rate"
    ].sql_template
    transformation_sql = checks["transformation_correctness"].sql_template
    assert "EXCEPT ALL" in transformation_sql
    assert 'src."entity_id" AS "entity_id"' in transformation_sql
    assert "difference_count = 0" in checks[
        "transformation_correctness"
    ].pass_criterion
    assert SOURCE_SCOPE_PREDICATE in checks["row_count"].sql_template
    assert SOURCE_SCOPE_PREDICATE in transformation_sql
    assert SOURCE_SCOPE_PREDICATE not in checks["key_uniqueness"].sql_template

    answer = render_test_protocol_answer(contract, protocol)
    assert "SQL-шаблоны не исполнялись" in answer
    assert answer.count("SQL-шаблон:") == 4
    assert answer.count("Критерий прохождения:") == 4
    displays = build_test_protocol_display_payloads(protocol)
    assert [item["name"] for item in displays] == [
        "read_s2t_source_to_target",
        "list_target_column_catalog",
    ]
    assert json.loads(displays[0]["content"])["load"] == {
        "sources": ["source_entity"],
        "target": "target_entity",
    }


def test_compiler_builds_independent_protocol_for_each_target():
    contract = TestProtocolContract(
        file_id=7,
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target=target,
                checks=list(STANDARD_PROTOCOL_CHECKS),
            )
            for target in ("first_target", "second_target")
        ],
    )
    results = [
        *_reader_results("first_target", load_index=1),
        *_reader_results("second_target", load_index=2),
    ]

    protocol = compile_test_protocol(contract, reader_results=results)
    answer = render_test_protocol_answer(contract, protocol)

    assert [target.target_table for target in protocol.targets] == [
        "first_target",
        "second_target",
    ]
    assert answer.count("Проверка количества строк") == 2
    assert 'FROM "first_target"' in answer
    assert 'FROM "second_target"' in answer


def test_compiler_marks_virtual_lineage_target_unavailable_without_sql():
    virtual_target = "mart.result::cte::src"
    contract = TestProtocolContract(
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target=virtual_target,
                checks=["row_count"],
            )
        ],
    )

    protocol = compile_test_protocol(
        contract,
        reader_results=_reader_results(target_table=virtual_target),
    )

    assert protocol.status == "unavailable"
    target = protocol.targets[0]
    assert target.status == "unavailable"
    assert target.target_table == virtual_target
    assert target.checks[0].status == "unavailable"
    assert target.checks[0].missing_dependencies == ["target_relation"]
    assert target.checks[0].sql_template.startswith(
        "-- SQL-шаблон не сформирован:"
    )
    assert "::" not in target.checks[0].sql_template
    assert any(
        item.kind == "target_relation_addressable"
        and item.status == "unavailable"
        and "виртуальным lineage scope" in item.conclusion
        for item in target.preflight
    )
    assert any(
        issue.code == "unavailable_check"
        and issue.check == "row_count"
        for issue in protocol.issues
    )


def test_compiler_rejects_expression_as_full_query():
    contract = TestProtocolContract(
        file_id=7,
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count", "transformation_correctness"],
            )
        ],
    )

    protocol = compile_test_protocol(
        contract,
        reader_results=_reader_results(rule="CAST(entity_id AS BIGINT)"),
    )

    assert all(
        check.sql_template.startswith("-- SQL-шаблон не сформирован:")
        for check in protocol.targets[0].checks
    )
    assert any(
        "не содержат полного SELECT/WITH" in limitation
        for limitation in protocol.targets[0].limitations
    )


def test_compiler_rejects_source_field_missing_from_select_output():
    contract = TestProtocolContract(
        file_id=7,
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["transformation_correctness"],
            )
        ],
    )

    protocol = compile_test_protocol(
        contract,
        reader_results=_reader_results(
            rule=(
                "SELECT s.entity_id AS renamed_id, "
                "s.required_value AS renamed_value "
                "FROM stage.source_entity AS s"
            )
        ),
    )

    check = protocol.targets[0].checks[0]
    target = protocol.targets[0]
    assert check.sql_template.startswith("-- SQL-шаблон не сформирован:")
    assert any(
        item.kind == "target_projection_consistency"
        and item.status == "fail"
        and "entity_id" in item.conclusion
        for item in target.preflight
    )


def _parse_compiled_sql(sql: str):
    parseable = sql.replace(SOURCE_SCOPE_PREDICATE, "TRUE").replace(
        TARGET_SCOPE_PREDICATE,
        "TRUE",
    )
    statements = sqlglot.parse(parseable, read=GREENPLUM_DIALECT)
    assert len(statements) == 1
    return statements[0]


def _full_reader_results(
    *,
    rule=(
        "SELECT s.entity_id, s.required_value "
        "FROM stage.source_entity AS s WHERE s.active = 1"
    ),
    mappings=None,
):
    mappings = mappings or [
        ("entity_id", "entity_id"),
        ("required_value", "required_value"),
    ]
    s2t_rows = [
        {
            "source_table": "source_entity",
            "source_field": source,
            "target_table": "target_entity",
            "target_field": target,
            "transformation_rule": rule,
        }
        for source, target in mappings
    ]
    target_types = {
        "entity_id": ("bigint", True, True),
        "required_value": ("text", False, True),
        "target_id": ("bigint", True, True),
        "target_value": ("text", False, True),
        "target_total": ("numeric", False, False),
    }
    target_rows = []
    for _, target in mappings:
        data_type, primary_key, not_null = target_types[target]
        target_rows.append(
            {
                "table_name": "target_entity",
                "column_name": target,
                "data_type": data_type,
                "primary_key": primary_key,
                "not_null": not_null,
            }
        )
    source_types = {
        "entity_id": "integer",
        "required_value": "varchar",
        "amount": "numeric",
        "tax": "numeric",
    }
    source_rows = [
        {
            "table_name": "source_entity",
            "column_name": source,
            "data_type": source_types[source],
        }
        for source, _ in mappings
    ]
    return [
        {
            "kind": "s2t_target",
            "load_index": 1,
            "args": {"target_table": "target_entity"},
            "payload": {"rows": s2t_rows, "truncated": False},
        },
        {
            "kind": "target_column_catalog",
            "load_index": 1,
            "args": {"file_id": 7, "table_name": "target_entity"},
            "payload": {"rows": target_rows, "truncated": False},
        },
        {
            "kind": "source_column_catalog",
            "load_index": 1,
            "args": {"file_id": 7, "table_name": "source_entity"},
            "payload": {"rows": source_rows, "truncated": False},
        },
    ]


def test_check_registry_covers_all_supported_checks_and_declares_dependencies():
    assert len(PROTOCOL_CHECKS) == 13
    assert tuple(CHECKS) == PROTOCOL_CHECKS
    assert all(definition.compiler for definition in CHECKS.values())
    assert {definition.phase for definition in CHECKS.values()} == {1, 2, 3}
    assert check_dependencies(["row_count"]) == ["transformation"]
    assert check_dependencies(["schema_compatibility"]) == [
        "source_catalog",
        "target_catalog",
    ]


def test_standard_and_exhaustive_modes_expand_checks_deterministically():
    standard = TestProtocolContract(
        mode="standard",
        loads=[TestProtocolLoad(sources=["source_entity"], target="target_entity")],
    )
    exhaustive = TestProtocolContract(
        mode="exhaustive",
        loads=[TestProtocolLoad(sources=["source_entity"], target="target_entity")],
    )

    standard_result = compile_test_protocol(
        standard,
        reader_results=_full_reader_results(),
    )
    exhaustive_result = compile_test_protocol(
        exhaustive,
        reader_results=_full_reader_results(),
    )

    assert [check.kind for check in standard_result.targets[0].checks] == list(
        STANDARD_PROTOCOL_CHECKS
    )
    assert [check.kind for check in exhaustive_result.targets[0].checks] == list(
        PROTOCOL_CHECKS
    )
    assert [phase.phase for phase in exhaustive_result.phases] == [0, 1, 2, 3]


def test_no_file_keeps_s2t_only_check_ready_and_catalog_check_unavailable():
    contract = TestProtocolContract(
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count", "required_null_rate"],
            )
        ]
    )
    reader_results = [
        result
        for result in _reader_results()
        if result["kind"] in {"s2t_pair", "s2t_target"}
    ]

    protocol = compile_test_protocol(contract, reader_results=reader_results)
    checks = {check.kind: check for check in protocol.targets[0].checks}

    assert checks["row_count"].status == "ready"
    assert checks["required_null_rate"].status == "unavailable"
    assert checks["required_null_rate"].missing_dependencies == [
        "target_catalog"
    ]
    assert protocol.status == "partial_protocol"
    assert any(issue.code == "unavailable_check" for issue in protocol.issues)


def test_failed_static_preflight_marks_otherwise_ready_protocol_partial():
    contract = TestProtocolContract(
        file_id=7,
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count"],
            )
        ],
    )
    reader_results = _reader_results()
    catalog = next(
        result
        for result in reader_results
        if result["kind"] == "target_column_catalog"
    )
    catalog["payload"]["rows"] = catalog["payload"]["rows"][:1]

    protocol = compile_test_protocol(contract, reader_results=reader_results)

    assert protocol.targets[0].checks[0].status == "ready"
    assert protocol.status == "partial_protocol"
    assert protocol.phases[0].failed_count == 1


def test_explicit_key_has_priority_and_does_not_require_catalog_pk():
    contract = TestProtocolContract(
        explicit_key=["entity_id"],
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=[
                    "key_uniqueness",
                    "key_reconciliation",
                    "missing_rows",
                    "extra_rows",
                ],
            )
        ],
    )
    reader_results = [
        result
        for result in _reader_results()
        if result["kind"] in {"s2t_pair", "s2t_target"}
    ]

    protocol = compile_test_protocol(contract, reader_results=reader_results)
    target = protocol.targets[0]

    assert target.comparison_key == ["entity_id"]
    assert target.comparison_key_source == "explicit"
    assert {check.status for check in target.checks} == {"ready"}
    assert protocol.status == "ready"
    for check in target.checks:
        _parse_compiled_sql(check.sql_template)


def test_normalized_transformation_preserves_expressions_and_structure():
    normalized = normalize_transformation(
        "SELECT CAST(s.entity_id AS BIGINT) AS target_id, "
        "CASE WHEN d.value IS NULL THEN COALESCE(s.required_value, 'x') "
        "ELSE d.value END AS target_value "
        "FROM stage.source_entity AS s "
        "LEFT JOIN stage.dictionary AS d ON d.id = s.entity_id "
        "WHERE s.active = 1 GROUP BY s.entity_id, s.required_value, d.value"
    )

    assert normalized.parse_status == "ok"
    assert normalized.sources == ["stage.source_entity", "stage.dictionary"]
    assert "CAST" in normalized.projections["target_id"]
    assert "CASE" in normalized.projections["target_value"]
    assert normalized.joins[0].join_type == "LEFT JOIN"
    assert normalized.filters == ["s.active = 1"]
    assert normalized.grouping == ["s.entity_id", "s.required_value", "d.value"]


def test_expression_aliases_are_used_as_expected_target_projection():
    rule = (
        "SELECT CAST(s.entity_id AS BIGINT) AS target_id, "
        "COALESCE(s.required_value, 'missing') AS target_value "
        "FROM stage.source_entity AS s;"
    )
    contract = TestProtocolContract(
        explicit_key=["target_id"],
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["transformation_correctness", "field_mismatch"],
            )
        ],
    )
    protocol = compile_test_protocol(
        contract,
        reader_results=_full_reader_results(
            rule=rule,
            mappings=[
                ("entity_id", "target_id"),
                ("required_value", "target_value"),
            ],
        ),
    )

    target = protocol.targets[0]
    transformation = target.checks[0]
    assert target.normalized_transformation.parse_status == "ok"
    assert 'src."target_id" AS "target_id"' in transformation.sql_template
    assert 'src."target_value" AS "target_value"' in transformation.sql_template
    assert "src.\"entity_id\" AS \"target_id\"" not in transformation.sql_template
    assert all(check.status == "ready" for check in target.checks)
    for check in target.checks:
        _parse_compiled_sql(check.sql_template)


def test_exhaustive_protocol_compiles_all_thirteen_parseable_checks():
    contract = TestProtocolContract(
        file_id=7,
        mode="exhaustive",
        loads=[TestProtocolLoad(sources=["source_entity"], target="target_entity")],
    )

    protocol = compile_test_protocol(
        contract,
        reader_results=_full_reader_results(),
    )
    target = protocol.targets[0]

    assert protocol.status == "ready"
    assert len(target.checks) == 13
    assert {check.status for check in target.checks} == {"ready"}
    assert {check.phase for check in target.checks} == {1, 2, 3}
    assert len(target.preflight) == 7
    assert all(item.status == "pass" for item in target.preflight)
    for check in target.checks:
        _parse_compiled_sql(check.sql_template)


def test_compiled_queries_use_separate_source_and_target_scope_placeholders():
    contract = TestProtocolContract(
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count", "transformation_correctness"],
            )
        ]
    )
    protocol = compile_test_protocol(
        contract,
        reader_results=_reader_results(),
    )

    for check in protocol.targets[0].checks:
        assert SOURCE_SCOPE_PREDICATE in check.sql_template
        assert TARGET_SCOPE_PREDICATE in check.sql_template
        _parse_compiled_sql(check.sql_template)


def test_target_only_s2t_reader_result_is_enough_for_exact_load_compilation():
    contract = TestProtocolContract(
        loads=[
            TestProtocolLoad(
                sources=["source_entity"],
                target="target_entity",
                checks=["row_count"],
            )
        ]
    )
    reader_results = [
        result for result in _reader_results() if result["kind"] == "s2t_target"
    ]

    protocol = compile_test_protocol(contract, reader_results=reader_results)

    assert protocol.targets[0].checks[0].status == "ready"
    assert next(
        item
        for item in protocol.targets[0].preflight
        if item.kind == "requested_sources_exist"
    ).status == "pass"
