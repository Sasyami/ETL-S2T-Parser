import json

from agents.test_protocol import (
    LOAD_SCOPE_PREDICATE,
    PROTOCOL_CHECKS,
    TestProtocolContract,
    TestProtocolLoad,
    build_test_protocol_display_payloads,
    compile_test_protocol,
    render_test_protocol_answer,
)
from agents.tools.common import pack_tabular_rows


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
            "column_name": "ENTITY_ID",
            "primary_key": True,
            "not_null": True,
        },
        {
            "column_name": "REQUIRED_VALUE",
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
                checks=list(PROTOCOL_CHECKS),
            )
        ],
    )

    protocol = compile_test_protocol(
        contract,
        reader_results=_reader_results(),
    )

    assert len(protocol.targets) == 1
    checks = {check.kind: check for check in protocol.targets[0].checks}
    assert set(checks) == set(PROTOCOL_CHECKS)
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
                checks=list(PROTOCOL_CHECKS),
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
    assert check.sql_template.startswith("-- SQL-шаблон не сформирован:")
    assert any(
        "Внешний SELECT не подтверждает выходные source_field" in limitation
        for limitation in protocol.targets[0].limitations
    )
