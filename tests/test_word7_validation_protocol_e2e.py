"""Offline end-to-end regressions for the critical Word requirement #7."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import sqlglot

import storage.database as db_storage
from agents.test_protocol import (
    RawTestProtocolContract,
    RawTestProtocolLoad,
    SOURCE_SCOPE_PREDICATE,
    TARGET_SCOPE_PREDICATE,
    compile_test_protocol,
)
from agents.test_protocol_resolution import (
    resolve_test_protocol_contract,
    validate_raw_contract_origin,
)
from agents.validation_protocol import read_test_protocol_inputs
from services.sql_dialects import GREENPLUM_DIALECT
from storage.database import get_db_connection, init_db


WORD_7_PROMPT = (
    "Для файла 's2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx' по сохранённой "
    "S2T-спецификации b3050000420005_paymentdetails → t_optn составь "
    "тест-протокол для проверки ETL-загрузки во внешней СУБД. Включи проверки "
    "количества строк, уникальности ключа, null-rate обязательных полей и "
    "корректности трансформаций. Для каждой проверки дай цель, SQL-шаблон и "
    "критерий прохождения. Используй подтверждённые таблицы, колонки и правила; "
    "фактические метрики не вычисляй."
)

WORD_7_TRANSFORMATION = """select
 b.*
 , technicalservice.product_entityid_uid
from
 $$305stg.b3050000420005_paymentdetails as b

 left outer join $$305stg.s305_0015_technicalservice as technicalservice
 on true
  and b.technicalservice_id = technicalservice.object_id
  and upper(technicalservice.ctl_action) <> 'D'

where 1 = 1"""

WORD_7_MAPPINGS = (
    ("b3050000420005_paymentdetails", "object_id_uid", "optn_id"),
    ("b3050000420005_paymentdetails", "valuedate", "start_dt"),
    (
        "b3050000420015_technicalservice",
        "product_entityid_uid",
        "agr_dep_id",
    ),
    (
        "b3050000420005_paymentdetails",
        "technicalservice_id_uid",
        "bus_srv_id",
    ),
    (
        "b3050000420005_paymentdetails",
        "currencycode_uid",
        "crncy_id",
    ),
    ("b3050000420005_paymentdetails", "object_id", "host_optn_id"),
    ("b3050000420005_paymentdetails", "amount", "optn_amt"),
    ("b3050000420005_paymentdetails", "amountrub", "optn_amt_rub"),
    (
        "b3050000420005_paymentdetails",
        "operationtypecode_uid",
        "optn_type_id",
    ),
    ("b3050000420005_paymentdetails", "registerid_uid", "registry_id"),
)

WORD_7_TARGET_COLUMNS = (
    ("agr_dep_id", "uuid", False, False),
    ("bus_srv_id", "uuid", False, False),
    ("crncy_id", "uuid", False, False),
    ("host_optn_id", "text", False, False),
    ("optn_amt", "numeric", False, False),
    ("optn_amt_rub", "numeric", False, False),
    ("optn_id", "uuid", True, True),
    ("optn_type_id", "uuid", False, False),
    ("registry_id", "uuid", False, False),
    ("start_dt", "date", False, False),
)

REQUESTED_CHECKS = (
    "row_count",
    "key_uniqueness",
    "required_null_rate",
    "transformation_correctness",
)


@dataclass(frozen=True)
class ProtocolScenario:
    name: str
    prompt: str
    file_id: int
    filename: str
    source_table: str
    target_table: str
    transformation: str
    mappings: tuple[tuple[str, str, str], ...]
    target_columns: tuple[tuple[str, str, bool, bool], ...]
    expected_pair_rows: int


SCENARIOS = (
    ProtocolScenario(
        name="historical_word_7",
        prompt=WORD_7_PROMPT,
        file_id=3,
        filename="s2t_sbrf_pprb_305000042_dul_b_t_v049.xlsx",
        source_table="b3050000420005_paymentdetails",
        target_table="t_optn",
        transformation=WORD_7_TRANSFORMATION,
        mappings=WORD_7_MAPPINGS,
        target_columns=WORD_7_TARGET_COLUMNS,
        expected_pair_rows=9,
    ),
    ProtocolScenario(
        name="renamed_holdout",
        prompt=(
            "Не выполняй запросы. Возьми загрузку landing.order_event_delta "
            "→ dwh.fact_order_event из файла "
            "'northwind_orders_acceptance_2026.xlsx' и подготовь четыре "
            "приёмочных SQL-контроля для Greenplum: сверку объёма, отсутствие "
            "повторов ключа, отсутствие NULL там, где они запрещены, и "
            "совпадение результата преобразования. Для каждого контроля "
            "опиши назначение, шаблон запроса и однозначный критерий успеха."
        ),
        file_id=17,
        filename="northwind_orders_acceptance_2026.xlsx",
        source_table="landing.order_event_delta",
        target_table="dwh.fact_order_event",
        transformation=(
            "SELECT src.event_uid, src.recorded_at, src.amount_src "
            "FROM landing.order_event_delta AS src "
            "WHERE src.is_current = TRUE"
        ),
        mappings=(
            ("landing.order_event_delta", "event_uid", "event_id"),
            ("landing.order_event_delta", "recorded_at", "loaded_at"),
            ("landing.order_event_delta", "amount_src", "event_amount"),
        ),
        target_columns=(
            ("event_id", "uuid", True, True),
            ("loaded_at", "timestamp", False, True),
            ("event_amount", "numeric", False, False),
        ),
        expected_pair_rows=3,
    ),
)


def _install_fixture_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    scenario: ProtocolScenario,
) -> None:
    database_path = tmp_path / f"{scenario.name}.db"
    monkeypatch.setattr(db_storage, "DB_PATH", str(database_path))
    init_db()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO files (file_id, filename, upload_time, description)
            VALUES (?, ?, '2026-09-15', 'Offline Word #7 regression fixture')
            """,
            (scenario.file_id, scenario.filename),
        )
        conn.executemany(
            """
            INSERT INTO s2t_transformations
            (file_id, sheet_name, row_num, source_table, source_field,
             target_table, target_field, transformation_rule,
             source_layer, target_layer)
            VALUES (?, 's2t', ?, ?, ?, ?, ?, ?, 'A', 'B')
            """,
            [
                (
                    scenario.file_id,
                    row_number,
                    source_table,
                    source_field,
                    scenario.target_table,
                    target_field,
                    scenario.transformation,
                )
                for row_number, (
                    source_table,
                    source_field,
                    target_field,
                ) in enumerate(scenario.mappings, start=1)
            ],
        )
        conn.executemany(
            """
            INSERT INTO target_columns
            (file_id, sheet_name, row_num, table_name, column_name,
             data_type, primary_key, not_null, description)
            VALUES (?, 'target_columns', ?, ?, ?, ?, ?, ?, '')
            """,
            [
                (
                    scenario.file_id,
                    row_number,
                    scenario.target_table,
                    column_name,
                    data_type,
                    int(primary_key),
                    int(not_null),
                )
                for row_number, (
                    column_name,
                    data_type,
                    primary_key,
                    not_null,
                ) in enumerate(scenario.target_columns, start=1)
            ],
        )


def _raw_model_contract(scenario: ProtocolScenario) -> RawTestProtocolContract:
    """Represent the valid model-owned extraction consumed downstream."""

    return RawTestProtocolContract(
        file_scope_kind="file_mention",
        file_mention=scenario.filename,
        loads=[
            RawTestProtocolLoad(
                source_mentions=[scenario.source_table],
                target_mention=scenario.target_table,
                requested_checks=list(REQUESTED_CHECKS),
            )
        ],
        mode="explicit",
    )


def _assert_greenplum_template_parses(sql_template: str) -> None:
    executable_shape = (
        sql_template.replace(SOURCE_SCOPE_PREDICATE, "TRUE")
        .replace(TARGET_SCOPE_PREDICATE, "TRUE")
    )
    assert sqlglot.parse_one(executable_shape, read=GREENPLUM_DIALECT) is not None


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.name)
def test_word_7_raw_contract_resolves_reads_and_compiles_four_ready_checks(
    monkeypatch,
    tmp_path,
    scenario: ProtocolScenario,
):
    _install_fixture_database(monkeypatch, tmp_path, scenario)
    raw_contract = _raw_model_contract(scenario)

    validate_raw_contract_origin(raw_contract, scenario.prompt)
    resolution = resolve_test_protocol_contract(raw_contract)

    assert resolution.status == "resolved"
    assert resolution.exact_bypass_count == 3
    assert resolution.contract is not None
    contract = resolution.contract
    assert contract.file_id == scenario.file_id
    assert contract.filename is None
    assert contract.loads[0].sources == [scenario.source_table]
    assert contract.loads[0].target == scenario.target_table
    assert [item.method for item in resolution.resolutions] == [
        "exact",
        "exact",
        "exact",
    ]
    assert [item.entity_type for item in contract.resolution_metadata] == [
        "file",
        "source_table",
        "target_table",
    ]

    reader_results = read_test_protocol_inputs(contract)

    assert [item["kind"] for item in reader_results] == [
        "s2t_pair",
        "s2t_target",
        "target_column_catalog",
    ]
    assert [item["args"] for item in reader_results] == [
        {
            "source_table": scenario.source_table,
            "target_table": scenario.target_table,
        },
        {"target_table": scenario.target_table},
        {"file_id": scenario.file_id, "table_name": scenario.target_table},
    ]
    assert reader_results[0]["payload"]["returned_rows"] == (
        scenario.expected_pair_rows
    )
    assert reader_results[1]["payload"]["returned_rows"] == len(
        scenario.mappings
    )
    assert reader_results[2]["payload"]["returned_rows"] == len(
        scenario.target_columns
    )

    protocol = compile_test_protocol(contract, reader_results=reader_results)

    assert protocol.status == "ready"
    assert protocol.issues == []
    assert len(protocol.targets) == 1
    target = protocol.targets[0]
    assert target.status == "ready"
    assert target.source_tables == [scenario.source_table]
    assert target.target_table == scenario.target_table
    assert [check.kind for check in target.checks] == list(REQUESTED_CHECKS)
    assert all(check.status == "ready" for check in target.checks)
    assert all(check.missing_dependencies == [] for check in target.checks)
    assert all(check.goal.strip() for check in target.checks)
    assert all(check.sql_template.strip() for check in target.checks)
    assert all(check.pass_criterion.strip() for check in target.checks)
    assert all(item.status == "pass" for item in target.preflight)
    for check in target.checks:
        _assert_greenplum_template_parses(check.sql_template)
