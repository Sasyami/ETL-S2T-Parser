"""Typed inputs and rendering for S2T-only validation analysis."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT  # noqa: F401

from .tools import (
    list_target_column_catalog,
    read_s2t_by_target_table,
    read_s2t_source_to_target,
    resolve_file,
)
from .tools.saved_results import (
    _tabular_payload,
    get_active_saved_result_store,
)


RequestedAnalysis = Literal[
    "row_loss_risk",
    "duplicate_risk",
    "unmapped_required_fields",
    "transformation_consistency",
]

REQUESTED_ANALYSES: tuple[RequestedAnalysis, ...] = (
    "row_loss_risk",
    "duplicate_risk",
    "unmapped_required_fields",
    "transformation_consistency",
)
LLM_ANALYSES: tuple[RequestedAnalysis, ...] = (
    "transformation_consistency",
)
MAX_ANALYSIS_OBJECTS = 8


class ValidationProtocolContract(BaseModel):
    """Strict S2T analysis request extracted from the original task."""

    model_config = ConfigDict(extra="forbid")

    file_id: Optional[int] = Field(default=None, gt=0)
    filename: Optional[str] = Field(default=None, min_length=1, max_length=500)
    source_tables: List[str] = Field(
        min_length=1,
        max_length=MAX_ANALYSIS_OBJECTS,
    )
    target_tables: List[str] = Field(
        min_length=1,
        max_length=MAX_ANALYSIS_OBJECTS,
    )
    requested_analyses: List[RequestedAnalysis] = Field(
        min_length=1,
        max_length=len(REQUESTED_ANALYSES),
    )

    @model_validator(mode="after")
    def keep_one_file_selector(self) -> "ValidationProtocolContract":
        if self.file_id is not None and str(self.filename or "").strip():
            raise ValueError("Укажи только file_id или filename.")
        return self


class S2TAnalysisItem(BaseModel):
    """One evidence-grounded conclusion produced by the S2T analyzer."""

    model_config = ConfigDict(extra="forbid")

    target_table: str = Field(min_length=1, max_length=300)
    kind: RequestedAnalysis
    conclusion: str = Field(min_length=1)
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class S2TAnalysisOutput(BaseModel):
    """Strict semantic output of the S2T analyzer."""

    model_config = ConfigDict(extra="forbid")

    analyses: List[S2TAnalysisItem] = Field(
        min_length=0,
        max_length=MAX_ANALYSIS_OBJECTS * len(REQUESTED_ANALYSES),
    )


class ValidationProtocolDataError(RuntimeError):
    """Raised when direct readers cannot supply an S2T analysis input."""


def _rows(payload: Any) -> List[Dict[str, Any]]:
    decoded = _tabular_payload(payload)
    if decoded is None:
        return []
    return [dict(row) for row in decoded.get("rows") or []]


def _truthy_catalog_flag(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return str(value or "").strip().casefold() in {
        "true",
        "истина",
        "да",
        "yes",
    }


def _qualified_table_name(table: exp.Table) -> str:
    return ".".join(
        part for part in (table.catalog, table.db, table.name) if part
    )


def _sql_text(node: Any) -> str:
    return node.sql(dialect=GREENPLUM_DIALECT) if node is not None else ""


def _is_sql_tautology(node: Any) -> bool:
    """Recognize constant predicates that cannot filter rows."""
    if node is None:
        return False
    if isinstance(node, exp.Paren):
        return _is_sql_tautology(node.this)
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.EQ):
        left, right = node.this, node.expression
        return (
            isinstance(left, (exp.Literal, exp.Boolean))
            and isinstance(right, (exp.Literal, exp.Boolean))
            and _sql_text(left) == _sql_text(right)
        )
    if isinstance(node, exp.And):
        return _is_sql_tautology(node.this) and _is_sql_tautology(
            node.expression
        )
    if isinstance(node, exp.Or):
        return _is_sql_tautology(node.this) or _is_sql_tautology(
            node.expression
        )
    return False


def _left_pipeline_selects(statement: exp.Expression) -> List[exp.Select]:
    """Return SELECT scopes feeding the preserved/root FROM side."""
    root = statement if isinstance(statement, exp.Select) else None
    if root is None:
        root = statement.find(exp.Select)
    if root is None:
        return []
    with_clause = root.args.get("with_")
    ctes = {
        cte.alias_or_name.casefold(): cte.this
        for cte in (with_clause.expressions if with_clause is not None else [])
        if cte.alias_or_name
    }
    selects: List[exp.Select] = []
    current: Any = root
    visited: set[int] = set()
    while isinstance(current, exp.Expression) and id(current) not in visited:
        visited.add(id(current))
        while isinstance(current, (exp.Subquery, exp.Paren)):
            current = current.this
        if not isinstance(current, exp.Select):
            break
        selects.append(current)
        from_clause = current.args.get("from_")
        relation = from_clause.this if from_clause is not None else None
        while isinstance(relation, (exp.Subquery, exp.Paren)):
            relation = relation.this
        if isinstance(relation, exp.Select):
            current = relation
            continue
        if isinstance(relation, exp.Table):
            cte_query = ctes.get(relation.name.casefold())
            if cte_query is not None:
                current = cte_query
                continue
        break
    return selects


def _join_facts(join: exp.Join) -> Dict[str, Any]:
    side = str(join.side or "").upper()
    kind = str(join.kind or "").upper()
    join_type = " ".join(part for part in (side, kind, "JOIN") if part)
    return {
        "join_type": join_type or "JOIN",
        "relation": _sql_text(join.this),
        "on": _sql_text(join.args.get("on")),
        "preserves_left_rows": side in {"LEFT", "FULL"},
        "on_predicate_filters_preserved_left_rows": False
        if side in {"LEFT", "FULL"}
        else None,
        "cardinality": "unknown",
    }


def _sql_facts(rule: str) -> Dict[str, Any]:
    """Extract clause boundaries that the LLM must not infer from prose."""
    parseable_rule = re.sub(
        r"(?<![\w$])\$\$([A-Za-z0-9_]+)(?=\.)",
        lambda match: f'"$${match.group(1)}"',
        rule,
    )
    try:
        statement = sqlglot.parse_one(parseable_rule, read=GREENPLUM_DIALECT)
    except (SqlglotError, ValueError) as exc:
        return {
            "parse_status": "error",
            "parse_error": type(exc).__name__,
            "parse_error_detail": str(exc)[:500],
        }

    joins = [_join_facts(join) for join in statement.find_all(exp.Join)]
    left_pipeline = _left_pipeline_selects(statement)
    effective_joins = [
        _join_facts(join)
        for select in left_pipeline
        for join in select.args.get("joins") or []
    ]

    where_predicates = [
        _sql_text(where.this) for where in statement.find_all(exp.Where)
    ]
    having_predicates = [
        _sql_text(having.this) for having in statement.find_all(exp.Having)
    ]
    effective_where_predicates = [
        _sql_text(where.this)
        for select in left_pipeline
        if (where := select.args.get("where")) is not None
        and not _is_sql_tautology(where.this)
    ]
    effective_having_predicates = [
        _sql_text(having.this)
        for select in left_pipeline
        if (having := select.args.get("having")) is not None
        and not _is_sql_tautology(having.this)
    ]
    groupings = [
        [_sql_text(item) for item in group.expressions]
        for group in statement.find_all(exp.Group)
    ]
    source_tables = list(
        dict.fromkeys(
            name
            for table in statement.find_all(exp.Table)
            if (name := _qualified_table_name(table))
        )
    )
    return {
        "parse_status": "ok",
        "quoted_dollar_schemas_for_parse": parseable_rule != rule,
        "source_tables": source_tables,
        "where_predicates": where_predicates,
        "having_predicates": having_predicates,
        "joins": joins,
        "effective_where_predicates": effective_where_predicates,
        "effective_having_predicates": effective_having_predicates,
        "effective_joins": effective_joins,
        "group_by": groupings,
        "has_distinct": any(
            select.args.get("distinct") is not None
            for select in statement.find_all(exp.Select)
        ),
    }


def _compact_s2t_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep every raw occurrence while storing each exact SQL rule once."""
    rule_ids: Dict[str, str] = {}
    rules: List[Dict[str, Any]] = []
    mappings: List[Dict[str, Any]] = []
    for row in rows:
        mapping = dict(row)
        rule = str(mapping.pop("transformation_rule", "") or "").strip()
        rule_id = None
        if rule:
            rule_id = rule_ids.get(rule)
            if rule_id is None:
                rule_id = f"rule_{len(rule_ids) + 1}"
                rule_ids[rule] = rule_id
                rules.append(
                    {
                        "rule_id": rule_id,
                        "sql": rule,
                        "sql_facts": _sql_facts(rule),
                    }
                )
        mapping["rule_id"] = rule_id
        mappings.append(mapping)
    return mappings, rules


def _coverage_facts(
    contract: ValidationProtocolContract,
    reader_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    s2t_rows_by_target: Dict[str, List[Dict[str, Any]]] = {}
    catalog_rows_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for result in reader_results:
        kind = str(result["kind"])
        args = dict(result["args"])
        rows = _rows(result["payload"])
        if kind == "s2t_target":
            s2t_rows_by_target[str(args.get("target_table") or "")] = rows
        elif kind == "target_column_catalog":
            catalog_rows_by_target[str(args.get("table_name") or "")] = rows

    facts: List[Dict[str, Any]] = []
    for target_table in contract.target_tables:
        s2t_rows = s2t_rows_by_target.get(target_table, [])
        catalog_rows = catalog_rows_by_target.get(target_table, [])
        required = list(
            dict.fromkeys(
                str(row.get("column_name") or "").strip()
                for row in catalog_rows
                if _truthy_catalog_flag(row.get("not_null"))
                and str(row.get("column_name") or "").strip()
            )
        )
        mapped = list(
            dict.fromkeys(
                str(row.get("target_field") or "").strip()
                for row in s2t_rows
                if str(row.get("target_field") or "").strip()
            )
        )
        mapped_set = {column.casefold() for column in mapped}
        catalog_fields = list(
            dict.fromkeys(
                str(row.get("column_name") or "").strip()
                for row in catalog_rows
                if str(row.get("column_name") or "").strip()
            )
        )
        catalog_set = {column.casefold() for column in catalog_fields}
        unmapped = [
            column for column in required if column.casefold() not in mapped_set
        ]
        unmapped_catalog_fields = [
            column
            for column in catalog_fields
            if column.casefold() not in mapped_set
        ]
        mapped_fields_not_in_catalog = [
            column for column in mapped if column.casefold() not in catalog_set
        ]
        exact_rules = list(
            dict.fromkeys(
                str(row.get("transformation_rule") or "").strip()
                for row in s2t_rows
                if str(row.get("transformation_rule") or "").strip()
            )
        )
        physical_source_tables = list(
            dict.fromkeys(
                table
                for rule in exact_rules
                for table in _sql_facts(rule).get("source_tables", [])
            )
        )
        facts.append(
            {
                "target_table": target_table,
                "catalog_row_count": len(catalog_rows),
                "s2t_row_count": len(s2t_rows),
                "required_fields": required,
                "mapped_target_fields": mapped,
                "catalog_fields": catalog_fields,
                "unmapped_catalog_fields": unmapped_catalog_fields,
                "mapped_fields_not_in_catalog": mapped_fields_not_in_catalog,
                "unmapped_required_fields": unmapped,
                "catalog_fields_count": len(catalog_fields),
                "mapped_target_fields_count": len(mapped),
                "required_fields_count": len(required),
                "unmapped_required_fields_count": len(unmapped),
                "exact_rule_count": len(exact_rules),
                "physical_source_tables": physical_source_tables,
                "requested_source_tables": list(contract.source_tables),
                "s2t_source_tables": list(
                    dict.fromkeys(
                        str(row.get("source_table") or "").strip()
                        for row in s2t_rows
                        if str(row.get("source_table") or "").strip()
                    )
                ),
            }
        )
        fact = facts[-1]
        actual_source_names = {
            value.casefold() for value in fact["s2t_source_tables"]
        }
        requested_source_names = {
            value.casefold() for value in contract.source_tables
        }
        fact["missing_requested_source_tables"] = [
            value
            for value in contract.source_tables
            if value.casefold() not in actual_source_names
        ]
        fact["additional_s2t_source_tables"] = [
            value
            for value in fact["s2t_source_tables"]
            if value.casefold() not in requested_source_names
        ]
    return facts


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
    return tool.invoke(args, config=config) if config is not None else tool.invoke(args)


def read_validation_protocol_inputs(
    contract: ValidationProtocolContract,
    *,
    callbacks: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    """Read complete S2T and catalog evidence; never query logical ETL data."""
    results: List[Dict[str, Any]] = []
    resolved_file_id = contract.file_id
    filename = str(getattr(contract, "filename", None) or "").strip()
    if resolved_file_id is None and filename:
        resolution = _invoke(
            resolve_file,
            {"filename": filename},
            callbacks,
        )
        if not isinstance(resolution, dict) or resolution.get("error"):
            detail = (
                resolution.get("error")
                if isinstance(resolution, dict)
                else "неизвестный формат результата"
            )
            raise ValidationProtocolDataError(
                f"Файл {filename!r} не разрешён: {detail}."
            )
        try:
            resolved_file_id = int(resolution["file_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationProtocolDataError(
                f"Файл {filename!r} не вернул корректный file_id."
            ) from exc
        _persist_full_result(
            tool_name="resolve_file",
            call_id="resolve_protocol_file",
            payload=resolution,
        )
    protocol_loads = list(getattr(contract, "loads", []) or [])
    if protocol_loads:
        load_specs = [
            (
                index,
                str(load.target),
                [str(source) for source in load.sources],
            )
            for index, load in enumerate(protocol_loads, start=1)
        ]
    else:
        load_specs = [
            (index, target_table, [])
            for index, target_table in enumerate(
                contract.target_tables,
                start=1,
            )
        ]

    requested_analyses = list(
        getattr(contract, "requested_analyses", []) or []
    )
    needs_target_catalog = bool(protocol_loads) or any(
        kind in {"unmapped_required_fields", "transformation_consistency"}
        for kind in requested_analyses
    )

    for index, target_table, exact_sources in load_specs:
        exact_reads = tuple(
            (
                "read_s2t_source_to_target",
                f"test_protocol_pair_{index}_{source_index}",
                "s2t_pair",
                {
                    "source_table": source_table,
                    "target_table": target_table,
                },
                read_s2t_source_to_target,
            )
            for source_index, source_table in enumerate(exact_sources, start=1)
        )
        reads = [
            *exact_reads,
            (
                "read_s2t_by_target_table",
                f"s2t_analysis_target_{index}",
                "s2t_target",
                {"target_table": target_table},
                read_s2t_by_target_table,
            ),
        ]
        if needs_target_catalog and resolved_file_id is not None:
            reads.append(
                (
                    "list_target_column_catalog",
                    f"s2t_analysis_catalog_{index}",
                    "target_column_catalog",
                    {
                        "file_id": resolved_file_id,
                        "table_name": target_table,
                    },
                    list_target_column_catalog,
                )
            )
        for tool_name, call_id, kind, args, tool in reads:
            payload = _invoke(tool, args, callbacks)
            results.append(
                {
                    "kind": kind,
                    "load_index": index if protocol_loads else None,
                    "args": args,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "payload": payload,
                }
            )
    for result in results:
        tool_name = result["tool_name"]
        call_id = result["call_id"]
        payload = result["payload"]
        if not isinstance(payload, dict):
            raise ValidationProtocolDataError(
                f"{tool_name} вернул результат неизвестного формата."
            )
        if payload.get("error"):
            raise ValidationProtocolDataError(
                f"{tool_name}: {payload.get('error')}"
            )
        _persist_full_result(
            tool_name=tool_name,
            call_id=call_id,
            payload=payload,
        )
    return results


def build_s2t_analysis_payload(
    contract: ValidationProtocolContract,
    *,
    reader_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a lossless compact view of typed reader results for the LLM."""
    compact_results: List[Dict[str, Any]] = []
    for result in reader_results:
        kind = str(result["kind"])
        rows = _rows(result["payload"])
        compact: Dict[str, Any] = {
            "kind": kind,
            "args": dict(result["args"]),
        }
        if kind == "s2t_target":
            mappings, rules = _compact_s2t_rows(rows)
            compact.update(
                {
                    "mappings": mappings,
                    "rules": rules,
                    "raw_row_count": len(rows),
                }
            )
        else:
            compact["rows"] = rows
        compact_results.append(compact)
    return {
        "contract": contract.model_dump(),
        "reader_results": compact_results,
        "derived_facts": {
            "mapping_coverage": _coverage_facts(contract, reader_results),
        },
        "llm_requested_analyses": [
            kind
            for kind in contract.requested_analyses
            if kind in LLM_ANALYSES
        ],
    }


def build_s2t_analysis_display_payloads(
    reader_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build user-visible copies of the exact inputs used by S2T analysis."""
    displays: List[Dict[str, str]] = []
    for result in reader_results:
        payload = result.get("payload")
        if not isinstance(payload, Mapping):
            continue
        public_payload = dict(payload)
        public_payload.pop("saved_result", None)
        displays.append(
            {
                "name": str(result.get("tool_name") or "s2t_analysis_input"),
                "content": json.dumps(
                    {
                        "args": dict(result.get("args") or {}),
                        "result": public_payload,
                    },
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                ),
            }
        )
    return displays


def build_deterministic_analysis_items(
    contract: ValidationProtocolContract,
    *,
    reader_results: Sequence[Dict[str, Any]],
) -> List[S2TAnalysisItem]:
    """Build exact SQL-structural and set-based conclusions without an LLM."""
    s2t_rows_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for result in reader_results:
        if str(result["kind"]) != "s2t_target":
            continue
        target_table = str(result["args"].get("target_table") or "")
        s2t_rows_by_target[target_table] = _rows(result["payload"])

    coverage_by_target = {
        str(fact["target_table"]): fact
        for fact in _coverage_facts(contract, reader_results)
    }
    items: List[S2TAnalysisItem] = []
    for target_table in contract.target_tables:
        rows = s2t_rows_by_target.get(target_table, [])
        exact_rules = list(
            dict.fromkeys(
                str(row.get("transformation_rule") or "").strip()
                for row in rows
                if str(row.get("transformation_rule") or "").strip()
            )
        )
        rule_facts = [_sql_facts(rule) for rule in exact_rules]
        parsed = [fact for fact in rule_facts if fact["parse_status"] == "ok"]
        parse_failures = len(rule_facts) - len(parsed)

        if "row_loss_risk" in contract.requested_analyses:
            where_predicates = [
                predicate
                for fact in parsed
                for predicate in fact["effective_where_predicates"]
            ]
            having_predicates = [
                predicate
                for fact in parsed
                for predicate in fact["effective_having_predicates"]
            ]
            filtering_joins = [
                join
                for fact in parsed
                for join in fact["effective_joins"]
                if not join["preserves_left_rows"]
            ]
            row_loss_evidence = [
                *(f"WHERE: {value}" for value in where_predicates),
                *(f"HAVING: {value}" for value in having_predicates),
                *(
                    f"{join['join_type']} {join['relation']} не сохраняет "
                    "все левые строки"
                    for join in filtering_joins
                ),
            ]
            limitations: List[str] = []
            if parse_failures:
                limitations.append(
                    f"Не разобрано точных правил: {parse_failures}."
                )
            if row_loss_evidence:
                conclusion = (
                    "Найден потенциальный структурный риск потери строк: "
                    "правило содержит фильтрующие конструкции."
                )
                limitations.append(
                    "Фактическая потеря строк без физических данных неизвестна."
                )
            elif parse_failures or not exact_rules:
                conclusion = (
                    "Структурный риск потери строк определить нельзя: "
                    "нет полностью разобранного правила."
                )
            else:
                conclusion = (
                    "Структурный риск удаления исходных строк не найден: "
                    "нет WHERE/HAVING и непредохраняющих JOIN."
                )
                row_loss_evidence = [
                    f"{join['join_type']} {join['relation']} сохраняет левые строки"
                    for fact in parsed
                    for join in fact["effective_joins"]
                    if join["preserves_left_rows"]
                ] or ["Фильтрующие SQL-конструкции не найдены."]
            items.append(
                S2TAnalysisItem(
                    target_table=target_table,
                    kind="row_loss_risk",
                    conclusion=conclusion,
                    evidence=row_loss_evidence,
                    limitations=limitations,
                )
            )

        if "duplicate_risk" in contract.requested_analyses:
            joins = [join for fact in parsed for join in fact["joins"]]
            limitations = []
            if parse_failures:
                limitations.append(
                    f"Не разобрано точных правил: {parse_failures}."
                )
            if joins:
                conclusion = (
                    "Найден потенциальный риск размножения строк: "
                    "кардинальность одного или нескольких JOIN неизвестна."
                )
                duplicate_evidence = [
                    f"{join['join_type']} {join['relation']}; "
                    f"cardinality={join['cardinality']}"
                    for join in joins
                ]
                limitations.append(
                    "Фактические дубли без физических данных неизвестны."
                )
            elif parse_failures or not exact_rules:
                conclusion = (
                    "Структурный риск размножения строк определить нельзя: "
                    "нет полностью разобранного правила."
                )
                duplicate_evidence = []
            else:
                conclusion = (
                    "JOIN-индуцированный риск размножения строк не найден."
                )
                duplicate_evidence = ["JOIN в точных правилах не найдены."]
                limitations.append(
                    "Уникальность исходных данных без их чтения неизвестна."
                )
            items.append(
                S2TAnalysisItem(
                    target_table=target_table,
                    kind="duplicate_risk",
                    conclusion=conclusion,
                    evidence=duplicate_evidence,
                    limitations=limitations,
                )
            )

        if "unmapped_required_fields" in contract.requested_analyses:
            fact = coverage_by_target[target_table]
            if int(fact["catalog_row_count"]) == 0:
                conclusion = (
                    "Невозможно определить обязательные поля без маппинга: "
                    "target-каталог пуст."
                )
                limitations = ["В reader-result нет строк target-каталога."]
            elif fact["unmapped_required_fields"]:
                conclusion = (
                    "Обязательные target-поля без S2T-маппинга: "
                    + ", ".join(fact["unmapped_required_fields"])
                    + "."
                )
                limitations = []
            else:
                conclusion = (
                    "Обязательных target-полей без S2T-маппинга нет."
                )
                limitations = []
            items.append(
                S2TAnalysisItem(
                    target_table=target_table,
                    kind="unmapped_required_fields",
                    conclusion=conclusion,
                    evidence=[
                        "required_fields=" + repr(fact["required_fields"]),
                        "mapped_target_fields="
                        + repr(fact["mapped_target_fields"]),
                        "unmapped_required_fields="
                        + repr(fact["unmapped_required_fields"]),
                    ],
                    limitations=limitations,
                )
            )
    return items


def validate_s2t_analysis_output(
    contract: ValidationProtocolContract,
    output: S2TAnalysisOutput,
) -> None:
    """Require exactly one conclusion for every requested analysis."""
    llm_kinds = {
        kind for kind in contract.requested_analyses if kind in LLM_ANALYSES
    }
    expected = {
        (target_table, kind)
        for target_table in contract.target_tables
        for kind in llm_kinds
    }
    actual = [(item.target_table, item.kind) for item in output.analyses]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            "S2T analyzer должен вернуть по одному выводу для каждого "
            "семантического requested_analyses без дополнительных видов."
        )


def merge_s2t_analysis_output(
    semantic_output: S2TAnalysisOutput,
    deterministic_items: Sequence[S2TAnalysisItem],
) -> S2TAnalysisOutput:
    """Combine LLM conclusions with exact code-derived conclusions."""
    return S2TAnalysisOutput(
        analyses=[*semantic_output.analyses, *deterministic_items]
    )


def render_s2t_analysis_answer(
    contract: ValidationProtocolContract,
    output: S2TAnalysisOutput,
) -> str:
    """Render analyzer conclusions without inventing physical ETL results."""
    by_scope = {
        (item.target_table, item.kind): item for item in output.analyses
    }
    titles = {
        "row_loss_risk": "Риск потери строк",
        "duplicate_risk": "Риск дубликатов",
        "unmapped_required_fields": "Обязательные поля без S2T-маппинга",
        "transformation_consistency": "Согласованность трансформации",
    }
    target_sections: List[str] = []
    for target_table in contract.target_tables:
        sections: List[str] = []
        for kind in contract.requested_analyses:
            item = by_scope[(target_table, kind)]
            section = f"{titles[kind]}: {item.conclusion}"
            if item.evidence:
                section += "\nОснования: " + "; ".join(item.evidence)
            if item.limitations:
                section += "\nОграничения: " + "; ".join(item.limitations)
            sections.append(section)
        target_sections.append(
            f"Target `{target_table}`\n\n" + "\n\n".join(sections)
        )
    sources = ", ".join(f"`{value}`" for value in contract.source_tables)
    targets = ", ".join(f"`{value}`" for value in contract.target_tables)
    header = (
        f"S2T-анализ sources [{sources}] → targets [{targets}]. "
        "Физические данные логических ETL-таблиц не запрашивались."
    )
    return header + "\n\n" + "\n\n".join(target_sections)


__all__ = [
    "REQUESTED_ANALYSES",
    "MAX_ANALYSIS_OBJECTS",
    "LLM_ANALYSES",
    "RequestedAnalysis",
    "S2TAnalysisItem",
    "S2TAnalysisOutput",
    "ValidationProtocolContract",
    "ValidationProtocolDataError",
    "build_s2t_analysis_display_payloads",
    "build_s2t_analysis_payload",
    "build_deterministic_analysis_items",
    "merge_s2t_analysis_output",
    "read_validation_protocol_inputs",
    "render_s2t_analysis_answer",
    "validate_s2t_analysis_output",
]
