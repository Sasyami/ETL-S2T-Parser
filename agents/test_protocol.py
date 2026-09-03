"""Typed compiler for external Greenplum S2T validation protocols."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Sequence

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

from services.sql_dialects import GREENPLUM_DIALECT  # noqa: F401

from .tools.saved_results import _tabular_payload


ProtocolCheck = Literal[
    "row_count",
    "key_uniqueness",
    "required_null_rate",
    "transformation_correctness",
]

PROTOCOL_CHECKS: tuple[ProtocolCheck, ...] = (
    "row_count",
    "key_uniqueness",
    "required_null_rate",
    "transformation_correctness",
)
MAX_PROTOCOL_OBJECTS = 8
LOAD_SCOPE_PREDICATE = "{{LOAD_SCOPE_PREDICATE}}"
_SIMPLE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class TestProtocolLoad(BaseModel):
    """One independently scoped source-set to target validation load."""

    model_config = ConfigDict(extra="forbid")

    sources: List[str] = Field(
        min_length=1,
        max_length=MAX_PROTOCOL_OBJECTS,
    )
    target: str = Field(min_length=1, max_length=300)
    checks: List[ProtocolCheck] = Field(
        min_length=1,
        max_length=len(PROTOCOL_CHECKS),
    )


class TestProtocolContract(BaseModel):
    """Exact independent S2T loads requested for external validation."""

    model_config = ConfigDict(extra="forbid")

    file_id: Optional[int] = Field(default=None, gt=0)
    filename: Optional[str] = Field(default=None, min_length=1, max_length=500)
    loads: List[TestProtocolLoad] = Field(
        min_length=1,
        max_length=MAX_PROTOCOL_OBJECTS,
    )

    @model_validator(mode="after")
    def require_file_selector(self) -> "TestProtocolContract":
        has_filename = bool(str(self.filename or "").strip())
        if self.file_id is None and not has_filename:
            raise ValueError("Нужен file_id или точное имя файла.")
        if self.file_id is not None and has_filename:
            raise ValueError("Укажи только file_id или filename.")
        return self

    @property
    def source_tables(self) -> List[str]:
        return list(
            dict.fromkeys(source for load in self.loads for source in load.sources)
        )

    @property
    def target_tables(self) -> List[str]:
        return list(dict.fromkeys(load.target for load in self.loads))


class CompiledProtocolCheck(BaseModel):
    """One non-executed Greenplum validation check."""

    model_config = ConfigDict(extra="forbid")

    kind: ProtocolCheck
    goal: str
    sql_template: str
    pass_criterion: str
    limitations: List[str] = Field(default_factory=list)


class CompiledTargetProtocol(BaseModel):
    """Validation protocol for one exact target table."""

    model_config = ConfigDict(extra="forbid")

    source_tables: List[str]
    target_table: str
    checks: List[CompiledProtocolCheck]
    evidence: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    s2t_evidence_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
    )
    catalog_evidence_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
    )


class CompiledTestProtocol(BaseModel):
    """Complete deterministic protocol for all requested targets."""

    model_config = ConfigDict(extra="forbid")

    targets: List[CompiledTargetProtocol]


def _rows(payload: Any) -> List[Dict[str, Any]]:
    decoded = _tabular_payload(payload)
    if decoded is None:
        return []
    return [dict(row) for row in decoded.get("rows") or []]


def _truthy(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return str(value or "").strip().casefold() in {
        "true",
        "истина",
        "да",
        "yes",
    }


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_qualified_name(value: str) -> str:
    parts = [part.strip() for part in str(value).split(".") if part.strip()]
    if not parts:
        raise ValueError("Пустое имя таблицы в typed contract.")
    if any(not _SIMPLE_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError(f"Небезопасное имя таблицы: {value}")
    return ".".join(_quote_identifier(part) for part in parts)


def _parse_full_query(rule: str) -> tuple[exp.Query | None, str | None]:
    """Accept one complete Greenplum query, never an expression or fragment."""
    parseable_rule = re.sub(
        r"(?<![\w$])\$\$([A-Za-z0-9_]+)(?=\.)",
        lambda match: f'"$${match.group(1)}"',
        str(rule or "").strip(),
    )
    try:
        statements = sqlglot.parse(parseable_rule, read=GREENPLUM_DIALECT)
    except (SqlglotError, ValueError) as exc:
        return None, f"SQLGlot не разобрал transformation_rule: {type(exc).__name__}"
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None, "transformation_rule не является одним полным SELECT/WITH query"
    return statements[0], None


def _validate_query_outputs(
    query: exp.Query,
    pairs: Sequence[tuple[str, str]],
) -> str | None:
    """Require each mapped source name in the outer query projection."""
    select = query if isinstance(query, exp.Select) else query.find(exp.Select)
    if select is None:
        return "SQLGlot не нашёл внешний SELECT в transformation_rule"
    has_wildcard = False
    output_names: set[str] = set()
    for projection in select.expressions:
        value = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(value, exp.Star) or (
            isinstance(value, exp.Column) and value.is_star
        ):
            has_wildcard = True
            continue
        name = str(projection.alias_or_name or "").strip()
        if name:
            output_names.add(name.casefold())
    if has_wildcard:
        return None
    missing = [source for source, _ in pairs if source.casefold() not in output_names]
    if not missing:
        return None
    return (
        "Внешний SELECT не подтверждает выходные source_field: "
        + ", ".join(dict.fromkeys(missing))
    )


def _load_results(
    load_index: int,
    target_table: str,
    reader_results: Sequence[Dict[str, Any]],
) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pair_rows: Dict[str, List[Dict[str, Any]]] = {}
    target_rows: List[Dict[str, Any]] = []
    catalog_rows: List[Dict[str, Any]] = []
    for result in reader_results:
        if result.get("load_index") != load_index:
            continue
        kind = str(result.get("kind") or "")
        args = dict(result.get("args") or {})
        if kind == "s2t_pair" and args.get("target_table") == target_table:
            source_table = str(args.get("source_table") or "").casefold()
            pair_rows[source_table] = _rows(result.get("payload"))
        elif kind == "s2t_target" and args.get("target_table") == target_table:
            target_rows = _rows(result.get("payload"))
        elif (
            kind == "target_column_catalog"
            and args.get("table_name") == target_table
        ):
            catalog_rows = _rows(result.get("payload"))
    return pair_rows, target_rows, catalog_rows


def _row_rule(row: Dict[str, Any]) -> str:
    return str(row.get("transformation_rule") or "").strip()


def _select_load_rule(
    sources: Sequence[str],
    pair_rows: Dict[str, List[Dict[str, Any]]],
) -> tuple[str, List[str], str | None]:
    """Select one full query shared by every exact requested source pair."""
    missing_sources: List[str] = []
    sources_without_full_query: List[str] = []
    full_rule_sets: List[set[str]] = []
    ordered_rules: List[str] = []
    for source in sources:
        rows = pair_rows.get(source.casefold(), [])
        if not rows:
            missing_sources.append(source)
            continue
        source_rules: set[str] = set()
        for row in rows:
            rule = _row_rule(row)
            if not rule:
                continue
            parsed, _ = _parse_full_query(rule)
            if parsed is None:
                continue
            source_rules.add(rule)
            if rule not in ordered_rules:
                ordered_rules.append(rule)
        if not source_rules:
            sources_without_full_query.append(source)
            continue
        full_rule_sets.append(source_rules)
    if missing_sources:
        return "", missing_sources, None
    if sources_without_full_query:
        return (
            "",
            [],
            "Exact source→target строки не содержат полного SELECT/WITH для: "
            + ", ".join(sources_without_full_query)
            + ".",
        )
    common_rules = set.intersection(*full_rule_sets) if full_rule_sets else set()
    selected = [rule for rule in ordered_rules if rule in common_rules]
    if len(selected) != 1:
        return (
            "",
            [],
            "Для exact source→target пар требуется одно общее полное правило "
            f"SELECT; найдено: {len(selected)}.",
        )
    return selected[0], [], None


def _mapping_pairs(rows: Sequence[Dict[str, Any]]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        source = str(row.get("source_field") or "").strip()
        target = str(row.get("target_field") or "").strip()
        pair = (source, target)
        folded = (source.casefold(), target.casefold())
        if not source or not target or folded in seen:
            continue
        seen.add(folded)
        pairs.append(pair)
    return pairs


def _expected_ctes(rule: str, pairs: Sequence[tuple[str, str]]) -> str:
    projection = ",\n        ".join(
        f"src.{_quote_identifier(source)} AS {_quote_identifier(target)}"
        for source, target in pairs
    )
    return (
        "WITH expected_raw AS (\n"
        f"{rule.strip()}\n"
        "),\nexpected AS (\n"
        "    SELECT\n        "
        f"{projection}\n"
        "    FROM expected_raw AS src\n"
        ")"
    )


def _unavailable_check(
    kind: ProtocolCheck,
    goal: str,
    reason: str,
) -> CompiledProtocolCheck:
    return CompiledProtocolCheck(
        kind=kind,
        goal=goal,
        sql_template=f"-- SQL-шаблон не сформирован: {reason}",
        pass_criterion="Проверка требует устранить указанное ограничение.",
        limitations=[reason],
    )


def compile_test_protocol(
    contract: TestProtocolContract,
    *,
    reader_results: Sequence[Dict[str, Any]],
) -> CompiledTestProtocol:
    """Compile Greenplum SQL templates without querying logical ETL tables."""
    targets: List[CompiledTargetProtocol] = []
    for load_index, load in enumerate(contract.loads, start=1):
        target_table = load.target
        pair_rows, target_rows, catalog_rows = _load_results(
            load_index,
            target_table,
            reader_results,
        )
        rule, missing_sources, rule_selection_error = _select_load_rule(
            load.sources,
            pair_rows,
        )
        s2t_rows = [
            row for row in target_rows if rule and _row_rule(row) == rule
        ]
        if rule and not s2t_rows:
            s2t_rows = [
                row
                for source in load.sources
                for row in pair_rows.get(source.casefold(), [])
                if _row_rule(row) == rule
            ]
        pairs = _mapping_pairs(s2t_rows)
        mapped_target_names = {
            target.casefold(): target for _, target in pairs
        }
        target_sql = _quote_qualified_name(target_table)
        pk_fields = list(
            dict.fromkeys(
                mapped_target_names.get(
                    str(row.get("column_name") or "").strip().casefold(),
                    str(row.get("column_name") or "").strip(),
                )
                for row in catalog_rows
                if _truthy(row.get("primary_key"))
                and str(row.get("column_name") or "").strip()
            )
        )
        required_fields = list(
            dict.fromkeys(
                mapped_target_names.get(
                    str(row.get("column_name") or "").strip().casefold(),
                    str(row.get("column_name") or "").strip(),
                )
                for row in catalog_rows
                if _truthy(row.get("not_null"))
                and str(row.get("column_name") or "").strip()
            )
        )
        source_tables = list(
            dict.fromkeys(
                str(row.get("source_table") or "").strip()
                for row in s2t_rows
                if str(row.get("source_table") or "").strip()
            )
        )
        limitations: List[str] = []
        if missing_sources:
            limitations.append(
                "В S2T target не подтверждены запрошенные sources: "
                + ", ".join(missing_sources)
                + "."
            )
        if rule_selection_error:
            limitations.append(rule_selection_error)
        invalid_pairs = [
            f"{source}→{target}"
            for source, target in pairs
            if not _SIMPLE_IDENTIFIER.fullmatch(source)
            or not _SIMPLE_IDENTIFIER.fullmatch(target)
        ]
        if invalid_pairs:
            limitations.append(
                "Нельзя безопасно спроецировать выражения как колонки: "
                + ", ".join(invalid_pairs)
                + "."
            )

        parsed_rule, rule_error = _parse_full_query(rule) if rule else (None, None)
        if rule_error:
            limitations.append(rule_error + ".")
        output_error = (
            _validate_query_outputs(parsed_rule, pairs)
            if parsed_rule is not None and pairs
            else None
        )
        if output_error:
            limitations.append(output_error + ".")
        projection_ready = bool(
            rule
            and parsed_rule is not None
            and pairs
            and not invalid_pairs
            and output_error is None
        )
        expected_ctes = _expected_ctes(rule, pairs) if projection_ready else ""
        checks: List[CompiledProtocolCheck] = []
        for kind in load.checks:
            if kind == "row_count":
                if not projection_ready:
                    check = _unavailable_check(
                        kind,
                        "Сравнить количество ожидаемых и загруженных строк.",
                        "нет однозначного полного SELECT и проекции S2T",
                    )
                else:
                    check = CompiledProtocolCheck(
                        kind=kind,
                        goal="Сравнить количество ожидаемых и загруженных строк.",
                        sql_template=(
                            f"{expected_ctes}\n"
                            "SELECT\n"
                            "    (SELECT COUNT(*) FROM expected) "
                            "AS expected_row_count,\n"
                            f"    (SELECT COUNT(*) FROM {target_sql} WHERE "
                            f"{LOAD_SCOPE_PREDICATE}) "
                            "AS actual_row_count;"
                        ),
                        pass_criterion=(
                            "expected_row_count = actual_row_count."
                        ),
                    )
            elif kind == "key_uniqueness":
                if not pk_fields:
                    check = _unavailable_check(
                        kind,
                        "Проверить уникальность подтверждённого target-ключа.",
                        "в target-каталоге не подтверждён primary key",
                    )
                else:
                    keys = ", ".join(
                        _quote_identifier(value) for value in pk_fields
                    )
                    check = CompiledProtocolCheck(
                        kind=kind,
                        goal="Проверить уникальность подтверждённого target-ключа.",
                        sql_template=(
                            f"SELECT {keys}, COUNT(*) AS duplicate_count\n"
                            f"FROM {target_sql}\n"
                            f"WHERE {LOAD_SCOPE_PREDICATE}\n"
                            f"GROUP BY {keys}\n"
                            "HAVING COUNT(*) > 1;"
                        ),
                        pass_criterion="Запрос не возвращает строк.",
                    )
            elif kind == "required_null_rate":
                if not required_fields:
                    check = _unavailable_check(
                        kind,
                        "Проверить NULL в обязательных target-полях.",
                        "в target-каталоге нет полей not_null=true",
                    )
                else:
                    null_counts = ",\n    ".join(
                        "COUNT(*) FILTER (WHERE "
                        f"{_quote_identifier(value)} IS NULL) AS "
                        f"{_quote_identifier(value + '_null_count')}"
                        for value in required_fields
                    )
                    check = CompiledProtocolCheck(
                        kind=kind,
                        goal="Проверить NULL в обязательных target-полях.",
                        sql_template=(
                            "SELECT\n    COUNT(*) AS total_rows,\n    "
                            f"{null_counts}\nFROM {target_sql}\n"
                            f"WHERE {LOAD_SCOPE_PREDICATE};"
                        ),
                        pass_criterion=(
                            "Каждый *_null_count равен 0; null-rate каждого "
                            "обязательного поля равен 0."
                        ),
                    )
            else:
                if not projection_ready:
                    check = _unavailable_check(
                        kind,
                        "Сравнить ожидаемую S2T-проекцию с target.",
                        "нет однозначного полного SELECT и проекции S2T",
                    )
                else:
                    target_fields = ", ".join(
                        _quote_identifier(target) for _, target in pairs
                    )
                    check = CompiledProtocolCheck(
                        kind=kind,
                        goal="Сравнить ожидаемую S2T-проекцию с target.",
                        sql_template=(
                            f"{expected_ctes},\n"
                            "actual AS (\n"
                            f"    SELECT {target_fields}\n"
                            f"    FROM {target_sql}\n"
                            f"    WHERE {LOAD_SCOPE_PREDICATE}\n"
                            "),\ndifferences AS (\n"
                            "    (SELECT * FROM expected EXCEPT ALL "
                            "SELECT * FROM actual)\n"
                            "    UNION ALL\n"
                            "    (SELECT * FROM actual EXCEPT ALL "
                            "SELECT * FROM expected)\n"
                            ")\n"
                            "SELECT COUNT(*) AS difference_count\n"
                            "FROM differences;"
                        ),
                        pass_criterion="difference_count = 0.",
                    )
            checks.append(check)

        targets.append(
            CompiledTargetProtocol(
                source_tables=list(load.sources),
                target_table=target_table,
                checks=checks,
                evidence=[
                    f"S2T sources: {source_tables!r}",
                    f"mapped target fields: {[target for _, target in pairs]!r}",
                    f"primary key fields: {pk_fields!r}",
                    f"required fields: {required_fields!r}",
                ],
                limitations=limitations,
                s2t_evidence_rows=s2t_rows,
                catalog_evidence_rows=catalog_rows,
            )
        )
    return CompiledTestProtocol(targets=targets)


def render_test_protocol_answer(
    contract: TestProtocolContract,
    protocol: CompiledTestProtocol,
) -> str:
    """Render exact compiler output without a final LLM rewrite."""
    sources = ", ".join(f"`{value}`" for value in contract.source_tables)
    targets = ", ".join(f"`{value}`" for value in contract.target_tables)
    sections = [
        "Тест-протокол проверки ETL-загрузки во внешней Greenplum СУБД "
        f"для sources [{sources}] → targets [{targets}]. SQL-шаблоны не "
        "исполнялись; фактические метрики не вычислялись. Во всех target-side "
        f"запросах замени `{LOAD_SCOPE_PREDICATE}` условием проверяемой "
        "загрузки либо `TRUE` для полного снимка."
    ]
    titles = {
        "row_count": "Проверка количества строк",
        "key_uniqueness": "Проверка уникальности ключа",
        "required_null_rate": "Проверка null-rate обязательных полей",
        "transformation_correctness": "Проверка корректности трансформаций",
    }
    for target in protocol.targets:
        load_sources = ", ".join(f"`{value}`" for value in target.source_tables)
        target_parts = [
            f"Load sources [{load_sources}] → Target `{target.target_table}`"
        ]
        for index, check in enumerate(target.checks, start=1):
            block = (
                f"{index}. {titles[check.kind]}\n"
                f"Цель: {check.goal}\n"
                "SQL-шаблон:\n```sql\n"
                f"{check.sql_template}\n```\n"
                f"Критерий прохождения: {check.pass_criterion}"
            )
            if check.limitations:
                block += "\nОграничения: " + "; ".join(check.limitations)
            target_parts.append(block)
        target_parts.append("Подтверждённые основания: " + "; ".join(target.evidence))
        if target.limitations:
            target_parts.append("Ограничения target: " + "; ".join(target.limitations))
        sections.append("\n\n".join(target_parts))
    return "\n\n".join(sections)


def build_test_protocol_display_payloads(
    protocol: CompiledTestProtocol,
) -> List[Dict[str, str]]:
    """Build user-visible copies of the exact evidence used by the compiler."""
    displays: List[Dict[str, str]] = []
    for target in protocol.targets:
        for name, rows in (
            ("read_s2t_source_to_target", target.s2t_evidence_rows),
            ("list_target_column_catalog", target.catalog_evidence_rows),
        ):
            if not rows:
                continue
            columns = list(dict.fromkeys(key for row in rows for key in row))
            payload = {
                "load": {
                    "sources": list(target.source_tables),
                    "target": target.target_table,
                },
                "columns": columns,
                "rows": [[row.get(column) for column in columns] for row in rows],
            }
            displays.append(
                {
                    "name": name,
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                        separators=(",", ":"),
                    ),
                }
            )
    return displays


__all__ = [
    "MAX_PROTOCOL_OBJECTS",
    "LOAD_SCOPE_PREDICATE",
    "PROTOCOL_CHECKS",
    "CompiledProtocolCheck",
    "CompiledTargetProtocol",
    "CompiledTestProtocol",
    "ProtocolCheck",
    "TestProtocolLoad",
    "TestProtocolContract",
    "compile_test_protocol",
    "build_test_protocol_display_payloads",
    "render_test_protocol_answer",
]
