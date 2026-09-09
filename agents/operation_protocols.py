"""Prompt-only protocol variants for typed SQL-risk operation skills.

The experiment changes only the stage context shown to the LLM.  Data
readers, acceptance guards, deterministic analysis and coordinator control
flow deliberately remain outside this module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Literal, Mapping, Tuple, cast

from .contracts import SqlRiskAspect

OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV = (
    "OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT"
)

ProtocolFamily = Literal[
    "minimal_artifact",
    "evidence_ledger",
    "epistemic_state_machine",
    "decision_table",
]
ProtocolStage = Literal[
    "plan",
    "planner",
    "observer",
    "upstream_decision",
    "upstream",
]

SQL_RISK_ASPECTS: Tuple[SqlRiskAspect, ...] = (
    "row_filtering",
    "cardinality",
    "constraint_rejection",
    "value_changes",
    "write_semantics",
)
SQL_RISK_PROTOCOL_FAMILIES: Tuple[ProtocolFamily, ...] = (
    "minimal_artifact",
    "evidence_ledger",
    "epistemic_state_machine",
    "decision_table",
)
SQL_RISK_PROTOCOL_STAGES: Tuple[ProtocolStage, ...] = (
    "plan",
    "planner",
    "observer",
    "upstream_decision",
    "upstream",
)


@dataclass(frozen=True)
class AspectSpec:
    """Immutable facts needed by every protocol-family renderer."""

    aspect: SqlRiskAspect
    primary_artifact: str
    exact_scope: str
    mandatory_support: Tuple[str, ...]
    conditional_support: Tuple[str, ...]
    mechanism: str
    condition: str
    evidence_boundary: str
    terminal_absence_rule: str


@dataclass(frozen=True)
class ProtocolVariant:
    """One canonical aspect/family experiment variant."""

    aspect: SqlRiskAspect
    family: ProtocolFamily

    @property
    def name(self) -> str:
        return f"{self.aspect}__{self.family}"


SQL_RISK_ASPECT_SPECS: Mapping[SqlRiskAspect, AspectSpec] = MappingProxyType(
    {
        "row_filtering": AspectSpec(
            aspect="row_filtering",
            primary_artifact=(
                "полный untruncated exact directed S2T mapping с фактическим "
                "SQL или правилом"
            ),
            exact_scope=(
                "точная source_table → target_table пара без field, "
                "transformation_id и file_id narrowing"
            ),
            mandatory_support=(),
            conditional_support=(),
            mechanism=(
                "WHERE/HAVING/QUALIFY, JOIN ON, set predicates и LIMIT"
            ),
            condition=(
                "predicate со значением FALSE/UNKNOWN удаляет строку или "
                "группу; INNER JOIN удаляет unmatched, а LEFT JOIN сам по "
                "себе не удаляет левую строку"
            ),
            evidence_boundary=(
                "вывод относится только к рассмотренному полному mapping, "
                "а не ко всему внешнему pipeline"
            ),
            terminal_absence_rule=(
                "если в полном mapping релевантный predicate не обнаружен, "
                "это terminal evidence «не обнаружено в рассмотренной "
                "трансформации», а не основание искать другие таблицы"
            ),
        ),
        "cardinality": AspectSpec(
            aspect="cardinality",
            primary_artifact=(
                "полный untruncated exact directed S2T mapping с фактическим "
                "SQL или правилом"
            ),
            exact_scope=(
                "точная source_table → target_table пара без field, "
                "transformation_id и file_id narrowing"
            ),
            mandatory_support=(),
            conditional_support=(
                "активируй только если original_task прямо запрашивает "
                "фактическую уникальность, ключи или точное число: для ключей "
                "прочитай полную role-preserving metadata названных таблиц; "
                "для фактической уникальности/count используй только "
                "доступное exact data evidence, а при его отсутствии верни "
                "«не оценено»; catalog PK и mapping не подменяют фактический "
                "count. Для обычного структурного вопроса о риске "
                "дубликатов этот support не активируется",
            ),
            mechanism=(
                "JOIN multiplicity, DISTINCT, GROUP BY, set operations и "
                "дедупликация"
            ),
            condition=(
                "без evidence уникальности полного набора join keys "
                "размножение остаётся условным; прямой field mapping не "
                "доказывает 1:1"
            ),
            evidence_boundary=(
                "структурный или условный вывод допустим по exact mapping; "
                "неизвестную уникальность нужно явно сохранить"
            ),
            terminal_absence_rule=(
                "если полный mapping не содержит механизма изменения "
                "кардинальности, это terminal evidence «не обнаружено в "
                "рассмотренной трансформации»"
            ),
        ),
        "constraint_rejection": AspectSpec(
            aspect="constraint_rejection",
            primary_artifact=(
                "полный untruncated exact directed S2T mapping с фактическим "
                "SQL или правилом"
            ),
            exact_scope=(
                "точная source_table.source_field → "
                "target_table.target_field пара: mapping без file narrowing, "
                "catalog в заданном file scope с сохранением обеих ролей"
            ),
            mandatory_support=(
                "metadata обеих endpoint-колонок с data_type, primary_key и "
                "not_null: при подтверждённом file_id — одно exact-pair "
                "чтение со всеми пятью идентификаторами; без подтверждённого "
                "file_id — один role-preserving full-table batch с "
                "file_scope=all и table_names только из точных имён source/"
                "target-таблиц без суффиксов .field; строки разных file_id "
                "сохраняй отдельными группами и не смешивай атрибуты версий",
            ),
            conditional_support=(),
            mechanism=(
                "target NOT NULL, несовместимый тип и иные подтверждённые "
                "target constraints"
            ),
            condition=(
                "nullable source создаёт rejection лишь при фактическом NULL; "
                "несовпадение типов — условный conversion/rejection до "
                "проверки преобразования и значений"
            ),
            evidence_boundary=(
                "target-only metadata не доказывает безопасность source, а "
                "пустой сохранённый catalog не доказывает отсутствие "
                "ограничений во внешней СУБД"
            ),
            terminal_absence_rule=(
                "точное пустое либо неполное по атрибутам catalog-чтение "
                "терминально даёт «не оценено», но никогда не превращается в "
                "вывод об отсутствии constraints"
            ),
        ),
        "value_changes": AspectSpec(
            aspect="value_changes",
            primary_artifact=(
                "полный untruncated exact directed S2T mapping с фактическими "
                "выражениями значений"
            ),
            exact_scope=(
                "названная source_table.source_field → "
                "target_table.target_field пара; mapping читается без field, "
                "transformation_id и file_id narrowing"
            ),
            mandatory_support=(),
            conditional_support=(),
            mechanism=(
                "выражение exact S2T-строки и внешняя SQL-проекция именно "
                "target field: CASE, COALESCE, CAST или арифметика"
            ),
            condition=(
                "выражение в другом output alias не доказывает изменение "
                "выбранного target field; прямая проекция не является "
                "механизмом изменения"
            ),
            evidence_boundary=(
                "соседние проекции и одноимённые поля вне exact field-пары "
                "не входят в вывод"
            ),
            terminal_absence_rule=(
                "прямая проекция exact target field без преобразования — "
                "terminal evidence «механизм изменения не обнаружен»"
            ),
        ),
        "write_semantics": AspectSpec(
            aspect="write_semantics",
            primary_artifact=(
                "один полный untruncated exact directed S2T mapping с "
                "фактическим SQL или правилом"
            ),
            exact_scope=(
                "точная source_table → target_table пара без field, "
                "transformation_id и file_id narrowing"
            ),
            mandatory_support=(),
            conditional_support=(),
            mechanism=(
                "явный INSERT/append, overwrite, MERGE/UPSERT или conflict "
                "handling statement"
            ),
            condition=(
                "PK/UNIQUE не доказывает write mode, идемпотентность, "
                "дедупликацию или обработку конфликта"
            ),
            evidence_boundary=(
                "режим записи устанавливается только по statement в полном "
                "mapping, а не по metadata или предполагаемому поведению"
            ),
            terminal_absence_rule=(
                "отсутствие write statement в полном exact mapping — "
                "terminal negative evidence со статусом «не оценено»; не "
                "reroute и не заменяй его target PK"
            ),
        ),
    }
)


_PROTOCOL_VARIANTS: Mapping[str, ProtocolVariant] = MappingProxyType(
    {
        variant.name: variant
        for aspect in SQL_RISK_ASPECTS
        for family in SQL_RISK_PROTOCOL_FAMILIES
        for variant in (ProtocolVariant(aspect=aspect, family=family),)
    }
)
SQL_RISK_PROTOCOL_CANDIDATES: Tuple[str, ...] = tuple(_PROTOCOL_VARIANTS)

_COMMON_SAFETY = (
    "Сохраняй точные source/target-роли и immutable scope. Не угадывай "
    "таблицы, поля, transformation_id или file_id; глобальный S2T mapping "
    "нельзя сужать по файлу. Анализируй только выбранный аспект."
)


def configured_sql_risk_protocol(value: str | None) -> str | None:
    """Return an active canonical candidate, or ``None`` for current mode."""
    selected = selected_sql_risk_protocol(value)
    return None if selected is None else selected.name


def selected_sql_risk_protocol(
    value: str | None,
) -> ProtocolVariant | None:
    """Parse a raw setting without reading environment or changing state."""
    if value is None:
        return None
    normalized = value.strip()
    if normalized in {"", "default", "current"}:
        return None
    if normalized not in _PROTOCOL_VARIANTS:
        raise ValueError(
            "Unknown OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT value "
            f"{normalized!r}; expected default/current or one of the 20 "
            "canonical '<aspect>__<family>' candidates"
        )
    return _PROTOCOL_VARIANTS[normalized]


def protocol_sha256(text: str) -> str:
    """Return a stable UTF-8 digest for an exact rendered protocol context."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _support_summary(spec: AspectSpec) -> str:
    if not spec.mandatory_support:
        return "обязательных support-artifacts нет"
    return "; ".join(spec.mandatory_support)


def _conditional_support_summary(spec: AspectSpec) -> str:
    if not spec.conditional_support:
        return "conditional support отсутствует"
    return "; ".join(spec.conditional_support)


def _co_location_rule(spec: AspectSpec) -> str:
    conditional = (
        " Активированный conditional support становится mandatory для "
        "исходного запроса и остаётся в той же task."
        if spec.conditional_support
        else ""
    )
    return (
        "Primary и весь mandatory support помести в одну самодостаточную "
        "worker task; не дроби их между workers."
        + conditional
    )


def _ledger_entries(spec: AspectSpec) -> str:
    entries = [
        f"- `S` (immutable scope): {spec.exact_scope}.",
        f"- `R1` (primary, mandatory): {spec.primary_artifact}.",
    ]
    entries.extend(
        f"- `R{index}` (support, mandatory): {support}."
        for index, support in enumerate(spec.mandatory_support, start=2)
    )
    if not spec.mandatory_support:
        entries.append("- `R2+`: обязательные support-entries отсутствуют.")
    entries.extend(
        f"- `C{index}` (conditional; mandatory только при явном trigger): "
        f"{support}."
        for index, support in enumerate(spec.conditional_support, start=1)
    )
    if not spec.conditional_support:
        entries.append("- `C1+`: conditional support отсутствует.")
    return "\n".join(entries)


def _minimal_artifact(spec: AspectSpec, stage: ProtocolStage) -> str:
    support = _support_summary(spec)
    conditional = _conditional_support_summary(spec)
    co_location = _co_location_rule(spec)
    by_stage = {
        "plan": (
            "Составь минимальный data-plan; не добавляй задачи анализа или "
            f"оформления. {co_location} Primary: "
            f"{spec.primary_artifact}. Support: {support}. Scope: "
            f"{spec.exact_scope}. Conditional support: {conditional}."
        ),
        "planner": (
            "Внутри одной worker task на каждое активное требование сделай "
            "одно точное untruncated чтение, сохрани evidence и не повторяй "
            "закрытое требование. Не сужай и "
            f"не меняй scope: {spec.exact_scope}. Primary: "
            f"{spec.primary_artifact}. Support: {support}. Conditional "
            f"support: {conditional}."
        ),
        "observer": (
            "Проверяй только exact scope, primary, mandatory support и "
            "активированный conditional support одной worker task. `complete` "
            "допустим, когда эти чтения полны; не оценивай риск. "
            f"Правило terminal absence: {spec.terminal_absence_rule}."
        ),
        "upstream_decision": (
            "Верни `pass`, если evidence позволяет подтверждённый, условный "
            "или terminal-ответ. Не делай reroute только для снятия допустимой "
            f"неопределённости. Граница: {spec.evidence_boundary}. Terminal: "
            f"{spec.terminal_absence_rule}."
        ),
        "upstream": (
            "Ответь только по выбранному аспекту в порядке `status → "
            "mechanism → condition → evidence boundary`. Mechanism: "
            f"{spec.mechanism}. Condition: {spec.condition}. Evidence "
            f"boundary: {spec.evidence_boundary}. Terminal: "
            f"{spec.terminal_absence_rule}."
        ),
    }
    return by_stage[stage]


def _evidence_ledger(spec: AspectSpec, stage: ProtocolStage) -> str:
    ledger = _ledger_entries(spec)
    co_location = _co_location_rule(spec)
    by_stage = {
        "plan": (
            "Создай immutable evidence ledger. "
            f"{co_location} Все активные `S`, `R1`, `R2+` и `C1+` остаются "
            "co-located. Не добавляй analysis task.\n" + ledger
        ),
        "planner": (
            "Внутри одной worker task закрывай каждую active named ledger-"
            "entry ровно одним exact call и сохраняй её evidence reference. "
            "Не изменяй `S`, не повторяй satisfied-entry и не применяй "
            "narrowing.\n" + ledger
        ),
        "observer": (
            "Для каждой mandatory или activated conditional ledger-entry "
            "одной worker task поставь ровно один state: "
            "`satisfied`, `missing` или `invalid`. `complete` допустим только "
            "когда все mandatory entries satisfied; вывод о риске не делай. "
            f"Exact empty обрабатывай так: {spec.terminal_absence_rule}.\n"
            + ledger
        ),
        "upstream_decision": (
            "Сверь immutable ledger. `reroute` разрешён только для названной "
            "mandatory entry со state `missing`/`invalid`; иначе верни `pass` "
            "для confirmed, conditional или terminal outcome. Terminal: "
            f"{spec.terminal_absence_rule}."
        ),
        "upstream": (
            "Сошлись на ledger `S`/`R1`/`R2+`/`C1+` и отдели `confirmed`, "
            "`conditional` и `not_assessed`. Укажи mechanism, condition и "
            f"boundary. Mechanism: {spec.mechanism}. Condition: "
            f"{spec.condition}. Boundary: {spec.evidence_boundary}."
        ),
    }
    return by_stage[stage]


def _epistemic_state_machine(spec: AspectSpec, stage: ProtocolStage) -> str:
    support = _support_summary(spec)
    conditional = _conditional_support_summary(spec)
    co_location = _co_location_rule(spec)
    by_stage = {
        "plan": (
            "Начальное состояние `UNREAD`. Запланируй переход только через "
            "exact primary read; support добавляй лишь когда он mandatory или "
            f"явно активирован. {co_location} "
            f"Primary: {spec.primary_artifact}. Support: {support}. Immutable "
            f"scope: {spec.exact_scope}. Conditional support: {conditional}."
        ),
        "planner": (
            "В одной worker task выполни переход `UNREAD → READ_EXACT`: одно "
            "точное untruncated чтение primary и mandatory/activated support, "
            "с evidence. Не повторяй `READ_EXACT` и не меняй scope: "
            f"{spec.exact_scope}. Support: {support}. Conditional support: "
            f"{conditional}."
        ),
        "observer": (
            "Проверь только переход `UNREAD → READ_EXACT` для exact scope и "
            "всех mandatory/activated чтений одной worker task. Используй "
            "существующие поля observer `status`, `gap` и `limitations`; не "
            "выбирай `ANSWERABLE_*` и не анализируй риск. Exact empty "
            f"принимай по terminal rule: {spec.terminal_absence_rule}."
        ),
        "upstream_decision": (
            "После принятого `READ_EXACT` выбери ровно одно состояние "
            "`ANSWERABLE_CONFIRMED`, `ANSWERABLE_CONDITIONAL` или "
            "`ANSWERABLE_NOT_ASSESSED`. Любое `ANSWERABLE_*` требует `pass`. "
            "`reroute` допустим только пока mandatory/activated read не "
            "достиг `READ_EXACT`; не reroute `ANSWERABLE_CONDITIONAL` ради "
            "превращения в confirmed. "
            f"Evidence boundary: {spec.evidence_boundary}."
        ),
        "upstream": (
            "Назови epistemic state, mechanism, trigger/condition и unknown "
            "boundary; не используй low/medium/high. Mechanism: "
            f"{spec.mechanism}. Trigger/condition: {spec.condition}. Unknown "
            f"boundary: {spec.evidence_boundary}. Terminal: "
            f"{spec.terminal_absence_rule}."
        ),
    }
    return by_stage[stage]


def _decision_table(spec: AspectSpec, stage: ProtocolStage) -> str:
    support = _support_summary(spec)
    conditional = _conditional_support_summary(spec)
    co_location = _co_location_rule(spec)
    by_stage = {
        "plan": (
            "Планируй чтение только discriminators будущей decision row: "
            "exact scope, primary availability и mandatory/explicitly "
            f"activated support. {co_location} Не планируй вывод. Scope: "
            f"{spec.exact_scope}. Primary: {spec.primary_artifact}. Support: "
            f"{support}. Conditional support: {conditional}."
        ),
        "planner": (
            "В одной worker task прочитай каждый mandatory/activated "
            "discriminator одним exact untruncated call, сохрани evidence, не "
            "повторяй и не сужай scope. Primary: "
            f"{spec.primary_artifact}. Support: {support}. Conditional "
            f"support: {conditional}."
        ),
        "observer": (
            "Проверь только наличие discriminator inputs одной worker task и "
            "exact scope; не выбирай conclusion row. `complete` означает, что "
            "mandatory/activated inputs прочитаны. Terminal absence input: "
            f"{spec.terminal_absence_rule}."
        ),
        "upstream_decision": (
            "Выбери ровно одну строку; `reroute` допустим только для "
            "отсутствующего mandatory discriminator. Default not_assessed "
            "row является answerable и требует `pass`.\n\n"
            "| Observed inputs | Decision |\n"
            "|---|---|\n"
            "| Exact mandatory evidence и явный mechanism | pass: confirmed |\n"
            "| Exact mandatory evidence, effect зависит от неизвестного "
            "condition | pass: conditional |\n"
            "| Exact mandatory evidence и применимо terminal absence rule | "
            "pass: not_assessed/not_detected |\n"
            "| Нет mandatory discriminator | reroute: назови только его |"
        ),
        "upstream": (
            "Выбери одну decision row и сообщи `observed inputs → conclusion "
            "→ condition → evidence boundary`; не смешивай строки. "
            f"Mechanism discriminator: {spec.mechanism}. Condition: "
            f"{spec.condition}. Boundary: {spec.evidence_boundary}. Terminal: "
            f"{spec.terminal_absence_rule}."
        ),
    }
    return by_stage[stage]


_FAMILY_RENDERERS = {
    "minimal_artifact": _minimal_artifact,
    "evidence_ledger": _evidence_ledger,
    "epistemic_state_machine": _epistemic_state_machine,
    "decision_table": _decision_table,
}


def render_sql_risk_protocol(
    candidate: str,
    aspects: Iterable[SqlRiskAspect],
    *,
    stage: str,
) -> str:
    """Render one candidate after strict aspect and stage validation."""
    variant = _PROTOCOL_VARIANTS.get(candidate)
    if variant is None:
        raise ValueError(f"Unknown SQL-risk protocol candidate: {candidate!r}")
    if stage not in SQL_RISK_PROTOCOL_STAGES:
        raise ValueError(
            f"Unknown SQL-risk protocol stage {stage!r}; expected one of "
            f"{SQL_RISK_PROTOCOL_STAGES}"
        )

    selected = tuple(aspects)
    unknown = [aspect for aspect in selected if aspect not in SQL_RISK_ASPECT_SPECS]
    if unknown:
        raise ValueError(
            "SQL-risk protocol received unknown selected aspect(s): "
            + ", ".join(repr(aspect) for aspect in unknown)
        )
    if len(selected) != 1:
        raise ValueError(
            "SQL-risk protocol candidate requires exactly one selected typed "
            f"aspect; received {len(selected)}"
        )
    if selected[0] != variant.aspect:
        raise ValueError(
            f"SQL-risk protocol candidate {candidate!r} is for aspect "
            f"{variant.aspect!r}, but selected aspect is {selected[0]!r}"
        )

    spec = SQL_RISK_ASPECT_SPECS[variant.aspect]
    renderer = _FAMILY_RENDERERS[variant.family]
    body = renderer(spec, cast(ProtocolStage, stage))
    return (
        "## Анализ SQL-рисков — экспериментальный протокол\n"
        f"Кандидат: `{variant.name}`. Выбранный аспект: `{spec.aspect}`.\n"
        f"{_COMMON_SAFETY}\n"
        f"{body}"
    )


def protocol_variant_sha256(
    variant_or_name: ProtocolVariant | str,
) -> str:
    """Hash the canonical name and all five contexts in fixed stage order."""
    if isinstance(variant_or_name, ProtocolVariant):
        variant = variant_or_name
        registered = _PROTOCOL_VARIANTS.get(variant.name)
        if registered != variant:
            raise ValueError(
                f"Unknown SQL-risk protocol variant: {variant.name!r}"
            )
    else:
        variant = _PROTOCOL_VARIANTS.get(variant_or_name)
        if variant is None:
            raise ValueError(
                f"Unknown SQL-risk protocol candidate: {variant_or_name!r}"
            )

    parts = [variant.name]
    parts.extend(
        f"[{stage}]\n"
        + render_sql_risk_protocol(
            variant.name,
            [variant.aspect],
            stage=stage,
        )
        for stage in SQL_RISK_PROTOCOL_STAGES
    )
    return protocol_sha256("\n\n".join(parts))


__all__ = [
    "AspectSpec",
    "OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV",
    "ProtocolFamily",
    "ProtocolStage",
    "ProtocolVariant",
    "SQL_RISK_ASPECTS",
    "SQL_RISK_ASPECT_SPECS",
    "SQL_RISK_PROTOCOL_CANDIDATES",
    "SQL_RISK_PROTOCOL_FAMILIES",
    "SQL_RISK_PROTOCOL_STAGES",
    "configured_sql_risk_protocol",
    "protocol_sha256",
    "protocol_variant_sha256",
    "render_sql_risk_protocol",
    "selected_sql_risk_protocol",
]
