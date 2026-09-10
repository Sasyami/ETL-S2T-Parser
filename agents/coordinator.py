"""LLM-driven workers with separate upstream and downstream coordination."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from .agent import chat_model
from .cardinality_sufficiency import (
    complete_cardinality_mapping_evidence_ids,
)
from .contracts import (
    EvidenceArtifact,
    MAX_PLAN_STEPS,
    PlanStep,
    SqlRiskAspect,
    UpstreamDecision,
    UpstreamOutput,
    WORKER_OPERATION_COMPLETENESS_MARKER,
    WORKER_OPERATION_EXECUTION_MARKER,
    WORKER_PREVIOUS_RESULTS_MARKER,
    WorkerOutcome,
    WorkerPlan,
)
from .chat_graph import WorkerDisplayItem
from .observability import get_callback_handler, langfuse_trace_context
from .operation_intent import is_exclusive_value_change_request
from .operation_protocols import (
    OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV,
    protocol_variant_sha256,
    selected_sql_risk_protocol,
)
from .plan_origin import PlanOriginError, validate_worker_plan_origin
from .plan_requirements import (
    ReroutePlanRequirementError,
    SqlRiskPlanRequirementError,
    validate_sql_risk_plan_requirements,
    validate_sql_risk_reroute_plan,
)
from .run_metrics import (
    get_run_metrics_callback,
    llm_stage,
    record_coordinator_plan,
    record_entity_resolution,
    record_sql_risk_facts,
    record_upstream_output,
    record_validation_protocol,
    record_worker_outcome,
)
from .sql_risk_scope_contract import (
    SqlRiskScopeContract,
    build_sql_risk_scope_contract,
    ensure_sql_risk_answer_scope,
    missing_sql_risk_requirements,
    render_sql_risk_scope_contract,
)
from .tools.context import (
    OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV,
    OPERATION_SKILL_CATALOG,
    get_downstream_capability_context,
    get_downstream_table_context,
    load_operation_skills,
    load_upstream_analysis_context,
)
from .worker import (
    discard_worker_display_refs,
    register_worker_display_items,
    worker_chat,
)
from .tools.saved_results import (
    SavedResultStore,
    get_active_saved_result_store,
    saved_result_store_scope,
)
from .test_protocol import (
    MAX_PROTOCOL_OBJECTS,
    PROTOCOL_CHECKS,
    RawTestProtocolContract,
    build_test_protocol_display_payloads,
    compile_test_protocol,
    render_test_protocol_answer,
)
from .test_protocol_resolution import (
    resolve_test_protocol_contract,
    validate_raw_contract_origin,
)
from .validation_protocol import (
    LLM_ANALYSES,
    MAX_ANALYSIS_OBJECTS,
    REQUESTED_ANALYSES,
    S2TAnalysisOutput,
    ValidationProtocolContract,
    ValidationProtocolDataError,
    build_deterministic_analysis_items,
    build_s2t_analysis_display_payloads,
    build_s2t_analysis_payload,
    merge_s2t_analysis_output,
    read_test_protocol_inputs,
    read_validation_protocol_inputs,
    render_s2t_analysis_answer,
    validate_s2t_analysis_output,
)
from .value_change_analysis import (
    FieldValueChangeFact,
    derive_field_value_change_facts,
    field_value_change_payload,
    render_field_value_change_answer,
)
from .write_semantics_analysis import (
    WriteSemanticsFact,
    derive_write_semantics_facts,
    is_exclusive_write_semantics_request,
    render_terminal_write_semantics_negative,
    write_semantics_payload,
)

logger = logging.getLogger(__name__)


_SERIALIZED_EVIDENCE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])evidence_[0-9a-f]+(?![A-Za-z0-9_])"
)
_SERIALIZED_EVIDENCE_KEYS = frozenset(
    {
        "used_evidence_ids",
        "display_evidence_ids",
        "evidence_id",
        "displayable",
    }
)
_SERIALIZED_EVIDENCE_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:used_evidence_ids|display_evidence_ids|"
    r"evidence_id|displayable)(?![A-Za-z0-9_])"
)
_SERIALIZED_JSON_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:true|false|null)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SERIALIZED_JSON_NOISE_RE = re.compile(r"^[\s\[\]{},:\"'\\]*$")

COORDINATOR_MAX_WORKERS = MAX_PLAN_STEPS
COORDINATOR_MAX_CYCLES = 2
COORDINATOR_CONTEXT_MAX_CHARS = 4000
_PLAN_TOOL_NAME = "submit_worker_plan"
_SQL_RISK_OPERATION_SKILL = "Анализ SQL-рисков"
_DEFAULT_SQL_RISK_PROTOCOL = "default/current"
_OPERATION_SKILL_TOOL_NAME = "select_operation_skills"
_S2T_ANALYSIS_CONTRACT_TOOL_NAME = "submit_s2t_analysis_contract"
_VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME = "submit_validation_protocol_contract"
_S2T_ANALYSIS_TOOL_NAME = "submit_s2t_analysis"
_UPSTREAM_ANSWER_TOOL_NAME = "submit_upstream_answer"
_UPSTREAM_DATA_DECISION_TOOL_NAME = "submit_upstream_data_decision"
_UPSTREAM_ANALYSIS_CONTEXT = load_upstream_analysis_context()
_DOWNSTREAM_CAPABILITY_CONTEXT = get_downstream_capability_context()
_DOWNSTREAM_TABLE_CONTEXT = get_downstream_table_context()


def _typed_sql_risk_aspects_enabled() -> bool:
    """Return whether E2 routes only explicitly selected SQL-risk aspects."""
    value = os.getenv(OPERATION_SQL_RISK_ASPECTS_EXPERIMENT_ENV)
    if value is None:
        return True
    return value.strip().casefold() not in {"0", "false", "no", "off"}


def _sql_risk_protocol_attestation(
    operation_skills: Sequence[str],
    sql_risk_aspects: Sequence[SqlRiskAspect],
) -> Dict[str, Any]:
    """Describe the exact opt-in SQL-risk protocol without changing it."""
    if _SQL_RISK_OPERATION_SKILL not in operation_skills:
        return {}

    variant = selected_sql_risk_protocol(
        os.getenv(OPERATION_SQL_RISK_PROTOCOL_EXPERIMENT_ENV)
    )
    if variant is None:
        return {
            "operation_sql_risk_protocol": _DEFAULT_SQL_RISK_PROTOCOL,
            "operation_sql_risk_protocol_sha256": None,
        }

    selected_aspects = tuple(sql_risk_aspects)
    if selected_aspects != (variant.aspect,):
        raise ValueError(
            f"SQL-risk protocol candidate {variant.name!r} is for aspect "
            f"{variant.aspect!r}, but selected aspects are "
            f"{selected_aspects!r}"
        )
    return {
        "operation_sql_risk_protocol": variant.name,
        "operation_sql_risk_protocol_sha256": (
            protocol_variant_sha256(variant)
        ),
    }


def _scope_evidence_calls(
    artifacts: Sequence[EvidenceArtifact],
    store: Optional[SavedResultStore],
) -> List[Dict[str, Any]]:
    """Project accepted artifacts onto source-completeness call evidence.

    ``EvidenceArtifact.truncated`` combines two different boundaries: source
    truncation and harmless model-preview clipping.  Required SQLite readers
    are materialized, so their saved descriptor is authoritative whenever a
    dataset reference exists.
    """

    calls: List[Dict[str, Any]] = []
    for artifact in artifacts:
        source_truncated = artifact.truncated
        if artifact.dataset_ref is not None:
            descriptor = (
                store.descriptor(artifact.dataset_ref)
                if store is not None
                else None
            )
            source_truncated = not (
                descriptor is not None
                and descriptor.source_tool == artifact.tool_name
                and not descriptor.truncated
                and descriptor.source_total is not None
                and descriptor.source_total == descriptor.row_count
            )
        calls.append(
            {
                "tool_name": artifact.tool_name,
                "args": artifact.compact_args,
                "truncated": source_truncated,
            }
        )
    return calls


def _first_scope_step_index(
    steps: Sequence[Any],
    contract: Optional[SqlRiskScopeContract],
) -> Optional[int]:
    """Return the zero-based first plan step containing both exact endpoints."""

    if contract is None:
        return None
    source_token = contract.scope.source.casefold()
    target_token = contract.scope.target.casefold()
    for index, raw_step in enumerate(steps):
        task = PlanStep.model_validate(raw_step).task.casefold()
        if source_token in task and target_token in task:
            return index
    return None


class CoordinatorAnswer(BaseModel):
    """Coordinator output consumed by the top-level supervisor."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    display_refs: List[str] = Field(default_factory=list)


class OperationSkillSelection(BaseModel):
    """Execution route and prompt profiles selected for one operation."""

    model_config = ConfigDict(extra="forbid")

    pipeline: Literal[
        "agentic",
        "validation_protocol",
    ] = "agentic"
    skills: List[str]
    sql_risk_aspects: List[SqlRiskAspect] = Field(default_factory=list)

    @model_validator(mode="after")
    def _aspects_match_selected_skill(self) -> "OperationSkillSelection":
        self.skills = list(dict.fromkeys(self.skills))
        self.sql_risk_aspects = list(dict.fromkeys(self.sql_risk_aspects))
        if not _typed_sql_risk_aspects_enabled():
            # E2 baseline deliberately loads the complete legacy profile.
            self.sql_risk_aspects = []
            return self
        if (
            self.sql_risk_aspects
            and "Анализ SQL-рисков" not in self.skills
        ):
            raise ValueError(
                "sql_risk_aspects require Анализ SQL-рисков skill"
            )
        if (
            "Анализ SQL-рисков" in self.skills
            and not self.sql_risk_aspects
        ):
            raise ValueError(
                "Анализ SQL-рисков requires at least one sql_risk_aspect"
            )
        return self


class CoordinatorWorkerRun(TypedDict):
    cycle: int
    step: int
    outcome: WorkerOutcome


class CoordinatorGraphState(TypedDict):
    task: str
    context: str
    operation_skills: Optional[List[str]]
    operation_sql_risk_aspects: Optional[List[SqlRiskAspect]]
    operation_pipeline: Optional[
        Literal["agentic", "s2t_analysis", "validation_protocol"]
    ]
    cycle: int
    plan: List[Dict[str, Any]]
    next_step: int
    worker_runs: List[CoordinatorWorkerRun]
    upstream_problem: Optional[str]
    upstream_output: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    selected_display_refs: List[str]


_OPERATION_SKILL_CATALOG_CONTEXT = "\n".join(
    f"- `{name}` — {description}"
    for name, description in OPERATION_SKILL_CATALOG.items()
)

_SQL_RISK_TYPED_ROUTER_GUIDANCE = """
Если выбрана `Анализ SQL-рисков`, заполни `sql_risk_aspects` только аспектами,
которые прямо нужны результату: `row_filtering`, `cardinality`,
`constraint_rejection`, `value_changes`, `write_semantics`. Для остальных
skills и при `skills=[]` верни `sql_risk_aspects=[]`. Не добавляй все аспекты
автоматически.
""".strip()

_SQL_RISK_TYPED_ROUTER_EXAMPLES = """
- «Может ли этот JOIN размножить строки?» → `Анализ SQL-рисков`,
  `sql_risk_aspects=["cardinality"]`;
- «Отфильтрует ли WHERE строки?» → `sql_risk_aspects=["row_filtering"]`;
- «Может ли NOT NULL отклонить загрузку?» →
  `sql_risk_aspects=["constraint_rejection"]`;
- «Меняет ли CASE значение?» → `sql_risk_aspects=["value_changes"]`;
- «Это MERGE или append?» → `sql_risk_aspects=["write_semantics"]`;
""".strip()

_SQL_RISK_LEGACY_ROUTER_GUIDANCE = """
Эксперимент выбора аспектов отключён: для любого маршрута всегда верни
`sql_risk_aspects=[]`. При выборе `Анализ SQL-рисков` код загрузит полный
legacy-профиль риска; не перечисляй отдельные аспекты.
""".strip()

_SQL_RISK_LEGACY_ROUTER_EXAMPLES = """
- «Может ли этот JOIN размножить строки?» → `Анализ SQL-рисков`,
  `sql_risk_aspects=[]`;
""".strip()

_OPERATION_SKILL_PROMPT = f"""
Ты operation router. Один раз для всей `original_task` выбери исполнительный
`pipeline` и operation-skills. Верни ровно один native call
`{_OPERATION_SKILL_TOOL_NAME}`.

`pipeline="validation_protocol"` выбирай для явной просьбы составить explicit,
standard либо exhaustive SQL test protocol внешней Greenplum-проверки
source→target S2T-загрузки. Файл необязателен: без него catalog-dependent checks
будут помечены unavailable, а остальные всё равно компилируются. Сюда относятся
row/key/field/schema/aggregate reconciliation, uniqueness, NULL и статический
preflight. SQL только проектируется и не исполняется.

Во всех остальных случаях выбирай `pipeline="agentic"`; выбранные
operation-skills направят downstream, workers и upstream внутри общего потока.

Operation-skill — профиль результата, которого добивается пользователь, а не
источник данных, тип объекта или retrieval-skill. Выбирай профиль по intent и
однозначно требуемому результату: буквальное название профиля в запросе не
требуется. Не выбирай профиль только из-за связанных терминов. Несколько
профилей допустимы, только если для ответа действительно нужны несколько
разных видов анализа; не добавляй смежный анализ «на всякий случай».

{_SQL_RISK_TYPED_ROUTER_GUIDANCE}

`skills=[]` — нормальный вариант по умолчанию. Оставляй массив пустым для
простого чтения, списка либо объяснения одной сохранённой трансформации, если
пользователь не просит сравнение атрибутов, оценку риска строк, разность покрытия
маппинга или проектирование проверки. Само наличие SQL, S2T, пары source→target,
колонок либо слова «трансформация» не является основанием выбрать профиль.

Примеры:
- «Покажи SQL transformation A → B» → `skills=[]`;
{_SQL_RISK_TYPED_ROUTER_EXAMPLES}
- «Какие mandatory target fields не замаплены?» → `Покрытие маппинга`.

Доступные operation-skills:
{_OPERATION_SKILL_CATALOG_CONTEXT}

Не отвечай на задачу, не планируй чтение и не придумывай новый профиль.
""".strip()

_OPERATION_SKILL_REPAIR_PROMPT = f"""
Предыдущий native call `{_OPERATION_SKILL_TOOL_NAME}` нарушает схему или содержит
имя вне каталога. Верни ровно один исправленный call с полями `pipeline`,
`skills` и `sql_risk_aspects`. Pipeline — `agentic` либо
`validation_protocol`; массив skills может быть пустым. Аспекты допустимы только
для `Анализ SQL-рисков`; при выборе этого профиля верни хотя бы один нужный
аспект, иначе верни пустой массив. Используй только дословные имена из каталогов.
""".strip()


def _operation_skill_prompt() -> str:
    """Build an E2-consistent router prompt for the active variant."""
    if _typed_sql_risk_aspects_enabled():
        return _OPERATION_SKILL_PROMPT
    return _OPERATION_SKILL_PROMPT.replace(
        _SQL_RISK_TYPED_ROUTER_GUIDANCE,
        _SQL_RISK_LEGACY_ROUTER_GUIDANCE,
    ).replace(
        _SQL_RISK_TYPED_ROUTER_EXAMPLES,
        _SQL_RISK_LEGACY_ROUTER_EXAMPLES,
    )


def _operation_skill_repair_prompt() -> str:
    """Build repair instructions that match the active E2 schema."""
    if _typed_sql_risk_aspects_enabled():
        return _OPERATION_SKILL_REPAIR_PROMPT
    return (
        f"Предыдущий native call `{_OPERATION_SKILL_TOOL_NAME}` нарушает "
        "схему или содержит имя вне каталога. Верни ровно один исправленный "
        "call с полями `pipeline`, `skills` и `sql_risk_aspects`. Pipeline — "
        "`agentic` либо `validation_protocol`; массив skills может быть "
        "пустым. Эксперимент выбора SQL-risk aspects отключён, поэтому всегда "
        "верни `sql_risk_aspects=[]`. Используй только дословные имена из "
        "каталога."
    )

_S2T_ANALYSIS_CONTRACT_PROMPT = f"""
Извлеки только явно заданный контракт S2T-анализа из `original_task`. Верни
ровно один native call `{_S2T_ANALYSIS_CONTRACT_TOOL_NAME}`.

- Каждый элемент `source_tables` и `target_tables` копируй дословно; направление
  задаёт source→target в запросе. Сохрани все явно перечисленные таблицы
  соответствующей роли. Файл копируй как явно заданный `file_id` либо
  `filename`; если он не задан и каталог не нужен, не придумывай scope.
- `requested_analyses` содержит только явно запрошенные классы:
  `row_loss_risk`, `duplicate_risk`, `unmapped_required_fields`,
  `transformation_consistency`.
- Не придумывай идентификаторы, scope-поля, SQL и дополнительные виды анализа.
""".strip()

_S2T_ANALYSIS_CONTRACT_REPAIR_PROMPT = f"""
Предыдущий `{_S2T_ANALYSIS_CONTRACT_TOOL_NAME}` нарушает строгую схему. Верни один
исправленный native call с непустыми точными массивами
`source_tables`/`target_tables` и `requested_analyses` из разрешённого enum.
`file_id` и `filename` необязательны и допустимы только при явном наличии в
`original_task`. Остальные значения также копируй только из `original_task`.
""".strip()

_VALIDATION_PROTOCOL_CONTRACT_PROMPT = f"""
Извлеки только явно заданный контракт SQL test protocol из `original_task`.
Верни ровно один native call `{_VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME}`.

- Файл необязателен. Явный числовой идентификатор копируй как `file_id`, а
  буквальное имя или смысловое описание — как `file_mention`; не заполняй оба.
- `loads` содержит отдельный элемент для каждой явно заданной загрузки;
  `source_mentions` — все исходные mentions именно этого `target_mention`.
  Копируй mention дословно, включая опечатку, неполное или смысловое имя:
  канонизацию выполнит код после этого native call.
- `mode="explicit"`, когда пользователь перечислил проверки; сохрани их в
  `requested_checks` всего contract либо конкретного load. `mode="standard"`
  для общей просьбы о тест-протоколе, `mode="exhaustive"` для максимально
  полного протокола. Не добавляй не запрошенные checks в explicit mode.
- Допустимые checks: {', '.join(PROTOCOL_CHECKS)}.
- Явно заданные поля ключа копируй в `explicit_key` contract/load.
- Не придумывай таблицы, колонки, SQL, проверки и scope-поля.
""".strip()

_VALIDATION_PROTOCOL_CONTRACT_REPAIR_PROMPT = f"""
Предыдущий `{_VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME}` нарушает строгую схему.
Верни один исправленный native call с `mode`, `requested_checks` и непустым
`loads`; в каждом load нужны непустые literal `source_mentions`, один
`target_mention`, `requested_checks` и при наличии `explicit_key`. Файл
необязателен; допустим только исходный `file_id` либо `file_mention`. Копируй
только значения из `original_task`, не исправляй mentions самостоятельно.
""".strip()

_S2T_ANALYSIS_PROMPT = f"""
Ты S2T analyzer. Получаешь строгий contract, lossless-компактные результаты
readers и детерминированные derived_facts. Верни ровно один native call
`{_S2T_ANALYSIS_TOOL_NAME}`.

Для каждой `target_tables` дай один вывод `transformation_consistency`; в
`target_table` дословно укажи текущий target. Сопоставь явно запрошенные
`source_tables` со
  `derived_facts.mapping_coverage.s2t_source_tables`. Missing/additional S2T
  sources, покрытие catalog-полей, числа и physical_source_tables бери дословно
  только из готового mapping_coverage. Если `unmapped_catalog_fields` не пуст,
  не утверждай полного покрытия каталога; не путай его с отдельным покрытием
  обязательных полей. Физические таблицы обязательно перечисли отдельно как
  SQL-зависимости, но отличие физического имени от логического S2T-имени само по
  себе не является противоречием. Оцени пустые/разные точные правила и
  неоднозначные source→target маппинги.

`mappings` сохраняет каждое raw-вхождение, а `rules` хранит каждый точный SQL
один раз; `rule_id` связывает их. Анализируй только переданные данные. Не
выполняй и не сочиняй запросы к
логическим ETL-таблицам, не возвращай `row_loss_risk`, `duplicate_risk` или
`unmapped_required_fields` (их считает код), не генерируй SQL-шаблоны и не
сообщай физические метрики. В evidence называй конкретные подтверждённые поля,
строки или фрагменты правила. Ограничения данных записывай в limitations.
""".strip()

_S2T_ANALYSIS_REPAIR_PROMPT = f"""
Предыдущий `{_S2T_ANALYSIS_TOOL_NAME}` нарушает строгую схему либо не покрывает
ровно все `llm_requested_analyses`. Верни один исправленный native call,
используя только исходный contract, reader_results и derived_facts. Не добавляй
виды анализа и факты.
""".strip()


class CoordinatorResponseError(RuntimeError):
    """Raised when an LLM response violates a structural coordinator contract."""


def _recover_serialized_evidence_ids(
    value: str,
    available_evidence_ids: set[str],
) -> Optional[List[str]]:
    """Recover only known opaque IDs from an obvious JSON/list fragment.

    Some providers occasionally put a serialized evidence-id list (and even
    adjacent schema keys) into one string item of the otherwise valid native
    array. Recovery is intentionally narrower than generic JSON repair: every
    opaque ID must already be available, and after removing known output keys,
    JSON literals, and IDs, only serialization punctuation may remain.
    """

    clean_value = str(value or "").strip()
    if clean_value in available_evidence_ids:
        return [clean_value]

    recovered_ids = _SERIALIZED_EVIDENCE_ID_RE.findall(clean_value)
    key_tokens = _SERIALIZED_EVIDENCE_KEY_RE.findall(clean_value)
    if any(
        evidence_id not in available_evidence_ids
        for evidence_id in recovered_ids
    ):
        return None
    if not recovered_ids and not key_tokens:
        return None

    has_list_or_json_marker = any(
        marker in clean_value for marker in '[]{}",:'
    )
    if not has_list_or_json_marker and (
        recovered_ids or clean_value not in _SERIALIZED_EVIDENCE_KEYS
    ):
        return None

    remainder = _SERIALIZED_EVIDENCE_ID_RE.sub("", clean_value)
    remainder = _SERIALIZED_EVIDENCE_KEY_RE.sub("", remainder)
    remainder = _SERIALIZED_JSON_LITERAL_RE.sub("", remainder)
    if not _SERIALIZED_JSON_NOISE_RE.fullmatch(remainder):
        return None
    return list(dict.fromkeys(recovered_ids))


def _normalize_upstream_evidence_id_list(
    values: Sequence[str],
    available_evidence_ids: set[str],
) -> List[str]:
    """Normalize provider serialization noise without accepting new IDs."""

    normalized: List[str] = []
    for value in values:
        clean_value = str(value or "").strip()
        recovered = _recover_serialized_evidence_ids(
            clean_value,
            available_evidence_ids,
        )
        candidates = [clean_value] if recovered is None else recovered
        for candidate in candidates:
            if candidate and candidate not in normalized:
                normalized.append(candidate)
    return normalized


_DOWNSTREAM_PLAN_PROMPT = f"""
Ты downstream planner. Верни native call `{_PLAN_TOOL_NAME}` с 1–{COORDINATOR_MAX_WORKERS}
`steps`. Каждая task читает необходимые факты.

Каждый step обязан быть незаменимым: без него нельзя ответить на original_task.
Удали незапрошенные проверки, обогащение и реализацию. Наличие
таблицы в справочнике не требует её чтения.

Сохрани сущность, направление, scope и фильтры. Роль source/target известна,
только если привязана к идентификатору в original_task/context или доказана
S2T-строкой; роль результата не задаёт роль кандидата. Точные идентификаторы
бери только из original_task/context, пиши в обратных кавычках без внешней
пунктуации. Не превращай бизнес-термин в техническое имя или tool.

Если файловый scope задан одним или несколькими полными `filename`, а чтению
нужен внутренний `file_id`, сначала запланируй точное разрешение всех имён и
только затем зависимые чтения с соответствующими принятыми результатами.
`file_id` допустим лишь из original_task либо принятого результата разрешения.
Каждая task самодостаточна: повторяй нужные идентификаторы, роли, scope и
фильтры. Результат предыдущего worker может
добавить ранее неизвестное значение, но не заменить уже заданный идентификатор
другой сущности: `filename` даёт `file_id`, но не определяет и не заменяет
`table_name`. Не используй вместо известного имени ссылки «эта таблица»,
«разрешённый объект», «та же target_table» или «найденное имя».

Context заканчивается здесь: перенеси нужные условия в `constraints`,
сущность/scope/полноту — в `entity`/`scope`/`coverage`, зависимости — в
`dependencies` (1-based номера прошлых steps).

Сохраняй тип поиска из original_task:
- смысл/бизнес-смысл/описание/назначение/«наиболее вероятный» при неизвестном
  имени — смысловой поиск цельной естественной фразой;
- содержит/подстрока/фрагмент — буквальный поиск, только если фрагмент явно дан.
Не превращай смысловой поиск в «найти содержащие», набор слов, синонимов,
переводов или OR-вариантов.

Смысл поля ищется в каталогах колонок; при неизвестной роли — сразу в обоих.
Смысл таблицы — в каталогах таблиц, правило — в S2T. Семантический кандидат не
имеет S2T-роли: определи её только по найденной S2T-строке.

SQLite-каталоги хранят метаданные. Значения `table_name`, `source_table` и
`target_table` — логические ETL-объекты, а не SQLite-таблицы для физического SQL.
Для target-объекта читай атрибуты из `target_columns`, для source — из
`source_columns`; глобальную `s2t_transformations` не ограничивай `file_id`.
Планируй чтение только тех фактов, без которых нельзя получить запрошенный
результат. Если пользователь просит описать способ будущего действия, не выполняй
это действие вместо описания. Для вывода по сохранённому выражению сначала читай
само выражение; дополнительные данные запрашивай лишь когда они действительно
нужны исходной задаче.
Не создавай значения для неподтверждённых физических объектов и полей. Передавай
upstream только подтверждённые имена, а нехватку данных опиши явно.

Минимизируй обмен. Последующий worker использует результат предыдущего, только
если без него нельзя читать дальше. Передаются только краткие lazy-ссылки;
зависимая task называет нужный результат и новое чтение, не будущие значения.

Если объект задан только бизнес-смыслом, отдельный worker может сначала получить
технические кандидаты из каталога, а следующий — найти эти кандидаты в S2T.
S2T-поиск по подстроке лексический, не семантический: не передавай ему русский
бизнес-термин, придуманный перевод или предполагаемое имя.

Сравнение, оценку, объяснение, вывод и оформление делает upstream: не создавай
для них tasks. Не выбирай tools/skills и не пиши task как вызов функции. При
reroute построй полный план по original_task и problem; прошлых результатов нет,
а problem не заменяет и не переопределяет явные идентификаторы original_task.

{_DOWNSTREAM_CAPABILITY_CONTEXT}

{_DOWNSTREAM_TABLE_CONTEXT}
""".strip()

_DOWNSTREAM_PLAN_REPAIR_PROMPT = f"""
Предыдущий native call `{_PLAN_TOOL_NAME}` нарушает схему или смысловой контракт.
Верни исправленный native call ровно один раз. Массив `steps` должен содержать
от 1 до {COORDINATOR_MAX_WORKERS} элементов; каждый элемент должен иметь
непустую `task` и может содержать только разрешённые structured-поля
`constraints`, `entity`, `scope`, `coverage`, `dependencies`.
Сохрани запрошенные роли, объекты, фильтры и результаты. Не придумывай
идентификаторы, функции, tools или требования. Используй только реальные таблицы
хранилища из system prompt; неизвестные бизнес-объекты оставляй текстом поиска.
Разрешение идентификатора одной сущности не заменяет идентификаторы другой;
явно известные значения повторяй дословно в каждой зависимой task.
Не добавляй анализ и оформление.

Причина отклонения: {{validation_error}}
""".strip()

_UPSTREAM_DATA_DECISION_PROMPT = f"""
Ты проверяешь достаточность `evidence` для `original_task`. `worker_outcomes`
различает complete, partial и failed чтения, причины остановки и
незакрытые требования. Partial evidence можно использовать только в его
подтверждённой границе; failed/unavailable нельзя считать полным. Верни один native call
`{_UPSTREAM_DATA_DECISION_TOOL_NAME}`:

- `decision="pass"`, если можно дать конечный ответ;
- `decision="reroute"`, если нужен новый цикл чтения.

При reroute необязательный `problem` кратко описывает недостающие данные новому
downstream-плану.
Не формируй пользовательский ответ и не выбирай display-results.

В `problem` не предлагай имена таблиц, колонок, схем, технические синонимы или
значения, которых нет во входе. Описывай только недостающий факт или чтение.

В evidence: `evidence_id`, `tool_name`, точные `args`, фактический `preview`,
`truncated`, `displayable`. Args подтверждают область чтения, preview — данные.
Не додумывай; `truncated=true` не подтверждает полный набор.

Сопоставь каждый запрошенный исходный результат и его scope с прямым
подтверждением в evidence. Нельзя считать значение одной метрики подтверждением
другой.

Промежуточный список кандидатов не подтверждает связь, правило, маппинг или
lineage. Если original_task требует следующего источника, верни `reroute`.
""".strip()

_UPSTREAM_ANSWER_PROMPT = f"""
Ты upstream answer coordinator. Предварительная проверка уже вернула `pass`.
Вход содержит `original_task`, состояния `worker_outcomes` и принятые
`evidence`. Сам выполни запрошенный
анализ и верни ровно один native call `{_UPSTREAM_ANSWER_TOOL_NAME}` с готовым
`answer`. При наличии подтверждающих evidence передай `used_evidence_ids` и
нужные `display_evidence_ids`.

Evidence содержит `evidence_id`, `tool_name`, точные `args`, фактический
`preview`, `truncated` и признак `displayable`. Аргументы подтверждают область
чтения, preview — найденные данные. Не додумывай отсутствующее; при
`truncated=true` не утверждай полноту набора. `display_evidence_ids` выбирай
как `evidence_id` только у результатов с `displayable=true` и включай также в
`used_evidence_ids`.

Перед `answer` сопоставь каждый запрошенный результат и его scope с прямым
подтверждением в evidence. Нельзя повторять значение одной метрики вместо
отсутствующей другой. Если данных всё же недостаточно, явно укажи это в ответе:
на этом линейном этапе возврата к чтению уже нет.

Соблюдай запрошенный формат. Если буквальный компактный формат не задан, ответ
должен быть самодостаточным: подпиши смысл каждого значения и не возвращай
безымянную CSV-последовательность. Если пользователь потребовал «только»
конкретные элементы, не добавляй вступление и заключение. Для шаблона вида
`имя=<значение>` сохрани имя и знак `=` дословно. Не упоминай coordinators,
workers, tools, previews, result refs и внутреннюю схему.
""".strip()

_UPSTREAM_DATA_DECISION_REPAIR_PROMPT = f"""
Предыдущий native call решения о данных не соответствует схеме. Верни ровно один
`{_UPSTREAM_DATA_DECISION_TOOL_NAME}` с обязательным `decision`: `pass` или
`reroute`. `problem` опционален. Не формируй ответ и не выбирай evidence.
""".strip()

_UPSTREAM_ANSWER_REPAIR_PROMPT = f"""
Предыдущий upstream answer call не соответствует схеме. Верни ровно один
`{_UPSTREAM_ANSWER_TOOL_NAME}` с обязательным `answer` и опциональными evidence
IDs. Используй только доступные evidence_id и не добавляй факты.
""".strip()


def _operation_skill_tool_schema() -> Dict[str, Any]:
    typed_sql_risk_aspects = _typed_sql_risk_aspects_enabled()
    sql_risk_aspects_schema: Dict[str, Any] = {
        "type": "array",
        "maxItems": 5 if typed_sql_risk_aspects else 0,
        "uniqueItems": True,
        "items": {
            "type": "string",
            "enum": [
                "row_filtering",
                "cardinality",
                "constraint_rejection",
                "value_changes",
                "write_semantics",
            ],
        },
        "description": (
            "Только явно нужные аспекты профиля Анализ SQL-рисков; "
            "при выборе этого профиля нужен хотя бы один аспект."
            if typed_sql_risk_aspects
            else (
                "E2 baseline: массив всегда пуст; полный legacy-профиль "
                "SQL-рисков загружается кодом."
            )
        ),
    }
    return {
        "type": "function",
        "function": {
            "name": _OPERATION_SKILL_TOOL_NAME,
            "description": (
                "Выбрать применимые prompt-профили для всей операции."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": {
                        "type": "string",
                        "enum": [
                            "agentic",
                            "validation_protocol",
                        ],
                        "description": (
                            "Общий агентный поток либо компиляция внешнего "
                            "SQL test protocol."
                        ),
                    },
                    "skills": {
                        "type": "array",
                        "maxItems": len(OPERATION_SKILL_CATALOG),
                        "items": {
                            "type": "string",
                            "enum": list(OPERATION_SKILL_CATALOG),
                        },
                        "description": (
                            "Точные имена применимых профилей; пустой массив "
                            "означает, что специальный профиль не нужен."
                        ),
                    },
                    "sql_risk_aspects": sql_risk_aspects_schema,
                },
                "required": ["pipeline", "skills", "sql_risk_aspects"],
                "additionalProperties": False,
            },
        },
    }


def _s2t_analysis_contract_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _S2T_ANALYSIS_CONTRACT_TOOL_NAME,
            "description": "Извлечь строгий контракт S2T-анализа.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Необязательный внутренний scope каталога; только "
                            "если file_id явно дан в запросе."
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                        "description": (
                            "Необязательное точное имя загруженного файла; "
                            "только если оно явно дано в запросе."
                        ),
                    },
                    "source_tables": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_ANALYSIS_OBJECTS,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                        "description": "Все точные source-table из запроса.",
                    },
                    "target_tables": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_ANALYSIS_OBJECTS,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                        "description": "Все точные target-table из запроса.",
                    },
                    "requested_analyses": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": len(REQUESTED_ANALYSES),
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": list(REQUESTED_ANALYSES),
                        },
                        "description": (
                            "Только явно запрошенные виды анализа сохранённого "
                            "S2T."
                        ),
                    },
                },
                "required": [
                    "source_tables",
                    "target_tables",
                    "requested_analyses",
                ],
                "additionalProperties": False,
            },
        },
    }


def _validation_protocol_contract_tool_schema() -> Dict[str, Any]:
    key_schema = {
        "type": "array",
        "minItems": 1,
        "maxItems": 32,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1, "maxLength": 300},
    }
    checks_schema = {
        "type": "array",
        "maxItems": len(PROTOCOL_CHECKS),
        "uniqueItems": True,
        "items": {"type": "string", "enum": list(PROTOCOL_CHECKS)},
    }
    return {
        "type": "function",
        "function": {
            "name": _VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME,
            "description": "Извлечь строгий scope внешнего SQL test protocol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Только явно указанный file_id.",
                    },
                    "file_mention": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                        "description": (
                            "Дословное имя либо смысловое описание файла; "
                            "канонизацию выполняет deterministic resolver."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["explicit", "standard", "exhaustive"],
                    },
                    "requested_checks": checks_schema,
                    "explicit_key": key_schema,
                    "loads": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PROTOCOL_OBJECTS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_mentions": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": MAX_PROTOCOL_OBJECTS,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 300,
                                    },
                                },
                                "target_mention": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 300,
                                },
                                "requested_checks": checks_schema,
                                "explicit_key": key_schema,
                            },
                            "required": [
                                "source_mentions",
                                "target_mention",
                                "requested_checks",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["mode", "requested_checks", "loads"],
                "additionalProperties": False,
            },
        },
    }


def _s2t_analysis_tool_schema() -> Dict[str, Any]:
    item_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(LLM_ANALYSES),
            },
            "target_table": {
                "type": "string",
                "description": "Точный target текущего вывода из contract.",
            },
            "conclusion": {
                "type": "string",
                "description": "Вывод только по переданным S2T/catalog данным.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Конкретные подтверждённые основания вывода.",
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Что нельзя установить по переданным данным.",
            },
        },
        "required": [
            "target_table",
            "kind",
            "conclusion",
            "evidence",
            "limitations",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": _S2T_ANALYSIS_TOOL_NAME,
            "description": "Вернуть выводы анализа сохранённого S2T.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": (
                            MAX_ANALYSIS_OBJECTS * len(LLM_ANALYSES)
                        ),
                        "items": item_schema,
                    }
                },
                "required": ["analyses"],
                "additionalProperties": False,
            },
        },
    }


def _plan_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _PLAN_TOOL_NAME,
            "description": (
                "Зафиксировать последовательность готовых worker tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": COORDINATOR_MAX_WORKERS,
                        "description": (
                            "Необходимые чтения исходных данных; без отдельных "
                            "шагов производного анализа"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": (
                                        "Готовая задача одного worker только "
                                        "на получение необходимых исходных "
                                        "данных; производный анализ выполняется "
                                        "upstream"
                                    ),
                                },
                                "constraints": {
                                    "type": "array",
                                    "maxItems": 20,
                                    "items": {"type": "string"},
                                    "description": (
                                        "Релевантные ограничения из context, "
                                        "материализованные для этого worker."
                                    ),
                                },
                                "entity": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "type": "string",
                                            "enum": [
                                                "source",
                                                "target",
                                                "unknown",
                                            ],
                                        },
                                        "table": {"type": "string"},
                                        "field": {"type": "string"},
                                    },
                                    "additionalProperties": False,
                                },
                                "scope": {
                                    "type": "object",
                                    "properties": {
                                        "file_id": {
                                            "type": "integer",
                                            "minimum": 1,
                                        },
                                        "filename": {"type": "string"},
                                        "sheet_name": {"type": "string"},
                                        "filters": {
                                            "type": "object",
                                            # GigaChat validates every object
                                            # node as a complete JSON Schema
                                            # object, including open maps.
                                            "properties": {},
                                            "additionalProperties": True,
                                        },
                                    },
                                    "additionalProperties": False,
                                },
                                "coverage": {
                                    "type": "string",
                                    "enum": [
                                        "single",
                                        "all_matches",
                                        "top_k",
                                        "aggregate",
                                    ],
                                },
                                "dependencies": {
                                    "type": "array",
                                    "maxItems": COORDINATOR_MAX_WORKERS - 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "integer",
                                        "minimum": 1,
                                    },
                                },
                            },
                            "required": ["task"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
        },
    }


def _upstream_answer_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _UPSTREAM_ANSWER_TOOL_NAME,
            "description": (
                "Вернуть готовый итоговый ответ по достаточным evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Непустой готовый пользовательский ответ.",
                    },
                    "used_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Все evidence_id, использованные в ответе.",
                    },
                    "display_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Подмножество used evidence для отдельного display."
                        ),
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def _upstream_data_decision_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _UPSTREAM_DATA_DECISION_TOOL_NAME,
            "description": (
                "Решить, перейти к upstream answer или повторить чтение данных."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["pass", "reroute"],
                        "description": (
                            "pass продолжает к ответу; reroute повторяет чтение."
                        ),
                    },
                    "problem": {
                        "type": "string",
                        "description": (
                            "Необязательное уточнение нехватки данных для нового плана."
                        ),
                    }
                },
                "required": ["decision"],
                "additionalProperties": False,
            },
        },
    }


def _native_payload(
    message: Any,
    tool_name: str,
    payload_model: type[BaseModel],
) -> BaseModel:
    if not isinstance(message, AIMessage):
        raise CoordinatorResponseError(
            f"Coordinator ожидал AIMessage с native call {tool_name}."
        )
    matching_calls = [
        call for call in message.tool_calls if call.get("name") == tool_name
    ]
    if len(matching_calls) != 1:
        raise CoordinatorResponseError(
            f"Coordinator должен вернуть ровно один native call {tool_name}."
        )
    try:
        return payload_model.model_validate(matching_calls[0].get("args") or {})
    except ValidationError as exc:
        details: List[str] = []
        for item in exc.errors():
            path = ".".join(str(part) for part in item.get("loc") or ())
            message_text = str(item.get("msg") or "validation failed")
            details.append(
                f"{path}: {message_text}" if path else message_text
            )
        detail_text = "; ".join(details)[:1200]
        raise CoordinatorResponseError(
            f"Coordinator вернул невалидную структуру {tool_name}: "
            + detail_text
        ) from exc
    except (TypeError, ValueError) as exc:
        raise CoordinatorResponseError(
            f"Coordinator вернул невалидную структуру {tool_name}: "
            + str(exc)[:1200]
        ) from exc


def _native_upstream_decision(message: Any) -> UpstreamDecision:
    """Parse the data decision made before upstream answer generation."""
    decision = _native_payload(
        message,
        _UPSTREAM_DATA_DECISION_TOOL_NAME,
        UpstreamDecision,
    )
    assert isinstance(decision, UpstreamDecision)
    return decision


def _native_operation_route(message: Any) -> OperationSkillSelection:
    """Parse and validate the once-per-operation execution route."""
    selection = _native_payload(
        message,
        _OPERATION_SKILL_TOOL_NAME,
        OperationSkillSelection,
    )
    assert isinstance(selection, OperationSkillSelection)
    unknown = [
        name for name in selection.skills if name not in OPERATION_SKILL_CATALOG
    ]
    if unknown:
        raise CoordinatorResponseError(
            "Operation router выбрал неизвестные skills: "
            + ", ".join(dict.fromkeys(unknown))
        )
    return selection.model_copy(
        update={"skills": list(dict.fromkeys(selection.skills))}
    )


def _native_s2t_analysis_contract(message: Any) -> ValidationProtocolContract:
    contract = _native_payload(
        message,
        _S2T_ANALYSIS_CONTRACT_TOOL_NAME,
        ValidationProtocolContract,
    )
    assert isinstance(contract, ValidationProtocolContract)
    return contract


def _native_validation_protocol_contract(message: Any) -> RawTestProtocolContract:
    contract = _native_payload(
        message,
        _VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME,
        RawTestProtocolContract,
    )
    assert isinstance(contract, RawTestProtocolContract)
    return contract


def _native_s2t_analysis(message: Any) -> S2TAnalysisOutput:
    output = _native_payload(
        message,
        _S2T_ANALYSIS_TOOL_NAME,
        S2TAnalysisOutput,
    )
    assert isinstance(output, S2TAnalysisOutput)
    return output


def _validate_contract_origin(
    contract: Any,
    original_task: str,
) -> None:
    """Reject identifiers invented by the typed-contract model."""
    task = str(original_task or "")
    missing = [
        value
        for value in (*contract.source_tables, *contract.target_tables)
        if value not in task
    ]
    file_id_missing = (
        contract.file_id is not None
        and f"file_id={contract.file_id}" not in task.replace(" ", "")
    )
    filename = str(getattr(contract, "filename", None) or "").strip()
    filename_missing = bool(
        filename and filename.casefold() not in task.casefold()
    )
    if missing or file_id_missing or filename_missing:
        details = (
            ", ".join(missing)
            if missing
            else f"file_id={contract.file_id}"
            if file_id_missing
            else filename
        )
        raise CoordinatorResponseError(
            "Typed contract содержит идентификатор не из original_task: "
            + details
        )


def _validation_contract_issue(error: Exception) -> Dict[str, Any]:
    """Turn a failed extraction into a stable no-fallback public state."""

    detail = str(error).strip()[:1200]
    lowered = detail.casefold()
    code = (
        "unsupported_check"
        if "requested_checks" in lowered
        and ("input should be" in lowered or "literal" in lowered)
        else "missing_parameter"
    )
    return {
        "code": code,
        "message": detail or "Строгий validation contract не сформирован.",
        "candidates": [],
    }


def _render_validation_failure(
    status: str,
    issues: Sequence[Dict[str, Any]],
) -> str:
    """Render a machine-readable validation failure without an agentic retry."""

    titles = {
        "missing_parameter": "Не хватает обязательных параметров",
        "unsupported_check": "Запрошена неподдерживаемая проверка",
        "unresolved_entity": "Сущность не разрешена",
        "ambiguous_entity": "Сущность неоднозначна",
    }
    lines = [
        "Тест-протокол не сформирован.",
        f"Статус: {status}.",
        titles.get(status, "Validation contract недоступен") + ".",
    ]
    for issue in issues:
        message = str(issue.get("message") or "").strip()
        candidates = [
            str(value)
            for value in issue.get("candidates", [])
            if str(value).strip()
        ]
        if message:
            lines.append(f"- {message}")
        if candidates:
            lines.append("  Кандидаты: " + ", ".join(candidates))
    if status == "ambiguous_entity":
        lines.append("Уточните один точный вариант из списка кандидатов.")
    return "\n".join(lines)


def _native_upstream_answer(message: Any) -> UpstreamOutput:
    """Parse the final answer after the data decision returned pass."""
    output = _native_payload(
        message,
        _UPSTREAM_ANSWER_TOOL_NAME,
        UpstreamOutput,
    )
    assert isinstance(output, UpstreamOutput)
    return output


def _repair_messages(
    base_messages: Sequence[BaseMessage],
    invalid_result: Any,
    repair_prompt: str,
) -> List[BaseMessage]:
    """Build provider-valid history after rejecting a native tool call."""
    messages = list(base_messages)
    if isinstance(invalid_result, AIMessage):
        tool_calls = list(invalid_result.tool_calls)
        call_ids = [
            str(call.get("id") or "").strip() for call in tool_calls
        ]
        if not tool_calls or all(call_ids):
            messages.append(invalid_result)
            for call, call_id in zip(tool_calls, call_ids):
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "status": "rejected",
                                "reason": "native call failed validation",
                            },
                            ensure_ascii=False,
                        ),
                        tool_call_id=call_id,
                        name=str(call.get("name") or "invalid_call"),
                    )
                )
    messages.append(HumanMessage(content=repair_prompt))
    return messages


def build_coordinator_graph(
    model: Any,
    *,
    callbacks: Optional[Sequence[Any]] = None,
    collected_display_refs: Optional[List[str]] = None,
):
    """Build downstream task flow and upstream result flow around workers."""
    callback_list = list(callbacks or [])
    model_config = {"callbacks": callback_list} if callback_list else None

    def bind_required_tool(schema: Dict[str, Any], tool_name: str) -> Any:
        try:
            return model.bind_tools([schema], tool_choice=tool_name)
        except TypeError:
            return model.bind_tools([schema])

    operation_skill_model = bind_required_tool(
        _operation_skill_tool_schema(),
        _OPERATION_SKILL_TOOL_NAME,
    )
    plan_model = bind_required_tool(_plan_tool_schema(), _PLAN_TOOL_NAME)
    upstream_data_decision_model = bind_required_tool(
        _upstream_data_decision_tool_schema(),
        _UPSTREAM_DATA_DECISION_TOOL_NAME,
    )
    upstream_answer_model = bind_required_tool(
        _upstream_answer_tool_schema(),
        _UPSTREAM_ANSWER_TOOL_NAME,
    )

    def invoke(
        selected_model: Any,
        messages: Sequence[BaseMessage],
        *,
        stage: str,
    ) -> Any:
        try:
            with llm_stage(stage):
                return (
                    selected_model.invoke(messages, config=model_config)
                    if model_config is not None
                    else selected_model.invoke(messages)
                )
        except Exception as exc:
            raise CoordinatorResponseError(
                f"Ошибка LLM coordinator: {type(exc).__name__}"
            ) from exc

    def downstream_plan_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        operation_skills = state.get("operation_skills")
        operation_sql_risk_aspects = (
            state.get("operation_sql_risk_aspects") or []
        )
        operation_pipeline = state.get("operation_pipeline")
        if operation_skills is None:
            operation_payload = {
                "original_task": state["task"],
            }
            operation_messages: List[BaseMessage] = [
                SystemMessage(content=_operation_skill_prompt()),
                HumanMessage(
                    content=json.dumps(operation_payload, ensure_ascii=False)
                ),
            ]
            operation_result = invoke(
                operation_skill_model,
                operation_messages,
                stage="operation_router",
            )
            try:
                operation_route = _native_operation_route(operation_result)
            except CoordinatorResponseError as first_error:
                logger.warning(
                    "Operation skill call violated selection schema; "
                    "requesting one LLM repair: %s",
                    first_error,
                )
                operation_result = invoke(
                    operation_skill_model,
                    _repair_messages(
                        operation_messages,
                        operation_result,
                        _operation_skill_repair_prompt()
                        + "\nОшибка: "
                        + str(first_error),
                    ),
                    stage="operation_router",
                )
                operation_route = _native_operation_route(operation_result)
            operation_skills = operation_route.skills
            operation_sql_risk_aspects = (
                operation_route.sql_risk_aspects
            )
            operation_pipeline = operation_route.pipeline
        if operation_pipeline is None:
            operation_pipeline = "agentic"

        if operation_pipeline == "validation_protocol":
            protocol_display_refs: List[str] = []
            protocol_reader_results: List[Dict[str, Any]] = []
            protocol_trace: Dict[str, Any]
            protocol_contract_model = bind_required_tool(
                _validation_protocol_contract_tool_schema(),
                _VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME,
            )
            protocol_payload = {"original_task": state["task"]}
            protocol_messages: List[BaseMessage] = [
                SystemMessage(content=_VALIDATION_PROTOCOL_CONTRACT_PROMPT),
                HumanMessage(
                    content=json.dumps(protocol_payload, ensure_ascii=False)
                ),
            ]
            protocol_result = invoke(
                protocol_contract_model,
                protocol_messages,
                stage="validation_protocol_contract",
            )
            raw_protocol_contract: Optional[RawTestProtocolContract] = None
            extraction_issue: Optional[Dict[str, Any]] = None
            try:
                raw_protocol_contract = _native_validation_protocol_contract(
                    protocol_result
                )
                validate_raw_contract_origin(
                    raw_protocol_contract,
                    state["task"],
                )
            except (CoordinatorResponseError, ValueError) as first_error:
                logger.warning(
                    "Typed test protocol contract was invalid; requesting "
                    "one LLM repair: %s",
                    first_error,
                )
                protocol_result = invoke(
                    protocol_contract_model,
                    _repair_messages(
                        protocol_messages,
                        protocol_result,
                        _VALIDATION_PROTOCOL_CONTRACT_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error),
                    ),
                    stage="validation_protocol_contract",
                )
                try:
                    raw_protocol_contract = _native_validation_protocol_contract(
                        protocol_result
                    )
                    validate_raw_contract_origin(
                        raw_protocol_contract,
                        state["task"],
                    )
                except (CoordinatorResponseError, ValueError) as second_error:
                    logger.warning(
                        "Typed test protocol contract remained invalid; "
                        "returning structured validation state: %s",
                        second_error,
                    )
                    raw_protocol_contract = None
                    extraction_issue = _validation_contract_issue(second_error)

            if raw_protocol_contract is None:
                assert extraction_issue is not None
                failure_status = str(extraction_issue["code"])
                failure_issues = [extraction_issue]
                answer = _render_validation_failure(
                    failure_status,
                    failure_issues,
                )
                protocol_trace = {
                    "mode": None,
                    "status": failure_status,
                    "issues": failure_issues,
                    "phases": [],
                    "targets": [],
                    "reader_calls": [],
                    "silent_fallback": False,
                }
            else:
                try:
                    resolution = resolve_test_protocol_contract(
                        raw_protocol_contract,
                        callbacks=callback_list,
                    )
                except Exception as exc:
                    logger.warning(
                        "Test protocol entity resolution failed safely: %s",
                        exc,
                    )
                    failure_status = "unresolved_entity"
                    failure_issues = [
                        {
                            "code": failure_status,
                            "message": (
                                "Entity resolution недоступен: "
                                f"{type(exc).__name__}."
                            ),
                            "candidates": [],
                        }
                    ]
                    answer = _render_validation_failure(
                        failure_status,
                        failure_issues,
                    )
                    protocol_trace = {
                        "mode": raw_protocol_contract.mode,
                        "status": failure_status,
                        "issues": failure_issues,
                        "phases": [],
                        "targets": [],
                        "reader_calls": [],
                        "silent_fallback": False,
                    }
                else:
                    record_entity_resolution(
                        [
                            {
                                **item.model_dump(mode="json"),
                                "resolver_invoked": item.method != "exact",
                            }
                            for item in resolution.resolutions
                        ]
                    )
                    if resolution.status != "resolved":
                        failure_issues = [
                            item.model_dump(mode="json")
                            for item in resolution.issues
                        ]
                        answer = _render_validation_failure(
                            resolution.status,
                            failure_issues,
                        )
                        protocol_trace = {
                            "mode": raw_protocol_contract.mode,
                            "status": resolution.status,
                            "issues": failure_issues,
                            "phases": [],
                            "targets": [],
                            "reader_calls": [],
                            "exact_bypass_count": (
                                resolution.exact_bypass_count
                            ),
                            "silent_fallback": False,
                        }
                    else:
                        protocol_contract = resolution.contract
                        assert protocol_contract is not None
                        protocol_reader_results = read_test_protocol_inputs(
                            protocol_contract,
                            callbacks=callback_list,
                        )
                        compiled_protocol = compile_test_protocol(
                            protocol_contract,
                            reader_results=protocol_reader_results,
                        )
                        answer = render_test_protocol_answer(
                            protocol_contract,
                            compiled_protocol,
                        )
                        protocol_display_refs = register_worker_display_items(
                            [
                                WorkerDisplayItem(**item)
                                for item in build_test_protocol_display_payloads(
                                    compiled_protocol
                                )
                            ]
                        )
                        if collected_display_refs is not None:
                            collected_display_refs.extend(protocol_display_refs)
                        protocol_trace = compiled_protocol.model_dump(mode="json")
                        protocol_trace.update(
                            {
                                "contract": protocol_contract.model_dump(
                                    mode="json"
                                ),
                                "reader_calls": [
                                    {
                                        key: item.get(key)
                                        for key in (
                                            "kind",
                                            "tool_name",
                                            "args",
                                            "error",
                                        )
                                        if item.get(key) is not None
                                    }
                                    for item in protocol_reader_results
                                ],
                                "exact_bypass_count": (
                                    resolution.exact_bypass_count
                                ),
                                "silent_fallback": False,
                            }
                        )

            record_validation_protocol(protocol_trace)
            direct_plan = [
                {
                    "cycle": state["cycle"],
                    "step": index,
                    "task": task,
                    "operation_skills": list(operation_skills),
                    "sql_risk_aspects": list(
                        operation_sql_risk_aspects
                    ),
                    "pipeline": "validation_protocol",
                }
                for index, task in enumerate(
                    (
                        "Извлечь RawTestProtocolContract из запроса.",
                        "Разрешить неподтверждённые сущности и прочитать "
                        "только зависимости checks.",
                        "Выполнить static preflight и скомпилировать Phase 0–3.",
                    ),
                    start=1,
                )
            ]
            record_coordinator_plan(direct_plan)
            direct_output = {
                "answer": answer,
                "pipeline": "validation_protocol",
                "protocol_status": protocol_trace.get("status"),
            }
            record_upstream_output(direct_output)
            return {
                "operation_skills": list(operation_skills),
                "operation_sql_risk_aspects": list(
                    operation_sql_risk_aspects
                ),
                "operation_pipeline": "validation_protocol",
                "plan": [],
                "next_step": 0,
                "upstream_output": direct_output,
                "final_answer": answer,
                "selected_display_refs": protocol_display_refs,
            }

        if operation_pipeline == "s2t_analysis":
            analysis_display_refs: List[str] = []
            contract_model = bind_required_tool(
                _s2t_analysis_contract_tool_schema(),
                _S2T_ANALYSIS_CONTRACT_TOOL_NAME,
            )
            contract_payload = {"original_task": state["task"]}
            contract_messages: List[BaseMessage] = [
                SystemMessage(content=_S2T_ANALYSIS_CONTRACT_PROMPT),
                HumanMessage(
                    content=json.dumps(contract_payload, ensure_ascii=False)
                ),
            ]
            contract_result = invoke(
                contract_model,
                contract_messages,
                stage="s2t_analysis_contract",
            )
            try:
                contract = _native_s2t_analysis_contract(contract_result)
                _validate_contract_origin(contract, state["task"])
            except CoordinatorResponseError as first_error:
                logger.warning(
                    "Typed validation contract was invalid; requesting one "
                    "LLM repair: %s",
                    first_error,
                )
                contract_result = invoke(
                    contract_model,
                    _repair_messages(
                        contract_messages,
                        contract_result,
                        _S2T_ANALYSIS_CONTRACT_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error),
                    ),
                    stage="s2t_analysis_contract",
                )
                try:
                    contract = _native_s2t_analysis_contract(contract_result)
                    _validate_contract_origin(contract, state["task"])
                except CoordinatorResponseError as second_error:
                    logger.warning(
                        "Typed legacy S2T contract remained invalid; refusing "
                        "an implicit pipeline change: %s",
                        second_error,
                    )
                    raise CoordinatorResponseError(
                        "Невалидный контракт legacy S2T pipeline после repair; "
                        "agentic fallback запрещён."
                    ) from second_error
                else:
                    operation_pipeline = "s2t_analysis"

            if operation_pipeline == "s2t_analysis":
                try:
                    reader_results = read_validation_protocol_inputs(
                        contract,
                        callbacks=callback_list,
                    )
                except ValidationProtocolDataError as exc:
                    answer = (
                        "S2T-анализ не выполнен: подтверждённые данные "
                        f"недостаточны ({exc})."
                    )
                else:
                    analysis_payload = build_s2t_analysis_payload(
                        contract,
                        reader_results=reader_results,
                    )
                    deterministic_items = build_deterministic_analysis_items(
                        contract,
                        reader_results=reader_results,
                    )
                    if analysis_payload["llm_requested_analyses"]:
                        analysis_model = bind_required_tool(
                            _s2t_analysis_tool_schema(),
                            _S2T_ANALYSIS_TOOL_NAME,
                        )
                        analysis_messages: List[BaseMessage] = [
                            SystemMessage(content=_S2T_ANALYSIS_PROMPT),
                            HumanMessage(
                                content=json.dumps(
                                    analysis_payload,
                                    ensure_ascii=False,
                                )
                            ),
                        ]
                        analysis_result = invoke(
                            analysis_model,
                            analysis_messages,
                            stage="s2t_analysis",
                        )
                        try:
                            analysis = _native_s2t_analysis(analysis_result)
                            validate_s2t_analysis_output(contract, analysis)
                        except (
                            CoordinatorResponseError,
                            ValueError,
                        ) as first_error:
                            logger.warning(
                                "S2T analysis violated its contract; requesting "
                                "one LLM repair: %s",
                                first_error,
                            )
                            analysis_result = invoke(
                                analysis_model,
                                _repair_messages(
                                    analysis_messages,
                                    analysis_result,
                                    _S2T_ANALYSIS_REPAIR_PROMPT
                                    + "\nОшибка: "
                                    + str(first_error),
                                ),
                                stage="s2t_analysis",
                            )
                            analysis = _native_s2t_analysis(analysis_result)
                            try:
                                validate_s2t_analysis_output(contract, analysis)
                            except ValueError as second_error:
                                raise CoordinatorResponseError(
                                    str(second_error)
                                ) from second_error
                    else:
                        analysis = S2TAnalysisOutput(analyses=[])
                    analysis = merge_s2t_analysis_output(
                        analysis,
                        deterministic_items,
                    )
                    answer = render_s2t_analysis_answer(contract, analysis)
                    analysis_display_refs = register_worker_display_items(
                        [
                            WorkerDisplayItem(**item)
                            for item in build_s2t_analysis_display_payloads(
                                reader_results
                            )
                        ]
                    )
                    if collected_display_refs is not None:
                        collected_display_refs.extend(analysis_display_refs)
                direct_plan = [
                    {
                        "cycle": state["cycle"],
                        "step": index,
                        "task": task,
                        "operation_skills": list(operation_skills),
                        "sql_risk_aspects": list(
                            operation_sql_risk_aspects
                        ),
                        "pipeline": "s2t_analysis",
                    }
                    for index, task in enumerate(
                        (
                            "Прочитать все S2T-строки каждого target.",
                            "Прочитать target-каталог каждого file/table.",
                            "Проанализировать только подтверждённые S2T и каталог.",
                        ),
                        start=1,
                    )
                ]
                record_coordinator_plan(direct_plan)
                direct_output = {
                    "answer": answer,
                    "pipeline": "s2t_analysis",
                }
                record_upstream_output(direct_output)
                return {
                    "operation_skills": list(operation_skills),
                    "operation_sql_risk_aspects": list(
                        operation_sql_risk_aspects
                    ),
                    "operation_pipeline": "s2t_analysis",
                    "plan": [],
                    "next_step": 0,
                    "upstream_output": direct_output,
                    "final_answer": answer,
                    "selected_display_refs": analysis_display_refs,
                }

        plan_operation_context = load_operation_skills(
            operation_skills,
            stage="plan",
            sql_risk_aspects=operation_sql_risk_aspects,
        )
        scope_evidence_contract = build_sql_risk_scope_contract(
            state["task"],
            operation_sql_risk_aspects,
        )
        scope_plan_context = render_sql_risk_scope_contract(
            scope_evidence_contract,
            stage="plan",
        )
        scope_evidence_attestation: Dict[str, Any] = {}
        if scope_evidence_contract is not None:
            scope_evidence_attestation = {
                "operation_sql_risk_scope_contract": {
                    "scope": scope_evidence_contract.scope.label,
                    "required_evidence": [
                        {
                            "tool_name": requirement.tool_name,
                            "arguments": dict(requirement.arguments),
                        }
                        for requirement in scope_evidence_contract.requirements
                    ],
                }
            }
        if scope_plan_context:
            plan_operation_context = "\n\n".join(
                part
                for part in (plan_operation_context, scope_plan_context)
                if part
            )
        sql_risk_protocol_attestation = _sql_risk_protocol_attestation(
            operation_skills,
            operation_sql_risk_aspects,
        )
        plan_payload: Dict[str, Any] = {
            "original_task": state["task"],
            "context": state["context"],
        }
        if state["upstream_problem"] is not None:
            plan_payload["problem"] = state["upstream_problem"]
        plan_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        _DOWNSTREAM_PLAN_PROMPT,
                        plan_operation_context,
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(
                    plan_payload,
                    ensure_ascii=False,
                )
            ),
        ]
        plan_result = invoke(
            plan_model,
            plan_messages,
            stage="downstream_plan",
        )

        def validate_plan_contract(candidate: WorkerPlan) -> None:
            contract_errors: List[str] = []
            try:
                validate_worker_plan_origin(
                    candidate,
                    state["task"],
                    context=state["context"],
                )
            except PlanOriginError as exc:
                contract_errors.append(str(exc))
            try:
                validate_sql_risk_plan_requirements(
                    candidate,
                    state["task"],
                    sql_risk_aspects=operation_sql_risk_aspects,
                )
            except SqlRiskPlanRequirementError as exc:
                contract_errors.append(str(exc))
            if (
                state["upstream_problem"] is not None
                and "Анализ SQL-рисков" in operation_skills
            ):
                try:
                    validate_sql_risk_reroute_plan(
                        candidate,
                        state["task"],
                        sql_risk_aspects=operation_sql_risk_aspects,
                    )
                except ReroutePlanRequirementError as exc:
                    contract_errors.append(str(exc))
            if contract_errors:
                raise CoordinatorResponseError("; ".join(contract_errors))

        try:
            plan = _native_payload(
                plan_result,
                _PLAN_TOOL_NAME,
                WorkerPlan,
            )
            assert isinstance(plan, WorkerPlan)
            validate_plan_contract(plan)
        except CoordinatorResponseError as first_error:
            logger.warning(
                "Coordinator plan call violated plan schema; requesting one "
                "LLM repair: %s",
                first_error,
            )
            repaired_result = invoke(
                plan_model,
                _repair_messages(
                    plan_messages,
                    plan_result,
                    _DOWNSTREAM_PLAN_REPAIR_PROMPT.replace(
                        "{validation_error}",
                        str(first_error),
                    ),
                ),
                stage="downstream_plan",
            )
            plan = _native_payload(
                repaired_result,
                _PLAN_TOOL_NAME,
                WorkerPlan,
            )
            plan_result = repaired_result
            assert isinstance(plan, WorkerPlan)
            try:
                validate_plan_contract(plan)
            except CoordinatorResponseError as second_error:
                raise CoordinatorResponseError(
                    "Исправленный worker plan нарушает plan contract: "
                    + str(second_error)
                ) from second_error
        assert isinstance(plan, WorkerPlan)

        scope_step_index = _first_scope_step_index(
            plan.steps,
            scope_evidence_contract,
        )
        recorded_plan = [
            {
                "cycle": state["cycle"],
                "step": index,
                **step.model_dump(mode="json", exclude_none=True),
                "operation_skills": list(operation_skills),
                "sql_risk_aspects": list(operation_sql_risk_aspects),
                "pipeline": operation_pipeline,
                **sql_risk_protocol_attestation,
                **(
                    scope_evidence_attestation
                    if index - 1 == scope_step_index
                    else {}
                ),
            }
            for index, step in enumerate(plan.steps, start=1)
        ]
        logger.info(
            "Coordinator planned worker_steps=%s plan=%s",
            len(plan.steps),
            json.dumps(recorded_plan, ensure_ascii=False),
        )
        record_coordinator_plan(recorded_plan)
        return {
            "operation_skills": list(operation_skills),
            "operation_sql_risk_aspects": list(
                operation_sql_risk_aspects
            ),
            "operation_pipeline": operation_pipeline,
            "plan": [step.model_dump() for step in plan.steps],
            "next_step": 0,
        }

    def worker_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        step_index = state["next_step"]
        plan_step = PlanStep.model_validate(state["plan"][step_index])
        planned_task = plan_step.task
        if not planned_task:
            raise CoordinatorResponseError(
                "Coordinator вызвал worker с пустой task из плана."
            )
        worker_task = planned_task
        structured_constraints = plan_step.model_dump(
            mode="json",
            exclude={"task", "dependencies"},
            exclude_none=True,
            exclude_defaults=True,
        )
        if structured_constraints:
            worker_task += (
                "\n\nСтруктурированные ограничения шага:\n"
                + json.dumps(structured_constraints, ensure_ascii=False)
            )
        selected_operation_skills = state.get("operation_skills") or []
        selected_sql_risk_aspects = (
            state.get("operation_sql_risk_aspects") or []
        )
        scope_evidence_contract = build_sql_risk_scope_contract(
            state["task"],
            selected_sql_risk_aspects,
        )
        step_scope_contract: SqlRiskScopeContract | None = None
        if scope_evidence_contract is not None:
            if step_index == _first_scope_step_index(
                state["plan"],
                scope_evidence_contract,
            ):
                step_scope_contract = scope_evidence_contract
        planner_context = load_operation_skills(
            selected_operation_skills,
            stage="planner",
            sql_risk_aspects=selected_sql_risk_aspects,
        )
        observer_context = load_operation_skills(
            selected_operation_skills,
            stage="observer",
            sql_risk_aspects=selected_sql_risk_aspects,
        )
        scope_planner_context = render_sql_risk_scope_contract(
            step_scope_contract,
            stage="planner",
        )
        scope_observer_context = render_sql_risk_scope_contract(
            step_scope_contract,
            stage="observer",
        )
        if scope_planner_context:
            planner_context = "\n\n".join(
                part
                for part in (planner_context, scope_planner_context)
                if part
            )
        if scope_observer_context:
            observer_context = "\n\n".join(
                part
                for part in (observer_context, scope_observer_context)
                if part
            )
        if planner_context:
            worker_task += (
                WORKER_OPERATION_EXECUTION_MARKER + planner_context
            )
        if observer_context:
            worker_task += (
                WORKER_OPERATION_COMPLETENESS_MARKER
                + observer_context
            )
        dependency_steps = (
            None
            if plan_step.dependencies is None
            else set(plan_step.dependencies)
        )
        previous_results = [
            reference
            for run in state["worker_runs"]
            if run["cycle"] == state["cycle"]
            and (
                dependency_steps is None
                or run["step"] in dependency_steps
            )
            for reference in run["outcome"].previous_results
        ]
        if previous_results:
            worker_task += (
                WORKER_PREVIOUS_RESULTS_MARKER
                + "\n"
                + json.dumps(
                    {
                        "previous_results": [
                            item.model_dump(mode="json", exclude_none=True)
                            for item in previous_results
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        logger.info(
            "Coordinator dispatches planned worker step=%s task=%s",
            step_index + 1,
            worker_task[:1000],
        )
        outcome = (
            worker_chat(
                worker_task,
                required_evidence=step_scope_contract.requirements,
            )
            if step_scope_contract is not None
            else worker_chat(worker_task)
        )
        record_worker_outcome(
            cycle=state["cycle"],
            step=step_index + 1,
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            unmet_requirements=list(outcome.unmet_requirements),
            evidence_count=len(outcome.evidence),
            dataset_count=len(outcome.datasets),
        )
        for artifact in outcome.evidence:
            if artifact.display_ref and collected_display_refs is not None:
                collected_display_refs.append(artifact.display_ref)
        run: CoordinatorWorkerRun = {
            "cycle": state["cycle"],
            "step": step_index + 1,
            "outcome": outcome,
        }
        return {
            "worker_runs": [*state["worker_runs"], run],
            "next_step": step_index + 1,
        }

    def validate_upstream_decision(
        message: Any,
        *,
        can_reroute: bool,
    ) -> UpstreamDecision:
        decision = _native_upstream_decision(message)
        if decision.decision == "reroute" and not can_reroute:
            raise CoordinatorResponseError(
                "На последнем цикле data decision должен быть pass."
            )
        return decision

    def validate_upstream_answer(
        message: Any,
        *,
        available_evidence_ids: set[str],
        available_display_refs: Dict[str, str],
        require_used_evidence: bool = False,
    ) -> UpstreamOutput:
        output = _native_upstream_answer(message)
        normalized_used_ids = _normalize_upstream_evidence_id_list(
            output.used_evidence_ids,
            available_evidence_ids,
        )
        normalized_display_ids = _normalize_upstream_evidence_id_list(
            output.display_evidence_ids,
            available_evidence_ids,
        )
        try:
            output = UpstreamOutput.model_validate(
                {
                    "answer": output.answer,
                    "used_evidence_ids": normalized_used_ids,
                    "display_evidence_ids": normalized_display_ids,
                }
            )
        except ValidationError as exc:
            raise CoordinatorResponseError(
                "Upstream coordinator вернул несогласованный выбор "
                "evidence_id после безопасной нормализации."
            ) from exc
        unknown_ids = sorted(
            (
                set(output.used_evidence_ids)
                | set(output.display_evidence_ids)
            )
            - available_evidence_ids
        )
        undisplayable_ids = sorted(
            set(output.display_evidence_ids) - set(available_display_refs)
        )
        if unknown_ids or undisplayable_ids:
            raise CoordinatorResponseError(
                "Upstream coordinator выбрал неизвестные evidence_id: "
                + ", ".join([*unknown_ids, *undisplayable_ids])
            )
        if (
            require_used_evidence
            and available_evidence_ids
            and not output.used_evidence_ids
        ):
            raise CoordinatorResponseError(
                "Data-backed SQL-risk answer обязан сослаться хотя бы на "
                "один доступный used_evidence_id."
            )
        return output

    def upstream_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        selected_operation_skills = state.get("operation_skills") or []
        selected_sql_risk_aspects = (
            state.get("operation_sql_risk_aspects") or []
        )
        scope_evidence_contract = build_sql_risk_scope_contract(
            state["task"],
            selected_sql_risk_aspects,
        )
        decision_context = load_operation_skills(
            selected_operation_skills,
            stage="upstream_decision",
            sql_risk_aspects=selected_sql_risk_aspects,
        )
        analysis_context = load_operation_skills(
            selected_operation_skills,
            stage="upstream",
            sql_risk_aspects=selected_sql_risk_aspects,
        )
        scope_decision_context = render_sql_risk_scope_contract(
            scope_evidence_contract,
            stage="upstream_decision",
        )
        scope_analysis_context = render_sql_risk_scope_contract(
            scope_evidence_contract,
            stage="upstream",
        )
        if scope_decision_context:
            decision_context = "\n\n".join(
                part
                for part in (decision_context, scope_decision_context)
                if part
            )
        if scope_analysis_context:
            analysis_context = "\n\n".join(
                part
                for part in (analysis_context, scope_analysis_context)
                if part
            )

        def scoped_answer(answer: str) -> str:
            return ensure_sql_risk_answer_scope(
                answer,
                state["task"],
                selected_sql_risk_aspects,
            )
        available_evidence_ids: set[str] = set()
        available_display_refs: Dict[str, str] = {}
        accepted_artifacts: List[EvidenceArtifact] = []
        evidence_payload: List[Dict[str, Any]] = []
        worker_outcomes: List[Dict[str, Any]] = []
        for run in state["worker_runs"]:
            outcome_payload = run["outcome"].upstream_payload()
            evidence_payload.extend(outcome_payload["evidence"])
            worker_outcomes.append(
                {
                    "cycle": run["cycle"],
                    "step": run["step"],
                    "status": outcome_payload["status"],
                    "stop_reason": outcome_payload["stop_reason"],
                    "unmet_requirements": outcome_payload[
                        "unmet_requirements"
                    ],
                    "evidence_ids": [
                        item["evidence_id"]
                        for item in outcome_payload["evidence"]
                    ],
                }
            )
            for artifact in run["outcome"].evidence:
                if artifact.evidence_id in available_evidence_ids:
                    raise CoordinatorResponseError(
                        "Workers вернули дублирующий evidence_id: "
                        + artifact.evidence_id
                    )
                available_evidence_ids.add(artifact.evidence_id)
                accepted_artifacts.append(artifact)
                if artifact.display_ref is not None:
                    available_display_refs[
                        artifact.evidence_id
                    ] = artifact.display_ref
        upstream_payload = {
            "original_task": state["task"],
            "worker_outcomes": worker_outcomes,
            "evidence": evidence_payload,
        }
        saved_result_store = get_active_saved_result_store()
        missing_scope_requirements = missing_sql_risk_requirements(
            scope_evidence_contract,
            _scope_evidence_calls(
                accepted_artifacts,
                saved_result_store,
            ),
        )
        if missing_scope_requirements:
            upstream_payload["missing_scope_evidence"] = [
                {
                    "tool_name": requirement.tool_name,
                    "arguments": dict(requirement.arguments),
                }
                for requirement in missing_scope_requirements
            ]
        cardinality_sufficient_evidence_ids: List[str] = []
        if (
            selected_operation_skills == ["Анализ SQL-рисков"]
            and selected_sql_risk_aspects == ["cardinality"]
        ):
            saved_store = get_active_saved_result_store()
            if saved_store is not None:
                cardinality_sufficient_evidence_ids = (
                    complete_cardinality_mapping_evidence_ids(
                        state["task"],
                        accepted_artifacts,
                        saved_store,
                    )
                )
        value_change_facts: List[FieldValueChangeFact] = []
        if "value_changes" in selected_sql_risk_aspects:
            saved_store = get_active_saved_result_store()
            if saved_store is not None:
                value_change_facts = derive_field_value_change_facts(
                    state["task"],
                    accepted_artifacts,
                    saved_store,
                )
            if value_change_facts:
                deterministic_sql_risk = field_value_change_payload(
                    value_change_facts
                )
                upstream_payload["deterministic_sql_risk"] = (
                    deterministic_sql_risk
                )
                record_sql_risk_facts(
                    deterministic_sql_risk,
                    cycle=state["cycle"],
                )
        write_semantics_facts: List[WriteSemanticsFact] = []
        if "write_semantics" in selected_sql_risk_aspects:
            saved_store = get_active_saved_result_store()
            if saved_store is not None:
                write_semantics_facts = derive_write_semantics_facts(
                    state["task"],
                    accepted_artifacts,
                    saved_store,
                )
            if write_semantics_facts:
                deterministic_write_semantics = write_semantics_payload(
                    write_semantics_facts
                )
                upstream_payload["deterministic_write_semantics"] = (
                    deterministic_write_semantics
                )
                record_sql_risk_facts(
                    deterministic_write_semantics,
                    cycle=state["cycle"],
                )
        decision_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        _UPSTREAM_DATA_DECISION_PROMPT,
                        (
                            "Разрешён ещё один полный цикл чтения: при "
                            "нехватке данных верни decision=reroute."
                            if state["cycle"] < COORDINATOR_MAX_CYCLES
                            else (
                                "Это последний цикл: верни decision=pass. "
                                "Возможную нехватку данных кратко укажи в problem."
                            )
                        ),
                        decision_context,
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(
                    upstream_payload,
                    ensure_ascii=False,
                )
            ),
        ]
        evidence_context = (
            "\nДоступные used_evidence_ids (копируй дословно): "
            + json.dumps(
                sorted(available_evidence_ids),
                ensure_ascii=False,
            )
            + "\nДоступные display_evidence_ids: "
            + json.dumps(
                sorted(available_display_refs),
                ensure_ascii=False,
            )
        )

        def invoke_decision(
            messages: Sequence[BaseMessage],
        ) -> tuple[Any, UpstreamDecision]:
            can_reroute = state["cycle"] < COORDINATOR_MAX_CYCLES
            result = invoke(
                upstream_data_decision_model,
                messages,
                stage="upstream",
            )
            try:
                decision = validate_upstream_decision(
                    result,
                    can_reroute=can_reroute,
                )
            except CoordinatorResponseError as first_error:
                logger.warning(
                    "Upstream data decision violated schema; "
                    "requesting one LLM repair: %s",
                    first_error,
                )
                result = invoke(
                    upstream_data_decision_model,
                    _repair_messages(
                        messages,
                        result,
                        _UPSTREAM_DATA_DECISION_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error),
                    ),
                    stage="upstream",
                )
                decision = validate_upstream_decision(
                    result,
                    can_reroute=can_reroute,
                )
            return result, decision

        def invoke_answer(
            messages: Sequence[BaseMessage],
        ) -> tuple[Any, UpstreamOutput]:
            require_used_evidence = bool(
                available_evidence_ids
                and "Анализ SQL-рисков" in selected_operation_skills
            )
            result = invoke(
                upstream_answer_model,
                messages,
                stage="upstream",
            )
            try:
                output = validate_upstream_answer(
                    result,
                    available_evidence_ids=available_evidence_ids,
                    available_display_refs=available_display_refs,
                    require_used_evidence=require_used_evidence,
                )
            except CoordinatorResponseError as first_error:
                logger.warning(
                    "Upstream answer violated schema; requesting one LLM "
                    "repair: %s",
                    first_error,
                )
                result = invoke(
                    upstream_answer_model,
                    _repair_messages(
                        messages,
                        result,
                        _UPSTREAM_ANSWER_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error)
                        + evidence_context,
                    ),
                    stage="upstream",
                )
                output = validate_upstream_answer(
                    result,
                    available_evidence_ids=available_evidence_ids,
                    available_display_refs=available_display_refs,
                    require_used_evidence=require_used_evidence,
                )
            return result, output

        def data_request_update(problem: str) -> Dict[str, Any]:
            if state["cycle"] >= COORDINATOR_MAX_CYCLES:
                raise CoordinatorResponseError(
                    "Последний upstream-цикл не может запросить новые данные."
                )
            logger.info(
                "Upstream requests clean data cycle=%s problem=%s",
                state["cycle"] + 1,
                problem,
            )
            return {
                "cycle": state["cycle"] + 1,
                "plan": [],
                "next_step": 0,
                "worker_runs": [],
                "upstream_problem": problem,
                "upstream_output": None,
                "final_answer": None,
                "selected_display_refs": [],
            }

        if missing_scope_requirements:
            missing_text = "; ".join(
                requirement.tool_name
                + "("
                + json.dumps(
                    requirement.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + ")"
                for requirement in missing_scope_requirements
            )
            if state["cycle"] < COORDINATOR_MAX_CYCLES:
                return data_request_update(
                    "Opt-in SQL-risk scope/evidence contract не закрыт: "
                    + missing_text
                )
            assert scope_evidence_contract is not None
            incomplete_answer = (
                "Не удалось завершить оценку SQL-риска для exact scope "
                f"`{scope_evidence_contract.scope.label}`: не получены "
                "обязательные подтверждения "
                f"{missing_text}. Результат: not assessed."
            )
            evidence = UpstreamOutput(
                answer=incomplete_answer,
                used_evidence_ids=[],
                display_evidence_ids=[],
            )
            upstream_output = evidence.model_dump()
            record_upstream_output(
                {
                    **upstream_output,
                    "answer_source": (
                        "deterministic_scope_evidence_unavailable"
                    ),
                }
            )
            return {
                "upstream_output": upstream_output,
                "final_answer": evidence.answer,
                "selected_display_refs": [],
            }

        terminal_write_semantics_answer = ""
        if (
            selected_operation_skills == ["Анализ SQL-рисков"]
            and selected_sql_risk_aspects == ["write_semantics"]
            and len(write_semantics_facts) == 1
            and is_exclusive_write_semantics_request(state["task"])
        ):
            terminal_write_semantics_answer = (
                render_terminal_write_semantics_negative(
                    write_semantics_facts
                ).strip()
            )

        _, decision = invoke_decision(decision_messages)
        ignored_cardinality_reroute = bool(
            decision.decision == "reroute"
            and cardinality_sufficient_evidence_ids
        )
        if decision.decision == "reroute" and not (
            terminal_write_semantics_answer or ignored_cardinality_reroute
        ):
            return data_request_update(decision.problem)

        if ignored_cardinality_reroute:
            logger.info(
                "Ignoring redundant upstream reroute because a complete "
                "exact directed mapping is sufficient for the conditional "
                "cardinality answer: evidence_ids=%s problem=%s",
                cardinality_sufficient_evidence_ids,
                decision.problem,
            )

        if terminal_write_semantics_answer:
            if decision.decision == "reroute":
                logger.info(
                    "Ignoring redundant upstream reroute because exact "
                    "write-semantics evidence proves terminal negative: %s",
                    decision.problem,
                )
            used_evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for fact in write_semantics_facts
                    for evidence_id in fact.evidence_ids
                    if evidence_id in available_evidence_ids
                )
            )
            evidence = UpstreamOutput(
                answer=scoped_answer(terminal_write_semantics_answer),
                used_evidence_ids=used_evidence_ids,
                display_evidence_ids=[],
            )
            upstream_output = evidence.model_dump()
            record_upstream_output(
                {
                    **upstream_output,
                    "answer_source": "deterministic_write_semantics",
                }
            )
            logger.info(
                "Deterministic write-semantics result: %s",
                json.dumps(upstream_output, ensure_ascii=False)[:8000],
            )
            return {
                "upstream_output": upstream_output,
                "final_answer": evidence.answer,
                "selected_display_refs": [],
            }

        if (
            selected_operation_skills == ["Анализ SQL-рисков"]
            and selected_sql_risk_aspects == ["value_changes"]
            and value_change_facts
            and is_exclusive_value_change_request(state["task"])
        ):
            deterministic_answer = scoped_answer(
                render_field_value_change_answer(value_change_facts).strip()
            )
            if deterministic_answer:
                used_evidence_ids = list(
                    dict.fromkeys(
                        evidence_id
                        for fact in value_change_facts
                        for evidence_id in fact.evidence_ids
                        if evidence_id in available_evidence_ids
                    )
                )
                evidence = UpstreamOutput(
                    answer=deterministic_answer,
                    used_evidence_ids=used_evidence_ids,
                    display_evidence_ids=[],
                )
                upstream_output = evidence.model_dump()
                record_upstream_output(
                    {
                        **upstream_output,
                        "answer_source": "deterministic_value_changes",
                    }
                )
                logger.info(
                    "Deterministic field value-change result: %s",
                    json.dumps(upstream_output, ensure_ascii=False)[:8000],
                )
                return {
                    "upstream_output": upstream_output,
                    "final_answer": evidence.answer,
                    "selected_display_refs": [],
                }

        answer_payload = dict(upstream_payload)
        if decision.problem and not ignored_cardinality_reroute:
            answer_payload["data_problem"] = decision.problem
        answer_messages: List[BaseMessage] = [
            SystemMessage(
                content="\n\n".join(
                    part
                    for part in (
                        _UPSTREAM_ANSWER_PROMPT,
                        _UPSTREAM_ANALYSIS_CONTEXT,
                        analysis_context,
                    )
                    if part
                )
            ),
            HumanMessage(
                content=json.dumps(answer_payload, ensure_ascii=False)
            ),
        ]
        _, evidence = invoke_answer(answer_messages)

        scoped_model_answer = scoped_answer(evidence.answer)
        if scoped_model_answer != evidence.answer:
            evidence = evidence.model_copy(
                update={"answer": scoped_model_answer}
            )
        upstream_output = evidence.model_dump()
        selected_display_refs = [
            available_display_refs[evidence_id]
            for evidence_id in evidence.display_evidence_ids
        ]
        record_upstream_output(
            {**upstream_output, "answer_source": "model"}
        )
        logger.info(
            "Upstream coordinator result: %s",
            json.dumps(upstream_output, ensure_ascii=False)[:8000],
        )
        return {
            "upstream_output": upstream_output,
            "final_answer": evidence.answer,
            "selected_display_refs": selected_display_refs,
        }

    def route_after_worker(
        state: CoordinatorGraphState,
    ) -> Literal["worker", "upstream"]:
        if state["next_step"] < len(state["plan"]):
            return "worker"
        return "upstream"

    def route_after_downstream(
        state: CoordinatorGraphState,
    ) -> Literal["worker", "end"]:
        if str(state.get("final_answer") or "").strip():
            return "end"
        return "worker"

    def route_after_upstream(
        state: CoordinatorGraphState,
    ) -> Literal["downstream_plan", "end"]:
        if str(state.get("final_answer") or "").strip():
            return "end"
        if state.get("upstream_problem") is not None:
            return "downstream_plan"
        raise CoordinatorResponseError(
            "Upstream не вернул ни ответ, ни запрос дополнительных данных."
        )

    graph = StateGraph(CoordinatorGraphState)
    graph.add_node("downstream_plan", downstream_plan_node)
    graph.add_node("worker", worker_node)
    graph.add_node("upstream", upstream_node)
    graph.add_edge(START, "downstream_plan")
    graph.add_conditional_edges(
        "downstream_plan",
        route_after_downstream,
        {
            "worker": "worker",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "worker",
        route_after_worker,
        {
            "worker": "worker",
            "upstream": "upstream",
        },
    )
    graph.add_conditional_edges(
        "upstream",
        route_after_upstream,
        {
            "downstream_plan": "downstream_plan",
            "end": END,
        },
    )
    return graph.compile()


def coordinator_chat(task: str, *, context: str = "") -> CoordinatorAnswer:
    """Send tasks downstream to workers and return verified results upstream."""
    clean_task = str(task or "").strip()
    clean_context = str(context or "").strip()[:COORDINATOR_CONTEXT_MAX_CHARS]
    if not clean_task:
        return CoordinatorAnswer(
            answer="Задача coordinator не должна быть пустой.",
            display_refs=[],
        )

    callback = get_callback_handler()
    callbacks = [callback] if callback is not None else []
    metrics_callback = get_run_metrics_callback()
    if metrics_callback is not None and metrics_callback not in callbacks:
        callbacks.append(metrics_callback)
    collected_display_refs: List[str] = []
    graph = build_coordinator_graph(
        chat_model,
        callbacks=callbacks,
        collected_display_refs=collected_display_refs,
    )
    initial_state: CoordinatorGraphState = {
        "task": clean_task,
        "context": clean_context,
        "operation_skills": None,
        "operation_sql_risk_aspects": None,
        "operation_pipeline": None,
        "cycle": 1,
        "plan": [],
        "next_step": 0,
        "worker_runs": [],
        "upstream_problem": None,
        "upstream_output": None,
        "final_answer": None,
        "selected_display_refs": [],
    }
    config = {
        "recursion_limit": (
            COORDINATOR_MAX_CYCLES * (COORDINATOR_MAX_WORKERS + 2) + 5
        ),
        "run_name": "worker_coordinator",
    }

    with (
        saved_result_store_scope(),
        langfuse_trace_context(
            trace_name="worker_coordinator",
            metadata={
                "max_workers": COORDINATOR_MAX_WORKERS,
                "max_cycles": COORDINATOR_MAX_CYCLES,
            },
            tags=["coordinator", "worker", "experiment"],
        ),
    ):
        try:
            final_state = graph.invoke(initial_state, config=config)
            final_answer = str(final_state.get("final_answer") or "").strip()
            if not final_answer:
                raise CoordinatorResponseError(
                    "Coordinator LangGraph завершился без ответа."
                )
            selected_refs = list(final_state.get("selected_display_refs") or [])
            selected_set = set(selected_refs)
            unselected_refs = [
                ref for ref in collected_display_refs if ref not in selected_set
            ]
            if unselected_refs:
                discard_worker_display_refs(unselected_refs)
            return CoordinatorAnswer(
                answer=final_answer,
                display_refs=selected_refs,
            )
        except Exception:
            if collected_display_refs:
                discard_worker_display_refs(collected_display_refs)
            raise


__all__ = [
    "COORDINATOR_MAX_CYCLES",
    "COORDINATOR_MAX_WORKERS",
    "COORDINATOR_CONTEXT_MAX_CHARS",
    "PlanStep",
    "UpstreamOutput",
    "UpstreamDecision",
    "CoordinatorAnswer",
    "CoordinatorGraphState",
    "CoordinatorResponseError",
    "WorkerPlan",
    "build_coordinator_graph",
    "coordinator_chat",
]
