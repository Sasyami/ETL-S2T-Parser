"""LLM-driven workers with separate upstream and downstream coordination."""

from __future__ import annotations

import json
import logging
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
)

from .agent import chat_model
from .contracts import (
    MAX_PLAN_STEPS,
    PlanStep,
    UpstreamDecision,
    UpstreamOutput,
    WORKER_OPERATION_COMPLETENESS_MARKER,
    WORKER_OPERATION_EXECUTION_MARKER,
    WORKER_PREVIOUS_RESULTS_MARKER,
    WORKER_STABLE_CONTEXT_MARKER,
    WorkerOutcome,
    WorkerPlan,
)
from .chat_graph import WorkerDisplayItem
from .observability import get_callback_handler, langfuse_trace_context
from .run_metrics import (
    get_run_metrics_callback,
    llm_stage,
    record_coordinator_plan,
    record_upstream_output,
)
from .tools.context import (
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
from .tools.saved_results import saved_result_store_scope
from .test_protocol import (
    MAX_PROTOCOL_OBJECTS,
    PROTOCOL_CHECKS,
    TestProtocolContract,
    build_test_protocol_display_payloads,
    compile_test_protocol,
    render_test_protocol_answer,
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
    read_validation_protocol_inputs,
    render_s2t_analysis_answer,
    validate_s2t_analysis_output,
)

logger = logging.getLogger(__name__)

COORDINATOR_MAX_WORKERS = MAX_PLAN_STEPS
COORDINATOR_MAX_CYCLES = 2
COORDINATOR_CONTEXT_MAX_CHARS = 4000
_PLAN_TOOL_NAME = "submit_worker_plan"
_OPERATION_SKILL_TOOL_NAME = "select_operation_skills"
_S2T_ANALYSIS_CONTRACT_TOOL_NAME = "submit_s2t_analysis_contract"
_VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME = "submit_validation_protocol_contract"
_S2T_ANALYSIS_TOOL_NAME = "submit_s2t_analysis"
_UPSTREAM_ANSWER_TOOL_NAME = "submit_upstream_answer"
_UPSTREAM_DATA_DECISION_TOOL_NAME = "submit_upstream_data_decision"
_UPSTREAM_ANALYSIS_CONTEXT = load_upstream_analysis_context()
_DOWNSTREAM_CAPABILITY_CONTEXT = get_downstream_capability_context()
_DOWNSTREAM_TABLE_CONTEXT = get_downstream_table_context()
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
        "s2t_analysis",
        "validation_protocol",
    ] = "agentic"
    skills: List[str]


class CoordinatorWorkerRun(TypedDict):
    cycle: int
    step: int
    outcome: WorkerOutcome


class CoordinatorGraphState(TypedDict):
    task: str
    context: str
    operation_skills: Optional[List[str]]
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

_OPERATION_SKILL_PROMPT = f"""
Ты operation router. Один раз для всей `original_task` выбери исполнительный
`pipeline` и operation-skills. Верни ровно один native call
`{_OPERATION_SKILL_TOOL_NAME}`.

`pipeline="validation_protocol"` выбирай для явной просьбы составить SQL
тест-протокол внешней Greenplum-проверки по заданному файлу и
source→target S2T-загрузкам: row count, уникальность ключа, NULL обязательных
полей или корректность трансформаций. SQL только проектируется и не исполняется.

`pipeline="s2t_analysis"` выбирай для оценки рисков или согласованности тех же
точно заданных S2T-загрузок без SQL-протокола. Во всех остальных случаях выбирай
`pipeline="agentic"`.

Operation-skill — профиль выполнения явно запрошенной операции, а не источник
данных, тип объекта или retrieval-skill. Выбирай профиль только при точном
совпадении операции с его назначением. Несколько профилей допустимы, только если
исходная задача явно содержит несколько таких операций.

`skills=[]` — нормальный вариант по умолчанию. Оставляй массив пустым для
простого чтения, списка либо объяснения одной сохранённой трансформации, если
пользователь не просит сравнение атрибутов, оценку риска строк, разность покрытия
маппинга или проектирование проверки. Само наличие SQL, S2T, пары source→target,
колонок либо слова «трансформация» не является основанием выбрать профиль.

Доступные operation-skills:
{_OPERATION_SKILL_CATALOG_CONTEXT}

Не отвечай на задачу, не планируй чтение и не придумывай новый профиль.
""".strip()

_OPERATION_SKILL_REPAIR_PROMPT = f"""
Предыдущий native call `{_OPERATION_SKILL_TOOL_NAME}` нарушает схему или содержит
имя вне каталога. Верни ровно один исправленный call с полями `pipeline` и
`skills`. Pipeline — `agentic`, `s2t_analysis` либо `validation_protocol`;
массив skills может быть пустым. Используй только дословные имена профилей из
каталога.
""".strip()

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

- Файл копируй как `file_id` либо полное `filename`; идентификаторы копируй
  дословно и не заполняй оба варианта одновременно.
- `loads` содержит отдельный элемент для каждой явно заданной загрузки;
  `sources` — все источники именно этого `target`, `target` — один приёмник.
- `checks` каждого load содержит только явно запрошенные классы: `row_count`,
  `key_uniqueness`, `required_null_rate`, `transformation_correctness`.
- Не придумывай таблицы, колонки, SQL, проверки и scope-поля.
""".strip()

_VALIDATION_PROTOCOL_CONTRACT_REPAIR_PROMPT = f"""
Предыдущий `{_VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME}` нарушает строгую схему.
Верни один исправленный native call: укажи исходный `file_id` либо `filename` и
непустой `loads`; в каждом load нужны непустые точные `sources`, один `target` и
непустой `checks` из разрешённого enum. Копируй только из `original_task`.
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


_DOWNSTREAM_PLAN_PROMPT = f"""
Ты downstream planner. Верни один native call `{_PLAN_TOOL_NAME}` с 1–{COORDINATOR_MAX_WORKERS}
`steps`. Каждая task — чтение прямо необходимых фактов.

Каждый step обязан быть незаменимым: без него нельзя ответить на original_task.
Удали не запрошенные проверки, обогащение и физическую реализацию. Наличие
таблицы в справочнике не требует её чтения.

Сохрани сущность, направление, scope и фильтры. Роль source/target известна,
только если привязана к идентификатору в original_task/context или доказана
S2T-строкой; роль результата не задаёт роль кандидата. Точные идентификаторы
бери только из original_task/context, пиши в обратных кавычках без внешней
пунктуации. Не превращай бизнес-термин в техническое имя или tool.

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
reroute построй полный план по original_task и problem; прошлых результатов нет.

{_DOWNSTREAM_CAPABILITY_CONTEXT}

{_DOWNSTREAM_TABLE_CONTEXT}
""".strip()

_DOWNSTREAM_PLAN_REPAIR_PROMPT = f"""
Предыдущий native call `{_PLAN_TOOL_NAME}` нарушает схему или смысловой контракт.
Верни исправленный native call ровно один раз. Массив `steps` должен содержать
от 1 до {COORDINATOR_MAX_WORKERS} элементов; каждый элемент должен иметь
только одну непустую `task`.
Сохрани запрошенные роли, объекты, фильтры и результаты. Не придумывай
идентификаторы, функции, tools или требования. Используй только реальные таблицы
хранилища из system prompt; неизвестные бизнес-объекты оставляй текстом поиска.
Не добавляй анализ и оформление.

Причина отклонения: {{validation_error}}
""".strip()

_UPSTREAM_DATA_DECISION_PROMPT = f"""
Ты проверяешь достаточность `evidence` для `original_task`. Верни один native call
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
Вход содержит `original_task` и принятые `evidence`. Сам выполни запрошенный
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
                            "s2t_analysis",
                            "validation_protocol",
                        ],
                        "description": (
                            "Общий агентный поток либо S2T-only анализ "
                            "точной загрузки."
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
                    }
                },
                "required": ["pipeline", "skills"],
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
                        "description": "Явный file_id каталога метаданных.",
                    },
                    "filename": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                        "description": "Точное имя загруженного файла.",
                    },
                    "loads": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PROTOCOL_OBJECTS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "sources": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": MAX_PROTOCOL_OBJECTS,
                                    "uniqueItems": True,
                                    "items": {"type": "string"},
                                },
                                "target": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "checks": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": len(PROTOCOL_CHECKS),
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string",
                                        "enum": list(PROTOCOL_CHECKS),
                                    },
                                },
                            },
                            "required": ["sources", "target", "checks"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["loads"],
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


def _native_operation_skills(message: Any) -> List[str]:
    """Backward-compatible helper returning only selected prompt profiles."""
    return _native_operation_route(message).skills


def _native_s2t_analysis_contract(message: Any) -> ValidationProtocolContract:
    contract = _native_payload(
        message,
        _S2T_ANALYSIS_CONTRACT_TOOL_NAME,
        ValidationProtocolContract,
    )
    assert isinstance(contract, ValidationProtocolContract)
    return contract


def _native_validation_protocol_contract(message: Any) -> TestProtocolContract:
    contract = _native_payload(
        message,
        _VALIDATION_PROTOCOL_CONTRACT_TOOL_NAME,
        TestProtocolContract,
    )
    assert isinstance(contract, TestProtocolContract)
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
        operation_pipeline = state.get("operation_pipeline")
        if operation_skills is None:
            operation_payload = {
                "original_task": state["task"],
                "context": state["context"],
            }
            operation_messages: List[BaseMessage] = [
                SystemMessage(content=_OPERATION_SKILL_PROMPT),
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
                        _OPERATION_SKILL_REPAIR_PROMPT
                        + "\nОшибка: "
                        + str(first_error),
                    ),
                    stage="operation_router",
                )
                operation_route = _native_operation_route(operation_result)
            operation_skills = operation_route.skills
            operation_pipeline = operation_route.pipeline
        if operation_pipeline is None:
            operation_pipeline = "agentic"

        if operation_pipeline == "validation_protocol":
            protocol_display_refs: List[str] = []
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
            try:
                protocol_contract = _native_validation_protocol_contract(
                    protocol_result
                )
                _validate_contract_origin(protocol_contract, state["task"])
            except CoordinatorResponseError as first_error:
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
                    protocol_contract = _native_validation_protocol_contract(
                        protocol_result
                    )
                    _validate_contract_origin(protocol_contract, state["task"])
                except CoordinatorResponseError as second_error:
                    logger.warning(
                        "Typed test protocol contract remained invalid; "
                        "falling back to agentic: %s",
                        second_error,
                    )
                    operation_pipeline = "agentic"

            if operation_pipeline == "validation_protocol":
                try:
                    protocol_reader_results = read_validation_protocol_inputs(
                        protocol_contract,
                        callbacks=callback_list,
                    )
                except ValidationProtocolDataError as exc:
                    answer = (
                        "Тест-протокол не сформирован: подтверждённые S2T-данные "
                        f"недостаточны ({exc})."
                    )
                else:
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
                direct_plan = [
                    {
                        "cycle": state["cycle"],
                        "step": index,
                        "task": task,
                        "operation_skills": list(operation_skills),
                        "pipeline": "validation_protocol",
                    }
                    for index, task in enumerate(
                        (
                            "Прочитать S2T и target-каталог каждого target.",
                            "Скомпилировать Greenplum SQL test protocol.",
                        ),
                        start=1,
                    )
                ]
                record_coordinator_plan(direct_plan)
                direct_output = {
                    "answer": answer,
                    "pipeline": "validation_protocol",
                }
                record_upstream_output(direct_output)
                return {
                    "operation_skills": list(operation_skills),
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
                        "Typed validation contract remained invalid; falling "
                        "back to the agentic pipeline: %s",
                        second_error,
                    )
                    operation_pipeline = "agentic"
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
        try:
            plan = _native_payload(
                plan_result,
                _PLAN_TOOL_NAME,
                WorkerPlan,
            )
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
        assert isinstance(plan, WorkerPlan)

        recorded_plan = [
            {
                "cycle": state["cycle"],
                "step": index,
                "task": step.task,
                "operation_skills": list(operation_skills),
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
            "operation_pipeline": operation_pipeline,
            "plan": [step.model_dump() for step in plan.steps],
            "next_step": 0,
        }

    def worker_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        step_index = state["next_step"]
        plan_step = state["plan"][step_index]
        planned_task = str(plan_step["task"] or "").strip()
        if not planned_task:
            raise CoordinatorResponseError(
                "Coordinator вызвал worker с пустой task из плана."
            )
        worker_task = planned_task
        selected_operation_skills = state.get("operation_skills") or []
        planner_context = load_operation_skills(
            selected_operation_skills,
            stage="planner",
        )
        observer_context = load_operation_skills(
            selected_operation_skills,
            stage="observer",
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
        context = state["context"].strip()
        if context:
            worker_task += WORKER_STABLE_CONTEXT_MARKER + context
        previous_results = [
            reference
            for run in state["worker_runs"]
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
        outcome = worker_chat(worker_task)
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
    ) -> UpstreamOutput:
        output = _native_upstream_answer(message)
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
        return output

    def upstream_node(state: CoordinatorGraphState) -> Dict[str, Any]:
        selected_operation_skills = state.get("operation_skills") or []
        decision_context = load_operation_skills(
            selected_operation_skills,
            stage="upstream_decision",
        )
        analysis_context = load_operation_skills(
            selected_operation_skills,
            stage="upstream",
        )
        available_evidence_ids: set[str] = set()
        available_display_refs: Dict[str, str] = {}
        evidence_payload: List[Dict[str, Any]] = []
        for run in state["worker_runs"]:
            evidence_payload.extend(
                run["outcome"].upstream_payload()["evidence"]
            )
            for artifact in run["outcome"].evidence:
                if artifact.evidence_id in available_evidence_ids:
                    raise CoordinatorResponseError(
                        "Workers вернули дублирующий evidence_id: "
                        + artifact.evidence_id
                    )
                available_evidence_ids.add(artifact.evidence_id)
                if artifact.display_ref is not None:
                    available_display_refs[
                        artifact.evidence_id
                    ] = artifact.display_ref
        upstream_payload = {
            "original_task": state["task"],
            "evidence": evidence_payload,
        }
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

        _, decision = invoke_decision(decision_messages)
        if decision.decision == "reroute":
            return data_request_update(decision.problem)

        answer_payload = dict(upstream_payload)
        if decision.problem:
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

        upstream_output = evidence.model_dump()
        selected_display_refs = [
            available_display_refs[evidence_id]
            for evidence_id in evidence.display_evidence_ids
        ]
        record_upstream_output(upstream_output)
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
