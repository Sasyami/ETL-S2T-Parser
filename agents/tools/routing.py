"""LLM routing that selects tools, skills, and data schemas independently."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, ValidationError

from ..contracts import parse_worker_request
from ..run_metrics import llm_stage
from .context import SCHEMA_CATALOG

logger = logging.getLogger(__name__)


SKILL_CATALOG: Dict[str, str] = {
    "S2T-строки": "Общие ETL-строки, S2T-маппинги, additional objects, правила и агрегации s2t_transformations.",
    "Neo4j": "Графовый lineage именованных ETL-таблиц и колонок.",
    "Excel и описания": (
        "Файлы, листы, заголовки, ячейки и семантические описания."
    ),
    "Сравнение": (
        "Получение одинакового набора исходных фактов отдельно для нескольких "
        "объектов без выдуманной связи или направления между ними."
    ),
    "Объяснение": (
        "Получение точных правил, выражений, ролей и метаданных, необходимых "
        "upstream coordinator для последующего объяснения."
    ),
}


GENERAL_FALLBACK_TOOL_NAMES = (
    "trace_transformation_path",
    "trace_neo4j_table_path",
    "trace_neo4j_lineage",
    "get_s2t_rules_by_ids",
    "list_s2t_table_mapping",
    "read_s2t_source_to_target",
    "read_s2t_by_source_table",
    "read_s2t_by_target_table",
    "read_s2t_mapping",
    "list_s2t_occurrences",
    "list_s2t_field_mapping",
    "list_s2t_source_table",
    "list_s2t_target_table",
    "list_s2t_source_field",
    "list_s2t_target_field",
    "list_s2t_transformations",
    "search_s2t_transformations",
    "get_source_target_column_pair",
    "list_column_metadata",
    "list_source_column_catalog",
    "list_target_column_catalog",
    "list_column_catalog",
    "filter_column_catalog",
    "search_column_catalog",
    "run_sql",
    "read_previous_result",
    "run_cypher",
    "search_excel_values",
    "semantic_search_descriptions",
)

_TOOL_ROUTING_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "show_plan": {
        "use_when": "Нужно явно показать выполненные и следующие шаги многошаговой задачи.",
        "not_for": "Чтение данных, одношаговый вопрос или финальный ответ.",
    },
    "search_excel_values": {
        "use_when": "Буквальная подстрока в сохранённых Excel-ячейках таблицы data.",
        "not_for": "S2T, SQL, каталоги, логические ETL-таблицы или выполнение запросов.",
    },
    "get_excel_row": {
        "use_when": "Известны точные file_id, sheet_name и row_num сохранённой Excel-строки.",
        "not_for": "Поиск строки, S2T или логическая ETL-таблица.",
    },
    "list_additional_objects": {
        "use_when": "Точный список Additional objects по file/name/id/sheet/row с полным SQL.",
        "not_for": "Подстрока, S2T-строки или выполнение SQL объекта.",
    },
    "search_additional_objects": {
        "use_when": "Подстрока в name или SQL Additional objects, опционально внутри file_id.",
        "not_for": "Точные фильтры, выполнение SQL или общий S2T-поиск.",
    },
    "list_column_catalog": {
        "use_when": (
            "Точная source/target table.column: атрибуты; scope обязателен."
        ),
        "not_for": "Атрибутный отбор, подстрока, смысл, S2T, lineage.",
    },
    "filter_column_catalog": {
        "use_when": "Колонки по data_type/primary_key/not_null.",
        "not_for": "Точная table.column, подстрока, смысл, S2T, lineage.",
    },
    "search_column_catalog": {
        "use_when": (
            "Явная буквальная подстрока в name/type/description колонки; "
            "scope и фильтры ограничивают выборку."
        ),
        "not_for": (
            "Смысл/назначение, синонимы, переводы, варианты или точные имена."
        ),
    },
    "semantic_search_descriptions": {
        "use_when": (
            "Смысл/бизнес-смысл/назначение/описание или вероятное соответствие "
            "при неизвестном имени; scope: files/tables/columns, source/target "
            "и фильтры колонок."
        ),
        "not_for": "Подстрока, точное имя, Excel-значения, S2T или lineage.",
    },
    "list_s2t_transformations": {
        "use_when": (
            "Точная S2T-пара source_table.source_field → target_table.target_field "
            "или transformation_rule без обхода; роли и значения известны; "
            "file_id не применяется."
        ),
        "not_for": (
            "Семантические кандидаты без подтверждённой S2T-ролью/таблицей; "
            "их последовательный перебор, неполное имя, агрегация, "
            "происхождение по цепочке."
        ),
    },
    "list_s2t_table_mapping": {
        "use_when": "Полный source_table → target_table mapping.",
        "not_for": "Поля/поиск/путь.",
    },
    "get_s2t_rules_by_ids": {
        "use_when": (
            "Точные transformation_id прочитаны из принятого результата lineage; "
            "нужны соответствующие rules."
        ),
        "not_for": (
            "Числа из task, имён таблиц или текста; без lineage-result ID не "
            "подтверждён."
        ),
    },
    "search_s2t_transformations": {
        "use_when": (
            "Подстрока с неизвестной ролью или неполное имя. Для технических "
            "кандидатов прошлого результата — один batch-вызов со всеми "
            "различающимися именами."
        ),
        "not_for": (
            "Точная полная source→target-пара, несколько условий, columns или агрегация."
        ),
    },
    "trace_transformation_path": {
        "use_when": (
            "Многошаговый S2T-путь от точной пары table+column с rules/SQL. "
            "Для известной target-пары путь строится upstream, для source-пары — "
            "downstream; неизвестное точное имя сначала разрешает search."
        ),
        "not_for": (
            "Одна точная source→target-пара или rule; сама стрелка не означает путь."
        ),
    },
    "visualize_s2t_table_graph": {
        "use_when": "Явно запрошен глобальный интерактивный граф всех S2T-таблиц.",
        "not_for": "Конкретный SQL, путь, таблица или колонка.",
    },
    "run_sql": {
        "use_when": (
            "Read-only агрегация, дословный SELECT, JOIN/UNION/подзапрос/окно "
            "или произвольное выражение."
        ),
        "not_for": (
            "Точные S2T/каталожные строки, логические ETL-таблицы, "
            "transformation_rule, $$-именам."
        ),
    },
    "query_saved_result": {
        "use_when": (
            "Текущая task требует read-only SQL-срез строк табличного результата "
            "прошлого worker; schema и result_ref доступны в description tool."
        ),
        "not_for": (
            "Предыдущий result даёт только значение фильтра для нового чтения, "
            "нужна основная SQLite-база либо сохранённый result truncated."
        ),
    },
    "read_previous_result": {
        "use_when": (
            "Для текущей task недостаточно краткого description принятого "
            "результата прошлого worker и нужен его точный result по result_id."
        ),
        "not_for": (
            "Новое чтение из SQLite/Neo4j/Excel, ответ уже следует из description "
            "либо result_id отсутствует в previous_results."
        ),
    },
    "parse_sql_column_lineage": {
        "use_when": (
            "Полный SQL уже явно передан и нужны expression и source_columns "
            "выходных SELECT-колонок."
        ),
        "not_for": (
            "SQL отсутствует, его нужно прочитать из хранилища, анализируются "
            "JOIN/WHERE/GROUP BY или нужна визуализация."
        ),
    },
    "parse_sql_table_lineage": {
        "use_when": (
            "Полный SQL уже явно передан и нужны только исходные и целевая таблицы."
        ),
        "not_for": (
            "SQL отсутствует, нужен колонковый lineage или анализ условий SQL."
        ),
    },
    "visualize_sql_lineage": {
        "use_when": (
            "Полный SQL уже явно дан и пользователь просит интерактивный lineage-граф."
        ),
        "not_for": (
            "Имя без SQL, получение SQL из хранилища или сохранённый S2T-путь."
        ),
    },
    "run_cypher": {
        "use_when": "Произвольный read-only Neo4j-обход или агрегация без готового graph tool.",
        "not_for": "Готовый lineage/path, S2T, SQL-текст или SQLite.",
    },
    "trace_neo4j_lineage": {
        "use_when": "Upstream/downstream полной точной ссылки ETL-колонки на заданную глубину.",
        "not_for": "Таблица без колонки, SQL, rules или объяснимый S2T-путь.",
    },
    "trace_neo4j_table_lineage": {
        "use_when": "Непосредственные upstream/downstream соседи точной ETL-таблицы.",
        "not_for": (
            "Путь, именованная колонка, SQL или rules; для них используй профильный tool."
        ),
    },
    "trace_neo4j_table_path": {
        "use_when": "Направленный Neo4j-путь между двумя точными ETL-таблицами.",
        "not_for": (
            "Сравнение независимых объектов, одна таблица, колонка, SQL, rules "
            "или неизвестное имя."
        ),
    },
    "list_files": {
        "use_when": "Нужен список всех загруженных Excel-файлов и их file_id.",
        "not_for": "Один точный файл, ETL-таблицы, листы, строки или S2T.",
    },
    "resolve_file": {
        "use_when": "Дано полное имя загруженного файла, но нужен его file_id.",
        "not_for": "Частичное имя, известный file_id или глобальный S2T.",
    },
    "get_file_description": {
        "use_when": "Нужно сохранённое описание файла с известным file_id.",
        "not_for": "Описание ETL-таблицы, semantic search или изменение данных.",
    },
    "list_s2t_table_names": {
        "use_when": "Глобальные множества source/target-таблиц и операции над ними.",
        "not_for": "Связи конкретной пары, counts, правила, путь или текстовый поиск.",
    },
    "summarize_s2t_tables": {
        "use_when": "Групповые counts маппингов, полей, соседей и rules по source/target.",
        "not_for": "Множества имён, точные строки, описания или путь.",
    },
    "summarize_table_descriptions": {
        "use_when": "Нужно каталожное описание логической таблицы с точным table_name.",
        "not_for": "Неизвестное имя, semantic search, S2T-маппинг или файл.",
    },
    "list_sheets": {
        "use_when": "Нужны имена и количество Excel-листов известного file_id.",
        "not_for": "Заголовки, колонки, строки, ETL-таблицы или S2T.",
    },
    "list_file_sheet_headers": {
        "use_when": "Нужны сохранённые результаты определения заголовков листов файла.",
        "not_for": "Только имена листов, значения строк или повторное распознавание.",
    },
    "list_columns": {
        "use_when": "Нужны распознанные физические колонки конкретного Excel-листа.",
        "not_for": "Колонки логической ETL-таблицы, S2T или значения ячеек.",
    },
}

_NARROW_S2T_ROUTING_CONTRACTS: Dict[str, Dict[str, str]] = {
    "read_s2t_source_to_target": {
        "use_when": (
            "Заданы обе точные роли source_table и target_table; tool читает "
            "все строки только этой направленной пары с provenance."
        ),
        "not_for": (
            "Source-only или target-only задача, неизвестная сторона, поле, "
            "подстрока, путь, ID либо file scope."
        ),
    },
    "read_s2t_by_source_table": {
        "use_when": (
            "Source-only задача: все S2T-строки одной точной source_table, "
            "когда target_table не задана."
        ),
        "not_for": (
            "Target-only задача, заданная source→target пара, поле, "
            "подстрока, путь, ID либо file scope."
        ),
    },
    "read_s2t_by_target_table": {
        "use_when": (
            "Target-only задача: все S2T-строки одной точной target_table, "
            "когда source_table не задана."
        ),
        "not_for": (
            "Source-only задача, заданная source→target пара, поле, "
            "подстрока, путь, ID либо file scope."
        ),
    },
    "read_s2t_mapping": {
        "use_when": (
            "Точная направленная source_table → target_table пара; tool всегда "
            "читает полную пару, а заданные task поля выбираются из результата; "
            "строки сохраняют provenance file_id без file-фильтра."
        ),
        "not_for": (
            "Одна таблица без второй, неизвестное/частичное имя или путь; "
            "field-фильтров и ID у tool нет."
        ),
    },
    "list_s2t_occurrences": {
        "use_when": (
            "Все точные occurrences одной table сразу в "
            "source/target-ролях с matched_role и provenance file_id."
        ),
        "not_for": (
            "Направленная source→target-пара, подстрока, путь или агрегация."
        ),
    },
    "get_source_target_column_pair": {
        "use_when": (
            "Атрибуты точной source_table.source_column → "
            "target_table.target_column в одном известном file_id."
        ),
        "not_for": (
            "Одна сторона, неизвестный file_id, поиск, S2T или множество "
            "колонок таблицы."
        ),
    },
    "list_column_metadata": {
        "use_when": (
            "Полная структура одной или нескольких точных таблиц в обеих "
            "catalog-ролях; file_scope — строка file_id либо явная строка all."
        ),
        "not_for": (
            "Сравнение двух точных разноимённых колонок, подстрока, смысл или "
            "S2T; у tool нет фильтров по предполагаемому type/PK/NOT NULL."
        ),
    },
    "list_source_column_catalog": {
        "use_when": (
            "Точные source-колонки одной table в обязательном file_id; "
            "опционально exact column/type/PK/NOT NULL."
        ),
        "not_for": (
            "Target-колонки, неизвестный file_id/table, подстрока, смысл, S2T."
        ),
    },
    "list_target_column_catalog": {
        "use_when": (
            "Точные target-колонки одной table в обязательном file_id; "
            "опционально exact column/type/PK/NOT NULL."
        ),
        "not_for": (
            "Source-колонки, неизвестный file_id/table, подстрока, смысл, S2T."
        ),
    },
    "list_s2t_field_mapping": {
        "use_when": (
            "Одна точная полная source_table.source_field → "
            "target_table.target_field со всеми четырьмя известными ролями."
        ),
        "not_for": "Неполная пара, таблица без поля, подстрока или путь.",
    },
    "list_s2t_source_table": {
        "use_when": "Все S2T-строки одной точной известной source_table.",
        "not_for": "Target-only запрос, отдельное поле, пара таблиц или подстрока.",
    },
    "list_s2t_target_table": {
        "use_when": "Все S2T-строки одной точной известной target_table.",
        "not_for": "Source-only запрос, отдельное поле, пара таблиц или подстрока.",
    },
    "list_s2t_source_field": {
        "use_when": "Все цели точной пары source_table.source_field.",
        "not_for": "Target-поле, таблица без поля, полная пара или подстрока.",
    },
    "list_s2t_target_field": {
        "use_when": "Все источники точной пары target_table.target_field.",
        "not_for": "Source-поле, таблица без поля, полная пара или подстрока.",
    },
}


class ToolRoutingError(RuntimeError):
    """Raised when the tool-router cannot produce a valid selection."""


class ToolRoute(BaseModel):
    """Strict structured schema for independent capability selection."""

    model_config = ConfigDict(extra="forbid")

    tools: List[str]
    skills: List[str]
    schemas: List[str]


_TOOL_ROUTER_PROMPT = """
Ты router read-only worker. По задаче и каталогам выбери необходимую planner
палитру `tools`, `skills`, `schemas`. Используй точные имена из каталогов.

`current_task` — операция worker, `stable_context` — общие ограничения.
`operation_context` — обязательные правила исполнения, не основание для tool.
`previous_results` содержит result_id, description и result_schema.
Внутренний reader уже доступен planner;
выбери внешние tools для операций над строками после чтения result, а не из-за
самого наличия result_id.

Для каждой операции выбери все tools с совпавшим `use_when`; `not_for` — запрет.
При разных трактовках включи релевантные альтернативы: полнота палитры важнее
компактности. Покрой все разные операции и обязательные входы; не добавляй
явно нерелевантные tools.
Сохраняй тип поиска из task: смысл/назначение/описание — semantic; явно данный
буквальный фрагмент — substring. Не заменяй один тип другим.
Обработчик выбирай, только если вход дан или будет получен выбранным tool.
Tool с обязательным opaque ID допустим, только если точный ID уже есть в
принятом результате другого tool. Числа из task, имени или описания не являются
ID. Не придумывай входы.

Если description даёт фильтр нового чтения, выбери источник этого чтения.
`query_saved_result` — только для строк сохранённого dataset с совместимой schema.

Tools, skills и schemas выбирай независимо и только по необходимости; каждый
список может быть пустым. Для ответа по уже данным фактам оставляй `tools=[]`.

При `reroute_context` сохрани последнюю палитру и добавь tool, закрывающий
указанный `gap`; не сокращай и не повторяй палитру без изменения.

Не отвечай и не вызывай tools. Верни только structured-поля `tools`, `skills`,
`schemas`.
""".strip()

_TOOL_ROUTER_REPAIR_PROMPT = """
Исправь только указанное нарушение structured-маршрута по исходному payload.
Если имя отсутствует в каталоге, замени его одним существующим инструментом с
тем же назначением. Не расширяй палитру из-за ошибки и не добавляй альтернативы
«на всякий случай». Верни только `tools`, `skills`, `schemas`; списки могут быть
пустыми.

Ошибка валидации: {validation_error}
""".strip()


def _history_payload(
    history: Optional[Sequence[Mapping[str, str]]],
) -> List[Dict[str, str]]:
    return [
        {
            "role": str(item.get("role") or ""),
            "content": str(item.get("content") or ""),
        }
        for item in (history or [])[-6:]
    ]


def _tool_catalog(tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for tool in tools:
        contract = (
            _TOOL_ROUTING_CONTRACTS.get(tool.name)
            or _NARROW_S2T_ROUTING_CONTRACTS.get(tool.name)
        )
        if contract is None:
            raise ToolRoutingError(
                f"Для зарегистрированного tool отсутствует routing contract: {tool.name}"
            )
        catalog.append({"name": tool.name, **contract})
    return catalog


def _named_catalog(catalog: Mapping[str, str]) -> List[Dict[str, str]]:
    return [
        {"name": name, "description": description}
        for name, description in catalog.items()
    ]


def _validated_route(
    result: Any,
    available_tools: Sequence[BaseTool],
    reroute_context: Optional[Mapping[str, Any]] = None,
) -> ToolRoute:
    try:
        route = (
            result
            if isinstance(result, ToolRoute)
            else ToolRoute.model_validate(result)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ToolRoutingError(
            "Tool-router вернул невалидный structured-маршрут"
        ) from exc

    selected_tools = list(dict.fromkeys(route.tools))
    selected_skills = list(dict.fromkeys(route.skills))
    selected_schemas = list(dict.fromkeys(route.schemas))
    available_names = {tool.name for tool in available_tools}
    unknown = [name for name in selected_tools if name not in available_names]
    if unknown:
        raise ToolRoutingError(
            f"Tool-router выбрал неизвестные tools: {', '.join(unknown)}"
        )
    unknown_skills = [
        name for name in selected_skills if name not in SKILL_CATALOG
    ]
    if unknown_skills:
        raise ToolRoutingError(
            "Tool-router выбрал неизвестные skills: "
            + ", ".join(unknown_skills)
        )
    unknown_schemas = [
        name for name in selected_schemas if name not in SCHEMA_CATALOG
    ]
    if unknown_schemas:
        raise ToolRoutingError(
            "Tool-router выбрал неизвестные schemas: "
            + ", ".join(unknown_schemas)
        )
    if reroute_context is not None:
        previous_palettes = list(
            reroute_context.get("previous_tool_palettes") or []
        )
        if not previous_palettes:
            raise ToolRoutingError(
                "Tool-router получил reroute без предыдущей палитры"
            )
        previous_tools = {
            str(name).strip()
            for name in previous_palettes[-1]
            if str(name).strip()
        }
        selected_set = set(selected_tools)
        removed = sorted(previous_tools - selected_set)
        if removed:
            raise ToolRoutingError(
                "Tool-router при reroute удалил tools прошлой палитры: "
                + ", ".join(removed)
            )
        if not selected_set - previous_tools:
            raise ToolRoutingError(
                "Tool-router при reroute не добавил новый tool для gap"
            )
    return ToolRoute(
        tools=selected_tools,
        skills=selected_skills,
        schemas=selected_schemas,
    )


def _general_fallback_route(
    available_tools: Sequence[BaseTool],
    reroute_context: Optional[Mapping[str, Any]] = None,
) -> ToolRoute:
    """Return a bounded read-only palette after two invalid router outputs."""
    available_names = {tool.name for tool in available_tools}
    selected_tools: List[str] = []
    if reroute_context is not None:
        previous_palettes = list(
            reroute_context.get("previous_tool_palettes") or []
        )
        if previous_palettes:
            selected_tools.extend(
                str(name).strip()
                for name in previous_palettes[-1]
                if str(name).strip() in available_names
            )
    selected_tools.extend(
        name
        for name in GENERAL_FALLBACK_TOOL_NAMES
        if name in available_names
    )
    selected_tools = list(dict.fromkeys(selected_tools))
    if not selected_tools:
        raise ToolRoutingError(
            "Tool-router не выбрал маршрут, а общие fallback tools недоступны"
        )
    return ToolRoute(tools=selected_tools, skills=[], schemas=[])


def select_chat_route(
    user_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    model: Any,
    available_tools: Sequence[BaseTool],
    callbacks: Optional[Sequence[Any]] = None,
    reroute_context: Optional[Mapping[str, Any]] = None,
) -> ToolRoute:
    """Select exact tools, skills, and schemas via structured output."""
    clean_query = str(user_query or "").strip()
    if not clean_query:
        raise ToolRoutingError("Tool-router получил пустой запрос")
    if not available_tools:
        raise ToolRoutingError("Tool-router не получил каталог tools")

    request_parts = parse_worker_request(clean_query)
    if not request_parts.current_task:
        raise ToolRoutingError("Tool-router получил пустую текущую task")

    payload = {
        "current_task": request_parts.current_task,
        "recent_history": _history_payload(history),
        "available_tools": _tool_catalog(available_tools),
        "available_skills": _named_catalog(SKILL_CATALOG),
        "available_schemas": _named_catalog(SCHEMA_CATALOG),
    }
    if request_parts.stable_context:
        payload["stable_context"] = request_parts.stable_context
    if request_parts.operation_execution_context:
        payload["operation_context"] = (
            request_parts.operation_execution_context
        )
    if request_parts.previous_results is not None:
        payload["previous_results"] = [
            item.model_dump(mode="json", exclude_none=True)
            for item in request_parts.previous_results
        ]
    if reroute_context:
        payload["reroute_context"] = dict(reroute_context)
    messages = [
        SystemMessage(content=_TOOL_ROUTER_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    config = {"callbacks": list(callbacks)} if callbacks else None

    with_structured_output = getattr(model, "with_structured_output", None)
    if not callable(with_structured_output):
        raise ToolRoutingError(
            "LLM tool-router не поддерживает structured output"
        )
    try:
        structured_model = with_structured_output(
            ToolRoute,
            method="function_calling",
        )
    except TypeError:
        # Compatibility with wrappers that expose the LangChain method without
        # the optional method keyword.
        structured_model = with_structured_output(ToolRoute)
    except Exception as exc:
        raise ToolRoutingError(
            f"Ошибка настройки structured output tool-router: {type(exc).__name__}"
        ) from exc

    def invoke_router(call_messages: Sequence[BaseMessage]) -> Any:
        try:
            with llm_stage("router"):
                return (
                    structured_model.invoke(call_messages, config=config)
                    if config is not None
                    else structured_model.invoke(call_messages)
                )
        except Exception as exc:
            raise ToolRoutingError(
                f"Ошибка LLM tool-router: {type(exc).__name__}"
            ) from exc

    try:
        return _validated_route(
            invoke_router(messages),
            available_tools,
            reroute_context,
        )
    except ToolRoutingError as first_error:
        logger.warning(
            "Tool-router structured output rejected; requesting one LLM "
            "repair: error=%s",
            first_error,
        )
        repair_messages: List[BaseMessage] = [
            *messages,
            HumanMessage(
                content=_TOOL_ROUTER_REPAIR_PROMPT.replace(
                    "{validation_error}", str(first_error)
                )
            ),
        ]
        try:
            return _validated_route(
                invoke_router(repair_messages),
                available_tools,
                reroute_context,
            )
        except ToolRoutingError as repair_error:
            logger.warning(
                "Tool-router structured repair rejected: %s",
                repair_error,
            )
            fallback_route = _general_fallback_route(
                available_tools,
                reroute_context,
            )
            logger.warning(
                "Tool-router uses general read-only fallback: tools=%s",
                fallback_route.tools,
            )
            return fallback_route


__all__ = [
    "GENERAL_FALLBACK_TOOL_NAMES",
    "SCHEMA_CATALOG",
    "SKILL_CATALOG",
    "ToolRoute",
    "ToolRoutingError",
    "select_chat_route",
]
