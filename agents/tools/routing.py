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
    "run_sql",
    "read_previous_result",
    "run_cypher",
    "search_excel_values",
    "semantic_search_descriptions",
    "list_s2t_transformations",
    "search_s2t_transformations",
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
            "Точный структурный срез source/target-каталога колонок по "
            "file/table/column/type/PK/not-null; scope обязателен, разные роли "
            "и пары запрашиваются отдельно."
        ),
        "not_for": "Подстрока, смысл описания, S2T-маппинг или lineage.",
    },
    "search_column_catalog": {
        "use_when": (
            "Подстрока в имени, типе или описании колонок внутри scope и "
            "точных структурных фильтров."
        ),
        "not_for": "Смысловая близость или точные table/column.",
    },
    "semantic_search_descriptions": {
        "use_when": (
            "Смысловой поиск объекта с неизвестным точным именем по descriptions; "
            "scope ограничивает files/tables/columns и source/target, для колонок "
            "доступны структурные фильтры подвыборки."
        ),
        "not_for": "Точное имя, значения Excel-ячеек, S2T-маппинг или lineage.",
    },
    "list_s2t_transformations": {
        "use_when": (
            "Прямая точная S2T-пара source_table.source_field → "
            "target_table.target_field или transformation_rule без обхода; "
            "роли — отдельные фильтры, file_id не применяется."
        ),
        "not_for": (
            "Неполное имя/роль, агрегация либо происхождение поля по цепочке rules/SQL."
        ),
    },
    "list_s2t_table_mapping": {
        "use_when": "Полный source_table → target_table mapping.",
        "not_for": "Поля/поиск/путь.",
    },
    "get_s2t_rules_by_ids": {
        "use_when": (
            "После lineage известны transformation_id и нужны их точные "
            "transformation_rule из SQLite."
        ),
        "not_for": "Поиск по имени, построение lineage или свободный SQL.",
    },
    "search_s2t_transformations": {
        "use_when": (
            "Одна подстрока по всем S2T-полям, когда роль значения неизвестна "
            "или имя таблицы неполное/неквалифицированное."
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
палитру `tools`, `skills` и `schemas`. Используй только точные имена из каталогов.

`current_task` — единственная операция worker. `stable_context` задаёт только
общие ограничения. `previous_results` содержит лишь непрозрачные result_id и
краткие descriptions принятых результатов прошлых workers. Не выбирай tool
только потому, что result_id упомянут в этом списке.

Для каждой отдельной операции с данными выбери tool, чей `use_when` совпадает.
`not_for` — жёсткий запрет для указанной операции. Не добавляй взаимозаменяемые
tools «на всякий случай», но покрой все разные операции и обязательные входы.
Обработчик текста или объекта выбирай только если вход уже дан либо будет получен
другим выбранным tool. Не придумывай входы.

Различай источник новой операции и входной аргумент: если description уже даёт
значение для фильтра нового чтения, выбери tool источника нового чтения. Выбирай
`read_previous_result`, только когда для task нужен точный прошлый result, а
description недостаточно. `query_saved_result` подходит лишь для операции над
строками сохранённого dataset с совместимой schema.

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
        contract = _TOOL_ROUTING_CONTRACTS.get(tool.name)
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
    if request_parts.previous_results is not None:
        payload["previous_results"] = [
            item.model_dump(mode="json")
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
