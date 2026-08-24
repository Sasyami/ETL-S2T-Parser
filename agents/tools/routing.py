"""LLM routing that selects tools, skills, and data schemas independently."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, ValidationError

from .context import SCHEMA_CATALOG, SchemaName

logger = logging.getLogger(__name__)


SkillName = Literal[
    "S2T-строки",
    "Neo4j",
    "Excel и описания",
    "Сравнение",
    "Объяснение",
]

SKILL_CATALOG: Dict[str, str] = {
    "S2T-строки": "Общие ETL-строки, S2T-маппинги, additional objects, правила и агрегации s2t_transformations.",
    "Neo4j": "Графовый lineage именованных ETL-таблиц и колонок.",
    "Excel и описания": (
        "Файлы, листы, заголовки, ячейки и семантические описания."
    ),
    "Сравнение": (
        "Сопоставление нескольких независимых объектов по одинаковому набору "
        "фактов без выдуманной связи или направления между ними."
    ),
    "Объяснение": (
        "Объяснение правил, выражений и метаданных по подтверждённым фактам "
        "с явным указанием недостающих данных."
    ),
}


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
            "Точные source/target table или field и точные columns сохранённых "
            "S2T-строк; роли задаются отдельными фильтрами, q принимает одну "
            "буквальную подстроку, file_id не применяется."
        ),
        "not_for": (
            "Неполное имя, неизвестная роль значения, агрегация или графовый путь."
        ),
    },
    "search_s2t_transformations": {
        "use_when": (
            "Одна подстрока по всем S2T-полям, когда роль значения неизвестна "
            "или имя таблицы неполное/неквалифицированное."
        ),
        "not_for": (
            "Точные ролевые table/field, несколько условий, точные columns или агрегация."
        ),
    },
    "trace_transformation_path": {
        "use_when": (
            "Многошаговый сохранённый S2T-путь от точного имени с rules, SQL, JOIN/FILTER и "
            "промежуточными таблицами; для неполного имени добавь "
            "search_s2t_transformations."
        ),
        "not_for": (
            "Одна S2T-строка, переданный SQL или сравнение физических данных."
        ),
    },
    "visualize_s2t_table_graph": {
        "use_when": "Явно запрошен глобальный интерактивный граф всех S2T-таблиц.",
        "not_for": "Конкретный SQL, путь, таблица или колонка.",
    },
    "run_sql": {
        "use_when": (
            "Произвольный read-only срез, выражение или агрегация по публичным "
            "таблицам SQLite, не покрытые точным специализированным tool."
        ),
        "not_for": (
            "Точные S2T/каталожные строки, логические ETL-таблицы, выполнение "
            "transformation_rule или запрос к $$-именам."
        ),
    },
    "query_saved_result": {
        "use_when": (
            "Есть точный result_ref предыдущего worker и нужен read-only SQL-срез "
            "сохранённых строк этого результата."
        ),
        "not_for": (
            "Нет result_ref, нужна основная SQLite-база или результат truncated."
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
        "use_when": "Upstream/downstream точной ETL-колонки на заданную глубину.",
        "not_for": "Таблица без колонки, SQL, rules или объяснимый S2T-путь.",
    },
    "trace_neo4j_table_lineage": {
        "use_when": "Непосредственные upstream/downstream соседи точной ETL-таблицы.",
        "not_for": "Путь между таблицами, колонка, SQL или rules.",
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
    skills: List[SkillName]
    schemas: List[SchemaName]


_TOOL_ROUTER_PROMPT = """
Ты router read-only worker. По задаче, недавней истории и каталогам выбери
необходимые tools, skills и schemas. Точные имена и назначение бери только из каталогов.

Полностью покрой все операции с данными. Поля `use_when` и `not_for` являются
контрактом выбора, а не справочным текстом. Если результат одного tool нужен как
обязательный вход другого, выбери оба. Обязательный вход должен быть явно дан в
задаче или истории либо получаться другим выбранным tool. Инструмент обработки
переданного текста или объекта не заменяет инструмент, который сначала должен
получить этот текст или объект из хранилища. Не придумывай входы из названий
сущностей и не считай похожий вид результата достаточным основанием для выбора.
Schemas — это фактическая структура данных и настроенные маппинги, которые
могут понадобиться planner для корректного вызова tools или анализа известных
фактов. Выбирай schema только если её структура или маппинги нужны самой задаче.
Schemas, skills и tools выбирай независимо: наличие tool не обязывает выбирать
schema или skill. Каждый из списков `tools`, `skills` и `schemas` может быть
пустым независимо от остальных. Оставляй `tools=[]`, если новые вызовы данных
не нужны и для ответа достаточно переданных фактов, выбранного skill или schema.

Не добавляй взаимозаменяемые дубли и tools для оформления уже полученных фактов.
Если новые данные и контекст не нужны, оставь все три списка пустыми.

При наличии `reroute_context` учти причину неуспеха. Палитру можно повторить,
если исправить нужно вызов или аргументы; меняй её только при нехватке нужной
возможности.

Не отвечай пользователю и не вызывай tools. Заполни только поля `tools`,
`skills` и `schemas` structured-схемы.
""".strip()

_TOOL_ROUTER_REPAIR_PROMPT = """
Предыдущий structured output отклонён схемой router. Повторно выбери tools,
skills и schemas по исходному payload и заполни только поля `tools`, `skills` и
`schemas` переданной схемы. Не добавляй description, reason, пояснения или
другие поля. Каждый из трёх списков может быть пустым независимо от остальных.

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
    return ToolRoute(
        tools=selected_tools,
        skills=selected_skills,
        schemas=selected_schemas,
    )


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

    payload = {
        "current_query": clean_query,
        "recent_history": _history_payload(history),
        "available_tools": _tool_catalog(available_tools),
        "available_skills": _named_catalog(SKILL_CATALOG),
        "available_schemas": _named_catalog(SCHEMA_CATALOG),
    }
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
            )
        except ToolRoutingError as repair_error:
            logger.warning(
                "Tool-router structured repair rejected: %s",
                repair_error,
            )
            raise


__all__ = [
    "SCHEMA_CATALOG",
    "SKILL_CATALOG",
    "ToolRoute",
    "ToolRoutingError",
    "select_chat_route",
]
