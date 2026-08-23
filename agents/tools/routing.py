"""LLM routing that selects tools, skills, and data schemas independently."""

from __future__ import annotations

import json
import logging
import re
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
]

SKILL_CATALOG: Dict[str, str] = {
    "S2T-строки": "Общие ETL-строки, S2T-маппинги, additional objects, правила и агрегации s2t_transformations.",
    "Neo4j": "Графовый lineage именованных ETL-таблиц и колонок.",
    "Excel и описания": (
        "Файлы, листы, заголовки, ячейки и семантические описания."
    ),
}


_TOOL_ROUTING_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "show_plan": {
        "use_when": "Нужно явно показать уже выполненные и следующие шаги сложной задачи.",
        "not_for": "Одношаговый вопрос, чтение данных или финальный ответ.",
        "fallback_only": False,
    },
    "search_excel_values": {
        "use_when": (
            "Буквальная подстрока в сохранённых значениях исходных "
            "Excel-ячеек таблицы data."
        ),
        "not_for": (
            "S2T, SQL, каталоги, логические ETL-таблицы и внешние БД; не "
            "выполняет запросы."
        ),
        "fallback_only": False,
    },
    "get_excel_row": {
        "use_when": "Известны точные file_id, sheet_name и row_num сохранённой Excel-строки.",
        "not_for": "Поиск строки, S2T или логическая ETL-таблица.",
        "fallback_only": False,
    },
    "list_column_catalog": {
        "use_when": (
            "Точный структурный срез каталога колонок: "
            "file/table/column/type/PK/not-null/source. Разные source/target "
            "пары вызывай отдельно; последующая агрегация не заменяет tool. "
            "Явный каталог обязательно покрывай. scope обязателен: "
            "source_columns/target_columns — роль; all_tables — обе/неизвестная."
        ),
        "not_for": "Фрагмент имени, смысл описания, S2T-маппинг или lineage.",
        "fallback_only": False,
    },
    "search_column_catalog": {
        "use_when": (
            "Нужен поиск подстроки в имени, типе или описании с точными фильтрами. "
            "scope обязателен; all_tables — только обе стороны или неизвестная роль."
        ),
        "not_for": "Смысловая близость или уже известные точные table/column.",
        "fallback_only": False,
    },
    "semantic_search_descriptions": {
        "use_when": (
            "Смысловой поиск по неизвестному точному имени. scope: files — "
            "только файлы; tables — все логические таблицы; source_tables — "
            "только исходные таблицы; target_tables — только целевые таблицы; "
            "columns — все колонки; source_columns — только исходные колонки; "
            "target_columns — только целевые колонки; all — только если домен "
            "неизвестен. Для колонковых scope допустимы точные фильтры "
            "подвыборки до cosine-ранжирования."
        ),
        "not_for": (
            "Точное имя объекта, поиск значений ячеек, S2T-маппинги или lineage."
        ),
        "fallback_only": False,
    },
    "list_s2t_transformations": {
        "use_when": (
            "Известны точные роли source/target table или field в сохранённом "
            "S2T либо нужны точные columns его строк. Точный маппинг задавай "
            "отдельными source_table, source_field, target_table, target_field. "
            "q принимает только одну буквальную подстроку; file_id не применяется."
        ),
        "not_for": (
            "Неполное или неквалифицированное имя; поиск значения с "
            "неизвестной ролью; агрегации."
        ),
        "fallback_only": False,
    },
    "search_s2t_transformations": {
        "use_when": (
            "Нужен поиск одной подстроки по S2T: роль значения неизвестна "
            "либо таблица названа неполным или неквалифицированным именем, "
            "даже если её роль source/target известна."
        ),
        "not_for": (
            "Точные source/target table или field с известными ролями; "
            "составная строка из нескольких точных условий; точные "
            "возвращаемые columns; агрегации."
        ),
        "fallback_only": False,
    },
    "trace_transformation_path": {
        "use_when": (
            "Нужен многошаговый сохранённый S2T-путь от точного полного имени: "
            "правила, SQL, JOIN/FILTER и промежуточные таблицы. Tool сам "
            "возвращает полный путь и подтверждение Neo4j; не добавляй отдельные "
            "list/Neo4j tools для повторного получения тех же фактов. Если имя "
            "дано фрагментом или без квалификатора, выбирай одновременно "
            "search_s2t_transformations для разрешения точного имени."
        ),
        "not_for": (
            "Сравнение физических записей; неизвестное точное имя без "
            "search_s2t_transformations; одна прямая S2T-строка."
        ),
        "fallback_only": False,
    },
    "visualize_s2t_table_graph": {
        "use_when": "Пользователь просит глобальный интерактивный граф всех S2T-таблиц.",
        "not_for": "Конкретный SQL, один путь, таблица или колонка; file_id не применяется.",
        "fallback_only": False,
    },
    "run_sql": {
        "use_when": (
            "Read-only срез, выражение или агрегация по публичным таблицам SQLite. "
            "Сохранённый SQL можно получить как значение, но нельзя выполнять."
        ),
        "not_for": (
            "Точные S2T-строки; логические ETL-таблицы как физические; "
            "выполнение transformation_rule; запрос к $$-именам; точный "
            "каталог колонок."
        ),
        "fallback_only": True,
    },
    "query_saved_result": {
        "use_when": (
            "Task содержит точный result_ref предыдущего worker и требует "
            "новый read-only SQL-срез, фильтр, сортировку или агрегацию именно "
            "по сохранённым строкам этого результата."
        ),
        "not_for": (
            "Нет result_ref; нужно заново читать основную SQLite-базу; "
            "сохранённый результат помечен truncated=true, а вывод требуется "
            "по полному исходному набору."
        ),
        "fallback_only": False,
    },
    "parse_sql_column_lineage": {
        "use_when": (
            "Полный SQL уже явно передан в task, истории или подтверждённом "
            "результате tool, и нужно происхождение выходных SELECT-колонок: "
            "их expression и source_columns."
        ),
        "not_for": (
            "Поиск или объяснение JOIN/ON, WHERE, GROUP BY, HAVING, ORDER BY, "
            "ролей алиасов и других произвольных частей SQL; table.column без "
            "SQL; чтение сохранённого SQL; интерактивная визуализация."
        ),
        "fallback_only": False,
    },
    "parse_sql_table_lineage": {
        "use_when": (
            "Полный SQL уже явно передан в task, истории или подтверждённом "
            "результате tool и нужны только исходные и целевая таблицы."
        ),
        "not_for": (
            "Получение сохранённого SQL, поиск additional objects, проверка "
            "WHERE/JOIN/GROUP BY/DISTINCT или имя таблицы без SQL-текста."
        ),
        "fallback_only": False,
    },
    "visualize_sql_lineage": {
        "use_when": (
            "Полный SQL-текст уже явно дан в task, истории или результате "
            "другого выбранного tool, и пользователь просит его интерактивный "
            "lineage-граф."
        ),
        "not_for": (
            "Имя таблицы или колонки без SQL; сохранённый S2T-путь или "
            "transformation; получение SQL из хранилища."
        ),
        "fallback_only": False,
    },
    "run_cypher": {
        "use_when": "Нестандартный read-only графовый обход, условия или агрегация в Neo4j.",
        "not_for": "Готовый lineage/path tool, S2T-строки, SQL-текст или SQLite.",
        "fallback_only": True,
    },
    "trace_neo4j_lineage": {
        "use_when": "Upstream/downstream точной именованной ETL-колонки на нужную глубину.",
        "not_for": "Таблица без колонки, SQL-текст, правила или объяснимый S2T-путь.",
        "fallback_only": False,
    },
    "trace_neo4j_table_lineage": {
        "use_when": "Непосредственные upstream/downstream соседи точной ETL-таблицы.",
        "not_for": "Длинный путь, колонка, SQL-текст или правила трансформации.",
        "fallback_only": False,
    },
    "trace_neo4j_table_path": {
        "use_when": "Направленный путь между двумя точными известными ETL-таблицами.",
        "not_for": "Одна таблица, колонка, SQL-текст или неизвестные имена.",
        "fallback_only": False,
    },
    "list_files": {
        "use_when": "Нужен каталог всех загруженных Excel-файлов и их file_id.",
        "not_for": "Один точный файл, ETL-таблицы, листы, строки или S2T.",
        "fallback_only": False,
    },
    "resolve_file": {
        "use_when": "Дано точное имя загруженного файла, но нужен его file_id.",
        "not_for": "Частичное имя, уже известный file_id или глобальный S2T.",
        "fallback_only": False,
    },
    "get_file_description": {
        "use_when": "Нужно сохранённое описание одного файла с известным file_id.",
        "not_for": "Описание ETL-таблицы, semantic search или изменение описания.",
        "fallback_only": False,
    },
    "list_s2t_table_names": {
        "use_when": "Глобальные множества source/target имён и union/intersection/difference.",
        "not_for": "Связи конкретной пары, counts, правила, путь или текстовый поиск.",
        "fallback_only": False,
    },
    "summarize_s2t_tables": {
        "use_when": "Групповые counts маппингов, полей, соседей и правил по source/target.",
        "not_for": "Множества имён, точные строки, описания или многошаговый путь.",
        "fallback_only": False,
    },
    "summarize_table_descriptions": {
        "use_when": "Нужно описание одной логической таблицы с точным table_name.",
        "not_for": "Неизвестное имя, смысловой поиск, S2T-маппинг или файл.",
        "fallback_only": False,
    },
    "list_sheets": {
        "use_when": "Нужны имена и точное число Excel-листов известного file_id.",
        "not_for": "Заголовки, колонки, строки, ETL-таблицы или S2T.",
        "fallback_only": False,
    },
    "list_file_sheet_headers": {
        "use_when": "Нужны сохранённые метаданные определения заголовков листов файла.",
        "not_for": "Только имена листов, значения строк или повторное распознавание.",
        "fallback_only": False,
    },
    "list_columns": {
        "use_when": "Нужны распознанные физические колонки одного Excel-листа.",
        "not_for": "Колонки логической ETL-таблицы, S2T или поиск значений.",
        "fallback_only": False,
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

Полностью покрой все операции с данными. Поля `use_when`, `not_for` и
`fallback_only` являются контрактом выбора, а не справочным текстом. Tool с
`fallback_only=true` выбирай только когда ни один готовый специализированный
tool не покрывает операцию. Если результат одного tool нужен как
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
    *,
    user_query: str = "",
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
    normalized_query = str(user_query or "").casefold()
    catalog_references = list(
        re.finditer(r"\b(?:source_columns|target_columns)\b", normalized_query)
    )
    positive_catalog_reference = any(
        not re.search(
            r"(?:не\s+(?:читай|используй|запрашивай)|do\s+not\s+(?:read|use|query))\s*$",
            normalized_query[max(0, match.start() - 50):match.start()],
        )
        for match in catalog_references
    )
    semantic_errors: List[str] = []
    if (
        positive_catalog_reference
        and "list_column_catalog" in available_names
        and "list_column_catalog" not in selected_tools
    ):
        semantic_errors.append(
            "Task явно требует публичный source_columns/target_columns; "
            "маршрут обязан включать list_column_catalog."
        )
    exact_s2t_role = re.search(
        r"\b(?:source|target)_(?:table|field)\s*(?:=|:)\s*"
        r"[A-Za-z0-9_$][A-Za-z0-9_.$]*",
        str(user_query or ""),
        flags=re.IGNORECASE,
    )
    if exact_s2t_role and "list_s2t_transformations" in available_names:
        if "list_s2t_transformations" not in selected_tools:
            semantic_errors.append(
                "Task задаёт точный ролевой S2T-фильтр; маршрут обязан "
                "включать list_s2t_transformations."
            )
        if "search_s2t_transformations" in selected_tools:
            semantic_errors.append(
                "Точный ролевой S2T-фильтр нельзя заменять подстрочным "
                "search_s2t_transformations."
            )
    if semantic_errors:
        raise ToolRoutingError(" ".join(semantic_errors))
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
            user_query=clean_query,
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
                user_query=clean_query,
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
