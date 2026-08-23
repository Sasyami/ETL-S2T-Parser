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
    "SQLite SQL",
    "SQL lineage",
    "S2T-строки",
    "Путь S2T-преобразования",
    "Neo4j",
    "Excel и описания",
]

SKILL_CATALOG: Dict[str, str] = {
    "SQLite SQL": "Выполнение read-only SQL по сохранённым данным SQLite.",
    "SQL lineage": "Статический разбор и визуализация зависимостей SQL-текста.",
    "S2T-строки": "Общие ETL-строки, S2T-маппинги, additional objects, правила и агрегации s2t_transformations.",
    "Путь S2T-преобразования": (
        "Объяснение и визуализация сохранённых многошаговых source/target "
        "путей: правила, SQL, additional objects и подтверждение Neo4j."
    ),
    "Neo4j": "Графовый lineage именованных ETL-таблиц и колонок.",
    "Excel и описания": (
        "Файлы, листы, заголовки, ячейки и семантические описания."
    ),
}


_TOOL_ROUTING_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "list_column_catalog": {
        "use_when": (
            "Известны роль и точные фильтры каталога колонок либо нужна "
            "структурная подвыборка по file/table/column/type/PK/not-null/source "
            "с выбранными возвращаемыми полями."
        ),
        "not_for": "Фрагмент имени, смысл описания, S2T-маппинг или lineage.",
        "fallback_only": False,
    },
    "search_column_catalog": {
        "use_when": (
            "Нужен поиск подстроки по имени, типу или описанию колонки; scope и "
            "точные фильтры могут предварительно ограничить подвыборку."
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
            "Известно точное полное source_table/target_table в сохранённом "
            "S2T либо нужны точные columns его строк."
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
            "Точное полное source_table/target_table с известной ролью; "
            "точные возвращаемые columns; агрегации."
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
    "run_sql": {
        "use_when": (
            "Нужен read-only SQLite-срез, выражение или агрегация, которых нет "
            "в готовом специализированном tool."
        ),
        "not_for": (
            "Точные S2T-строки по source_table/target_table/columns; логические "
            "ETL-таблицы как физические SQLite-таблицы."
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


def _compact_tool_description(tool: BaseTool) -> str:
    text = " ".join(str(tool.description or "").split())
    sentence_ends = list(re.finditer(r"[.!?](?:\s|$)", text))
    if sentence_ends:
        selected_end = sentence_ends[min(1, len(sentence_ends) - 1)].end()
        text = text[:selected_end].strip()
    if len(text) > 420:
        text = text[:419].rstrip() + "…"
    return text


def _tool_catalog(tools: Sequence[BaseTool]) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for tool in tools:
        contract = _TOOL_ROUTING_CONTRACTS.get(tool.name)
        if contract is None:
            contract = {
                "use_when": _compact_tool_description(tool),
                "not_for": "",
                "fallback_only": False,
            }
        catalog.append({"name": tool.name, **contract})
    return catalog


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
        "available_skills": SKILL_CATALOG,
        "available_schemas": SCHEMA_CATALOG,
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
        return _validated_route(invoke_router(messages), available_tools)
    except ToolRoutingError as first_error:
        logger.warning(
            "Tool-router structured output rejected; requesting one LLM repair: error=%s",
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
            logger.warning("Tool-router structured repair rejected: %s", repair_error)
            raise


__all__ = [
    "SCHEMA_CATALOG",
    "SKILL_CATALOG",
    "ToolRoute",
    "ToolRoutingError",
    "select_chat_route",
]
