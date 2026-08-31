"""Explicit read-only and mutating tool registries."""

import os
from typing import Dict, Iterable, Tuple

from langchain_core.tools import BaseTool

from .additional_objects import list_additional_objects, search_additional_objects
from .files import (
    get_file_description,
    list_files,
    resolve_file,
    update_file_description,
    update_table_info_from_user_query,
)
from .columns import (
    filter_column_catalog,
    list_column_catalog,
    search_column_catalog,
)
from .planning import show_plan
from .data import get_excel_row, search_excel_values, semantic_search_descriptions
from .neo4j import (
    run_cypher,
    trace_neo4j_lineage,
    trace_neo4j_table_lineage,
    trace_neo4j_table_path,
)
from .s2t import (
    get_s2t_rules_by_ids,
    list_s2t_source_field,
    list_s2t_source_table,
    list_s2t_table_mapping,
    list_s2t_table_names,
    list_s2t_target_field,
    list_s2t_target_table,
    list_s2t_transformations,
    search_s2t_transformations,
    summarize_s2t_tables,
    summarize_table_descriptions,
)
from .s2t_graph import visualize_s2t_table_graph
from .sheets import (
    list_columns,
    list_file_sheet_headers,
    # list_sheet_group_classifications,  # Только внутренняя диагностика extraction.
    list_sheets,
)
from .sql import run_sql
from .saved_results import query_saved_result
from .sql_lineage import (
    parse_sql_column_lineage,
    parse_sql_table_lineage,
    visualize_sql_lineage,
)
from .transformation_paths import trace_transformation_path

S2T_NARROW_TOOLS_EXPERIMENT_ENV = "S2T_NARROW_TOOLS_EXPERIMENT"

READ_ONLY_TOOLS: Tuple[BaseTool, ...] = (
    show_plan,
    search_excel_values,
    get_excel_row,
    semantic_search_descriptions,
    list_additional_objects,
    search_additional_objects,
    list_column_catalog,
    filter_column_catalog,
    search_column_catalog,
    visualize_s2t_table_graph,
    trace_transformation_path,
    parse_sql_column_lineage,
    parse_sql_table_lineage,
    visualize_sql_lineage,
    run_sql,
    query_saved_result,
    run_cypher,
    trace_neo4j_lineage,
    trace_neo4j_table_lineage,
    trace_neo4j_table_path,
    list_files,
    resolve_file,
    get_file_description,
    get_s2t_rules_by_ids,
    list_s2t_table_mapping,
    list_s2t_table_names,
    list_s2t_transformations,
    search_s2t_transformations,
    summarize_s2t_tables,
    summarize_table_descriptions,
    list_sheets,
    list_file_sheet_headers,
    # list_sheet_group_classifications,
    list_columns,
)

WRITE_TOOLS: Tuple[BaseTool, ...] = (
    update_file_description,
    update_table_info_from_user_query,
)

_NARROW_S2T_TOOLS: Tuple[BaseTool, ...] = (
    list_s2t_source_table,
    list_s2t_target_table,
    list_s2t_source_field,
    list_s2t_target_field,
)
_EXPERIMENT_READ_ONLY_TOOLS: Tuple[BaseTool, ...] = tuple(
    tool
    for tool in READ_ONLY_TOOLS
    if tool.name != "list_s2t_transformations"
) + _NARROW_S2T_TOOLS
_REGISTERED_READ_ONLY_TOOLS: Tuple[BaseTool, ...] = (
    READ_ONLY_TOOLS + _NARROW_S2T_TOOLS
)

ALL_TOOLS: Tuple[BaseTool, ...] = _REGISTERED_READ_ONLY_TOOLS + WRITE_TOOLS
TOOLS: Tuple[BaseTool, ...] = READ_ONLY_TOOLS
TOOLS_BY_NAME: Dict[str, BaseTool] = {
    tool.name: tool for tool in TOOLS
}
WRITE_TOOLS_BY_NAME: Dict[str, BaseTool] = {
    tool.name: tool for tool in WRITE_TOOLS
}
ALL_TOOLS_BY_NAME: Dict[str, BaseTool] = {
    tool.name: tool for tool in ALL_TOOLS
}


def _narrow_s2t_experiment_enabled() -> bool:
    return str(
        os.getenv(S2T_NARROW_TOOLS_EXPERIMENT_ENV, "")
    ).strip().lower() in {"1", "true", "yes", "on"}


def get_tools() -> Tuple[BaseTool, ...]:
    """Вернуть активную неизменяемую коллекцию read-only инструментов."""
    return (
        _EXPERIMENT_READ_ONLY_TOOLS
        if _narrow_s2t_experiment_enabled()
        else TOOLS
    )


def get_tools_for_names(
    tool_names: Iterable[str],
) -> Tuple[BaseTool, ...]:
    """Вернуть ровно выбранные read-only tools в порядке общего registry."""
    selected = tuple(dict.fromkeys(tool_names))
    if not selected:
        return ()

    active_tools = get_tools()
    active_by_name = {tool.name: tool for tool in active_tools}
    unknown = [name for name in selected if name not in active_by_name]
    if unknown:
        raise ValueError(f"Неизвестные read-only tools: {', '.join(unknown)}")

    selected_set = set(selected)
    return tuple(tool for tool in active_tools if tool.name in selected_set)


def get_tools_by_name() -> Dict[str, BaseTool]:
    """Вернуть копию read-only реестра инструментов по именам."""
    return {tool.name: tool for tool in get_tools()}


def get_write_tools() -> Tuple[BaseTool, ...]:
    """Вернуть мутирующие инструменты для подтверждаемого runtime."""
    return WRITE_TOOLS


def get_all_tools() -> Tuple[BaseTool, ...]:
    """Вернуть полный набор инструментов, включая мутирующие."""
    return ALL_TOOLS
