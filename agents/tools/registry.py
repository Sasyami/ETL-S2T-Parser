"""Explicit read-only and mutating tool registries."""

from typing import Dict, Tuple

from langchain_core.tools import BaseTool

from .files import (
    get_file_description,
    list_files,
    resolve_file,
    update_file_description,
    update_table_info_from_user_query,
)
from .planning import show_plan
from .neo4j import run_cypher, trace_neo4j_lineage
from .s2t import (
    list_s2t_transformations,
    search_s2t_transformations,
    summarize_s2t_tables,
    summarize_table_descriptions,
)
from .sheets import (
    list_columns,
    list_file_sheet_headers,
    list_sheet_group_classifications,
    list_sheets,
)
from .sql import run_sql

READ_ONLY_TOOLS: Tuple[BaseTool, ...] = (
    show_plan,
    run_sql,
    run_cypher,
    trace_neo4j_lineage,
    list_files,
    resolve_file,
    get_file_description,
    list_s2t_transformations,
    search_s2t_transformations,
    summarize_s2t_tables,
    summarize_table_descriptions,
    list_sheets,
    list_file_sheet_headers,
    list_sheet_group_classifications,
    list_columns,
)

WRITE_TOOLS: Tuple[BaseTool, ...] = (
    update_file_description,
    update_table_info_from_user_query,
)

ALL_TOOLS: Tuple[BaseTool, ...] = READ_ONLY_TOOLS + WRITE_TOOLS
TOOLS: Tuple[BaseTool, ...] = READ_ONLY_TOOLS
TOOLS_BY_NAME: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}
WRITE_TOOLS_BY_NAME: Dict[str, BaseTool] = {
    tool.name: tool for tool in WRITE_TOOLS
}
ALL_TOOLS_BY_NAME: Dict[str, BaseTool] = {
    tool.name: tool for tool in ALL_TOOLS
}


def get_tools() -> Tuple[BaseTool, ...]:
    """Вернуть неизменяемую коллекцию read-only инструментов."""
    return TOOLS


def get_tools_by_name() -> Dict[str, BaseTool]:
    """Вернуть копию read-only реестра инструментов по именам."""
    return dict(TOOLS_BY_NAME)


def get_write_tools() -> Tuple[BaseTool, ...]:
    """Вернуть мутирующие инструменты для подтверждаемого runtime."""
    return WRITE_TOOLS


def get_all_tools() -> Tuple[BaseTool, ...]:
    """Вернуть полный набор инструментов, включая мутирующие."""
    return ALL_TOOLS
