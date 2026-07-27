"""Tools for stored S2T transformations and table catalogs."""

from typing import Any, Dict, List, Literal, Optional

from langchain_core.tools import tool

from .common import clamped_int


@tool(parse_docstring=True)
def list_s2t_transformations(
    limit: int = 20,
    q: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Получить компактный фрагмент строк колоночного S2T-маппинга.

    Это основной инструмент для запросов «покажи таблицу трансформаций»,
    «покажи строки/маппинги/правила» и обычных связей source → target. Такие
    запросы относятся к табличному SQLite-сценарию, а не к Neo4j lineage.
    Возвращает row_num и настроенные поля S2T, включая target_table, target_field,
    source_table, source_field и
    transformation_rule из всей глобальной s2t_transformations. Никогда не
    ограничивает результат file_id, активным UI-файлом или последней загрузкой.
    Результат возвращается planner-у как наблюдение и сам по себе не завершает
    ответ.

    Args:
        limit: Максимальное число возвращаемых строк; фактически ограничивается 20.
        q: Опциональная подстрока для фильтрации полей S2T-маппинга.
    """
    from storage.s2t import list_s2t_transformations as db_list_s2t_transformations

    clean_limit = max(1, min(int(limit or 20), 20))
    return db_list_s2t_transformations(
        file_id=None,
        limit=clean_limit,
        q=q,
    )


@tool(parse_docstring=True)
def search_s2t_transformations(
    needle: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Найти строки S2T-трансформаций по известному имени или фрагменту значения.

    Используй для табличного поиска маппингов, колонок и правил в SQLite. Это не
    инструмент поиска графового пути, upstream/downstream или impact analysis:
    для таких задач предназначены Neo4j-tools.
    Ищет подстроку одновременно во всех настроенных S2T-полях, включая
    target_table, target_field, source_table, source_field и transformation_rule.
    Всегда ищет по всей глобальной таблице без file_id, активного UI-файла или
    выбора последней загрузки. Возвращает только фактические строки
    s2t_transformations.

    Args:
        needle: Непустая подстрока имени таблицы, колонки или правила преобразования.
        limit: Максимальное число возвращаемых совпадений, от 1 до 100.
    """
    text = (needle or "").strip()
    if not text:
        return {"error": "needle must be non-empty", "query": needle, "total": 0, "rows": []}
    if len(text) > 200:
        return {"error": "needle too long", "query": text, "total": 0, "rows": []}

    from storage.s2t import list_s2t_transformations as db_list_s2t_transformations

    clean_limit = max(1, min(int(limit or 20), 100))
    data = db_list_s2t_transformations(
        file_id=None,
        limit=clean_limit,
        q=text,
    )
    data["query"] = text
    data["searched_table"] = "s2t_transformations"
    from storage.database import S2T_FIELDS

    data["searched_columns"] = list(S2T_FIELDS)
    return data


@tool(parse_docstring=True)
def summarize_s2t_tables(
    group_by: Literal["source", "target"] = "target",
    min_related_tables: int = 1,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Агрегировать показатели колоночных маппингов по источникам или приёмникам.

    Используй для табличных подсчётов и сводок source → target в SQLite, а не для
    lineage, путей, цепочек зависимостей или других графовых обходов Neo4j.
    Для каждой логической таблицы считает строки маппинга, представленные поля,
    связанные таблицы противоположной роли и заполненность правил трансформации.
    Это агрегатор структуры s2t_transformations, а не поиск текстовых описаний из
    каталогов source_tables и target_tables. Всегда анализирует глобальную таблицу
    без ограничения по file_id.

    Args:
        group_by: Роль группировки — source для источников или target для приёмников.
        min_related_tables: Минимальное число связанных таблиц противоположной роли.
        limit: Максимальное число агрегированных групп, от 1 до 200.
    """
    from storage.s2t import summarize_s2t_transformations

    data = summarize_s2t_transformations(
        group_by=group_by,
        file_id=None,
        min_related_tables=min_related_tables,
        limit=limit,
    )
    data.pop("file_id", None)
    data["scope"] = "global"
    return data


@tool(parse_docstring=True)
def summarize_table_descriptions(
    table_name: str,
    file_id: Optional[int] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Собрать описания одной логической ETL-таблицы из обоих каталогов.

    Ищет точное имя без учёта регистра и внешних пробелов одновременно в
    source_tables и target_tables через UNION ALL. Возвращает роли, исходные строки
    и объединённый список непустых описаний. Одинаковые записи сохраняются отдельно;
    итоговое краткое русское описание формирует planner по полученным фактам.

    Args:
        table_name: Точное логическое имя таблицы, например t_agr_cred.
        file_id: Опциональный числовой идентификатор загрузки для ограничения поиска.
        limit: Максимальное число исходных строк, возвращаемых planner-у, от 1 до 100.
    """
    clean_name = str(table_name or "").strip()
    if not clean_name:
        return {
            "error": "table_name must be non-empty",
            "table_name": table_name,
            "matches": [],
            "total_matches": 0,
        }
    if len(clean_name) > 300:
        return {
            "error": "table_name too long",
            "table_name": clean_name,
            "matches": [],
            "total_matches": 0,
        }

    clean_file_id = int(file_id) if file_id is not None else None
    clean_limit = clamped_int(limit, 50, 1, 100)
    scope_sql = " AND catalog.file_id = ?" if clean_file_id is not None else ""
    params: List[Any] = [clean_name]
    if clean_file_id is not None:
        params.append(clean_file_id)
    params.append(clean_name)
    if clean_file_id is not None:
        params.append(clean_file_id)
    params.append(clean_limit)

    query = f"""
        WITH matched AS (
            SELECT
                catalog.id AS catalog_id,
                catalog.file_id,
                files.filename,
                catalog.sheet_id,
                catalog.sheet_name,
                catalog.row_num,
                'source' AS table_role,
                catalog.table_name,
                catalog.description
            FROM source_tables AS catalog
            LEFT JOIN files ON files.file_id = catalog.file_id
            WHERE TRIM(catalog.table_name) = ? COLLATE NOCASE
              {scope_sql}

            UNION ALL

            SELECT
                catalog.id AS catalog_id,
                catalog.file_id,
                files.filename,
                catalog.sheet_id,
                catalog.sheet_name,
                catalog.row_num,
                'target' AS table_role,
                catalog.table_name,
                catalog.description
            FROM target_tables AS catalog
            LEFT JOIN files ON files.file_id = catalog.file_id
            WHERE TRIM(catalog.table_name) = ? COLLATE NOCASE
              {scope_sql}
        )
        SELECT
            catalog_id,
            file_id,
            filename,
            sheet_id,
            sheet_name,
            row_num,
            table_role,
            table_name,
            description,
            COUNT(*) OVER () AS total_matches,
            SUM(CASE WHEN table_role = 'source' THEN 1 ELSE 0 END)
                OVER () AS source_matches,
            SUM(CASE WHEN table_role = 'target' THEN 1 ELSE 0 END)
                OVER () AS target_matches,
            SUM(CASE WHEN NULLIF(TRIM(description), '') IS NOT NULL THEN 1 ELSE 0 END)
                OVER () AS descriptions_present
        FROM matched
        ORDER BY table_role, file_id, sheet_id, row_num, catalog_id
        LIMIT ?
    """

    from storage.database import get_db_connection

    conn = get_db_connection()
    try:
        fetched = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    total_matches = int(fetched[0]["total_matches"]) if fetched else 0
    role_counts = {
        "source": int(fetched[0]["source_matches"] or 0) if fetched else 0,
        "target": int(fetched[0]["target_matches"] or 0) if fetched else 0,
    }
    descriptions_present = (
        int(fetched[0]["descriptions_present"] or 0) if fetched else 0
    )
    matches = []
    combined_descriptions = []
    for row in fetched:
        item = dict(row)
        for aggregate_field in (
            "total_matches",
            "source_matches",
            "target_matches",
            "descriptions_present",
        ):
            item.pop(aggregate_field, None)
        matches.append(item)
        description = str(item.get("description") or "").strip()
        if description:
            combined_descriptions.append(
                {
                    "table_role": item["table_role"],
                    "description": description,
                }
            )

    return {
        "table_name": clean_name,
        "file_id": clean_file_id,
        "searched_tables": ["source_tables", "target_tables"],
        "total_matches": total_matches,
        "returned_matches": len(matches),
        "role_counts": role_counts,
        "descriptions_present": descriptions_present,
        "matches": matches,
        "combined_descriptions": combined_descriptions,
    }
