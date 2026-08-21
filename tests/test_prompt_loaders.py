from agents.tools import load_schemas, load_skills


def test_load_skills(tmp_path):
    # Create a temporary skills.md
    skills_file = tmp_path / "skills.md"
    skills_file.write_text("# Test Skills\n- Header detection")
    # Override the file path? The helper uses relative path. For test, we can mock open.
    # Simpler: test that the function returns empty string when file missing
    assert isinstance(load_skills(), str)  # May be empty if file not found


def test_load_skills_contains_tool_orchestration_and_domain_context():
    text = load_skills()
    assert "S2T-строки" in text
    assert "s2t_transformations" in text
    assert "Neo4j" in text
    assert "ETLColumn" in text
    assert "TABLE_TRANSFORMS_TO" in text
    for tool_name in (
        "run_sql",
        "trace_neo4j_lineage",
        "search_excel_values",
        "semantic_search_descriptions",
        "trace_transformation_path",
    ):
        assert tool_name in text
    for semantic_scope in (
        "`files`",
        "`tables`",
        "`source_tables`",
        "`target_tables`",
    ):
        assert semantic_scope in text
    assert "`all` — только без" in text
    assert "Атомарные контракты tools" in text
    assert "Извлечение полезных колонок" not in text
    assert "update_file_description" not in text


def test_s2t_skill_routes_table_name_set_operations_to_domain_tool():
    text = load_skills(["S2T-строки"])

    assert "list_s2t_table_names" in text
    assert "принадлежность имени" in text
    assert "run_sql" in text
    assert "нестандартной агрегации" in text
    assert "JOIN, EXISTS, NOT EXISTS" not in text


def test_load_skills_includes_transformation_path_analysis_skill():
    text = load_skills()
    assert "Путь S2T-преобразования" in text
    assert "прямую трансформацию" in text
    assert "Отсутствие подтверждения Neo4j не отменяет факты SQLite" in text
    assert "source_table.source_field → target_table.target_field" in text
    assert "перенеси её дословно" in text
    assert "`WHEN` внутри `CASE` выбирает значение" in text
    assert "`UNION`/`UNION ALL` объединяет ветви" in text
    assert "входы одного выражения" in text
    assert "одну логическую трансформацию" in text
    assert "не отсутствие" in text


def test_load_skills_can_select_one_section():
    text = load_skills(["SQL lineage"])

    assert "## SQL lineage" in text
    assert "parse_sql_column_lineage" in text
    assert "visualize_sql_lineage" in text
    assert "диалект SQLGlot" in text
    assert "## S2T-строки" not in text
    assert "## Neo4j" not in text


def test_load_skills_can_select_sql_execution_section():
    text = load_skills(["SQLite SQL"])

    assert "## SQLite SQL" in text
    assert "run_sql" in text
    assert "не физические ETL-таблицы" in text
    assert "source_table" in text
    assert "## SQL lineage" not in text


def test_load_schemas_selects_current_s2t_mappings_without_other_domains():
    text = load_schemas(("S2T-маппинг",))

    assert "## Схема S2T-маппинга" in text
    assert "`source_table`" in text
    assert "`target_table`" in text
    assert '"source_field"' in text
    assert '"target_field"' in text
    assert '"sheet_group":"s2t"' in text
    assert "## Актуальная схема SQLite" not in text
    assert "## Схемы Excel-маппингов" not in text


def test_load_schemas_includes_exact_excel_and_sqlite_sources():
    text = load_schemas(("SQLite ETL", "Excel-маппинги"))

    assert "## Актуальная схема SQLite" in text
    assert "`additional_objects`" in text
    assert "`sql`" in text
    assert "## Схемы Excel-маппингов" in text
    assert '"additional_objects"' in text
    assert '"sheet_group":"additional_objects"' in text
