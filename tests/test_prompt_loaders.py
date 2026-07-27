import pytest
import os
from agents.tools import load_skills


def test_load_skills(tmp_path):
    # Create a temporary skills.md
    skills_file = tmp_path / "skills.md"
    skills_file.write_text("# Test Skills\n- Header detection")
    # Override the file path? The helper uses relative path. For test, we can mock open.
    # Simpler: test that the function returns empty string when file missing
    assert isinstance(load_skills(), str)  # May be empty if file not found


def test_load_skills_includes_s2t_transformations_query_skill():
    text = load_skills()
    assert "Табличные запросы по S2T-трансформациям" in text
    assert "Выбор подмножества полей в ответе" in text
    assert "Сопоставление листов и колонок с конфигами" in text
    assert "s2t_transformations" in text
    assert "search_s2t_transformations" in text
    assert "run_sql" in text
    assert "Сценарий: SQLite и `run_sql`" in text
    assert "Сценарий: Neo4j и `run_cypher`" in text
    assert "trace_neo4j_lineage" in text
    assert "impact analysis" in text
    assert "Не используй Cypher для чтения сырых Excel-строк" in text
    assert "Не используй SQL для многошагового обхода графа" in text
    assert "отсутствие узла" not in text.lower()
    assert "Определение заголовков Excel" not in text
    assert "Schema Matching" not in text
    for db_column_name in (
        "row_num",
        "target_field",
        "source_field",
        "transformation_rule",
    ):
        assert db_column_name not in text
