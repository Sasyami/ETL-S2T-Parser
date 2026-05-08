import json
import tempfile
import pytest
from unittest.mock import patch

import db_storage
from db_storage import init_db, get_db_connection


@pytest.fixture(autouse=True)
def _temp_db_path():
    original = db_storage.DB_PATH
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_storage.DB_PATH = tmp.name
        init_db()
        yield
        db_storage.DB_PATH = original


@patch("semantic_layer.create_embedding")
def test_similarity_search_empty_embeddings(mock_embed):
    mock_embed.return_value = json.dumps([1.0, 0.0, 0.0])
    from semantic_layer import similarity_search

    assert similarity_search("hello") == []


@patch("semantic_layer.create_embedding")
def test_similarity_search_ranks_by_cosine(mock_embed):
    mock_embed.return_value = json.dumps([1.0, 0.0, 0.0])
    from semantic_layer import similarity_search

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO embeddings (id, entity_id, entity_type, vector)
        VALUES ('e1', 'c1', 'column', ?), ('e2', 'c2', 'column', ?)
        """,
        (json.dumps([0.0, 1.0, 0.0]), json.dumps([1.0, 0.0, 0.0])),
    )
    conn.commit()
    conn.close()

    hits = similarity_search("q", top_k=2)
    assert len(hits) == 2
    assert hits[0]["entity_id"] == "c2"
    assert hits[0]["similarity"] == pytest.approx(1.0)
    assert hits[1]["entity_id"] == "c1"
    assert hits[1]["similarity"] == pytest.approx(0.0)
