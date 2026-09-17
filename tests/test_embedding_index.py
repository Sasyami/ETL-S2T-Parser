from array import array

import pytest

from services import embeddings
from storage.database import get_db_connection, update_file_description
from storage.embedding_index import (
    EmbeddingIndexCompatibilityError,
    get_embedding_index_metadata,
    register_embedding_blobs,
    reindex_description_embeddings,
    validate_embedding_index,
)


def _vector(*values: float) -> bytes:
    return array("f", values).tobytes()


def _configure_plain_profile(monkeypatch, model_name="test-model"):
    monkeypatch.setenv("EMBEDDING_MODEL", model_name)
    monkeypatch.setenv("EMBEDDING_PROFILE", "plain-normalized-v1")
    monkeypatch.delenv("EMBEDDING_MODEL_REVISION", raising=False)


def test_fresh_database_has_no_claimed_embedding_identity(temp_db):
    assert get_embedding_index_metadata(temp_db.cursor()) is None


def test_document_write_rejects_legacy_blobs_without_metadata(
    temp_db, monkeypatch
):
    _configure_plain_profile(monkeypatch)
    monkeypatch.setattr(
        embeddings, "embed_document", lambda text: _vector(1.0, 0.0)
    )
    temp_db.execute(
        """
        INSERT INTO files
        (file_id, filename, upload_time, description, description_embedding)
        VALUES (1, 'legacy.xlsx', '2026-01-01', 'old', ?)
        """,
        (_vector(0.0, 1.0),),
    )
    temp_db.commit()

    with pytest.raises(
        EmbeddingIndexCompatibilityError,
        match="Legacy embeddings",
    ):
        update_file_description(1, "new")

    row = temp_db.execute(
        "SELECT description, description_embedding FROM files WHERE file_id = 1"
    ).fetchone()
    assert tuple(row) == ("old", _vector(0.0, 1.0))
    assert get_embedding_index_metadata(temp_db.cursor()) is None


def test_same_dimension_from_different_model_is_incompatible(temp_db, monkeypatch):
    _configure_plain_profile(monkeypatch, "model-a")
    register_embedding_blobs(temp_db.cursor(), [_vector(1.0, 0.0)])
    temp_db.commit()

    _configure_plain_profile(monkeypatch, "model-b")
    runtime_identity = embeddings.embedding_index_identity(2)
    with pytest.raises(
        EmbeddingIndexCompatibilityError,
        match="model_name",
    ) as exc_info:
        validate_embedding_index(temp_db.cursor(), runtime_identity)
    assert exc_info.value.error_code == "embedding_index_incompatible"


def test_registered_index_rejects_runtime_dimension_change(temp_db, monkeypatch):
    _configure_plain_profile(monkeypatch)
    register_embedding_blobs(temp_db.cursor(), [_vector(1.0, 0.0)])
    temp_db.commit()

    with pytest.raises(
        EmbeddingIndexCompatibilityError,
        match="dimension",
    ):
        validate_embedding_index(
            temp_db.cursor(), embeddings.embedding_index_identity(3)
        )


def test_explicit_reindex_migrates_every_embedding_catalog(temp_db, monkeypatch):
    _configure_plain_profile(monkeypatch, "reindexed-model")
    legacy = b"legacy-vector"
    temp_db.execute(
        """
        INSERT INTO files
        (file_id, filename, upload_time, description, description_embedding)
        VALUES (1, 'catalog.xlsx', '2026-01-01', 'Файл', ?)
        """,
        (legacy,),
    )
    for table_name, record_id, table_name_value, description in (
        ("source_tables", 2, "src_order", "Заказы"),
        ("target_tables", 3, "t_order", "Витрина заказов"),
    ):
        temp_db.execute(
            f"""
            INSERT INTO {table_name}
            (id, file_id, sheet_name, row_num, table_name, description,
             description_embedding)
            VALUES (?, 1, 'Tables', 0, ?, ?, ?)
            """,
            (record_id, table_name_value, description, legacy),
        )
    for table_name, record_id, table_name_value, column_name, description in (
        ("source_columns", 4, "src_order", "order_id", "Ключ заказа"),
        ("target_columns", 5, "t_order", "order_id", "Ключ витрины"),
    ):
        temp_db.execute(
            f"""
            INSERT INTO {table_name}
            (id, file_id, sheet_name, row_num, table_name, column_name,
             description, description_embedding)
            VALUES (?, 1, 'Columns', 0, ?, ?, ?, ?)
            """,
            (record_id, table_name_value, column_name, description, legacy),
        )
    temp_db.commit()

    encoded_documents = []

    def fake_embed_documents(documents):
        encoded_documents.extend(documents)
        return [_vector(float(len(document)), 1.0) for document in documents]

    monkeypatch.setattr(embeddings, "embed_documents", fake_embed_documents)

    report = reindex_description_embeddings(batch_size=2)

    assert report["updated"] == 5
    assert report["counts"] == {
        "files": 1,
        "source_tables": 1,
        "target_tables": 1,
        "source_columns": 1,
        "target_columns": 1,
    }
    assert "Название колонки: order_id\nОписание: Ключ заказа" in encoded_documents
    assert report["index_identity"] == {
        "model_name": "reindexed-model",
        "model_revision": "",
        "profile_id": "plain-normalized-v1",
        "query_prefix": "",
        "document_prefix": "",
        "normalize_embeddings": True,
        "dimension": 2,
    }
    for table_name, _ in (
        ("files", "file_id"),
        ("source_tables", "id"),
        ("target_tables", "id"),
        ("source_columns", "id"),
        ("target_columns", "id"),
    ):
        blob = temp_db.execute(
            f"SELECT description_embedding FROM {table_name}"
        ).fetchone()[0]
        assert blob is not None
        assert blob != legacy
    assert get_embedding_index_metadata(temp_db.cursor()) == report["index_identity"]


def test_reindex_encoder_failure_preserves_legacy_index(temp_db, monkeypatch):
    _configure_plain_profile(monkeypatch)
    temp_db.execute(
        """
        INSERT INTO files
        (file_id, filename, upload_time, description, description_embedding)
        VALUES (1, 'legacy.xlsx', '2026-01-01', 'old', ?)
        """,
        (b"legacy",),
    )
    temp_db.commit()
    monkeypatch.setattr(
        embeddings,
        "embed_documents",
        lambda documents: (_ for _ in ()).throw(RuntimeError("encoder failed")),
    )

    with pytest.raises(RuntimeError, match="encoder failed"):
        reindex_description_embeddings()

    assert temp_db.execute(
        "SELECT description_embedding FROM files WHERE file_id = 1"
    ).fetchone()[0] == b"legacy"
    assert get_embedding_index_metadata(temp_db.cursor()) is None
