"""Identity checks and explicit migration for persisted description embeddings."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

from services.embeddings import EmbeddingIndexIdentity


EMBEDDING_INDEX_NAME = "description_embeddings_v1"
EMBEDDING_TABLES = (
    ("files", "file_id"),
    ("source_tables", "id"),
    ("target_tables", "id"),
    ("source_columns", "id"),
    ("target_columns", "id"),
)
IDENTITY_FIELDS = (
    "model_name",
    "model_revision",
    "profile_id",
    "query_prefix",
    "document_prefix",
    "normalize_embeddings",
    "dimension",
)


class EmbeddingIndexCompatibilityError(RuntimeError):
    """Raised when stored vectors cannot be compared with the runtime encoder."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def column_embedding_document(
    column_name: Optional[Any],
    description: Optional[Any],
) -> str:
    """Build the canonical document text for one stored column."""
    clean_name = str(column_name).strip() if column_name is not None else ""
    clean_description = str(description).strip() if description is not None else ""
    parts = []
    if clean_name:
        parts.append(f"Название колонки: {clean_name}")
    if clean_description:
        parts.append(f"Описание: {clean_description}")
    return "\n".join(parts)


def _embedding_blob_count(cursor: sqlite3.Cursor) -> int:
    return sum(
        int(
            cursor.execute(
                f"SELECT COUNT(*) FROM {table_name} "
                "WHERE description_embedding IS NOT NULL"
            ).fetchone()[0]
        )
        for table_name, _ in EMBEDDING_TABLES
    )


def get_embedding_index_metadata(
    cursor: sqlite3.Cursor,
) -> Optional[dict[str, Any]]:
    """Read the one supported index identity from an existing connection."""
    row = cursor.execute(
        """
        SELECT model_name, model_revision, profile_id, query_prefix,
               document_prefix, normalize_embeddings, dimension
        FROM embedding_index_metadata
        WHERE index_name = ?
        """,
        (EMBEDDING_INDEX_NAME,),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["normalize_embeddings"] = bool(result["normalize_embeddings"])
    result["dimension"] = int(result["dimension"])
    return result


def _mismatched_fields(
    stored: dict[str, Any],
    identity: EmbeddingIndexIdentity,
) -> list[str]:
    runtime = identity.as_dict()
    return [field for field in IDENTITY_FIELDS if stored.get(field) != runtime[field]]


def validate_embedding_index(
    cursor: sqlite3.Cursor,
    identity: EmbeddingIndexIdentity,
) -> Optional[dict[str, Any]]:
    """Validate index metadata without mutating a read-only search connection."""
    stored = get_embedding_index_metadata(cursor)
    if stored is None:
        if _embedding_blob_count(cursor):
            raise EmbeddingIndexCompatibilityError(
                "Stored embeddings have no confirmed model/encoding metadata; "
                "run the explicit embedding reindex migration",
                error_code="embedding_index_unregistered",
            )
        return None
    mismatches = _mismatched_fields(stored, identity)
    if mismatches:
        raise EmbeddingIndexCompatibilityError(
            "Stored embedding index is incompatible with the runtime encoder "
            f"({', '.join(mismatches)}); run the explicit embedding reindex migration",
            error_code="embedding_index_incompatible",
        )
    return stored


def _write_embedding_index_metadata(
    cursor: sqlite3.Cursor,
    identity: EmbeddingIndexIdentity,
) -> None:
    cursor.execute(
        """
        INSERT INTO embedding_index_metadata
        (index_name, model_name, model_revision, profile_id, query_prefix,
         document_prefix, normalize_embeddings, dimension, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(index_name) DO UPDATE SET
            model_name = excluded.model_name,
            model_revision = excluded.model_revision,
            profile_id = excluded.profile_id,
            query_prefix = excluded.query_prefix,
            document_prefix = excluded.document_prefix,
            normalize_embeddings = excluded.normalize_embeddings,
            dimension = excluded.dimension,
            updated_at = excluded.updated_at
        """,
        (
            EMBEDDING_INDEX_NAME,
            identity.model_name,
            identity.model_revision,
            identity.profile_id,
            identity.query_prefix,
            identity.document_prefix,
            1 if identity.normalize_embeddings else 0,
            identity.dimension,
            datetime.now().isoformat(),
        ),
    )


def register_embedding_index(
    cursor: sqlite3.Cursor,
    identity: EmbeddingIndexIdentity,
) -> None:
    """Register a fresh index or require an exact match before adding vectors."""
    stored = get_embedding_index_metadata(cursor)
    if stored is not None:
        mismatches = _mismatched_fields(stored, identity)
        if mismatches:
            raise EmbeddingIndexCompatibilityError(
                "Cannot mix embeddings from different index identities "
                f"({', '.join(mismatches)}); run the explicit reindex migration",
                error_code="embedding_index_incompatible",
            )
        return
    if _embedding_blob_count(cursor):
        raise EmbeddingIndexCompatibilityError(
            "Legacy embeddings without confirmed metadata must be explicitly reindexed",
            error_code="embedding_index_unregistered",
        )
    _write_embedding_index_metadata(cursor, identity)


def register_embedding_blobs(
    cursor: sqlite3.Cursor,
    blobs: Sequence[bytes],
) -> Optional[EmbeddingIndexIdentity]:
    """Validate one generated batch and register its identity transactionally."""
    if not blobs:
        return None
    from services.embeddings import embedding_index_identity_for_blobs

    identity = embedding_index_identity_for_blobs(blobs)
    register_embedding_index(cursor, identity)
    return identity


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def reindex_description_embeddings(batch_size: int = 128) -> dict[str, Any]:
    """Explicitly replace all description vectors and their persisted identity."""
    from services.embeddings import (
        embed_documents,
        embedding_index_identity_for_blobs,
    )
    from storage.database import get_db_connection

    clean_batch_size = int(batch_size)
    if clean_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    conn = get_db_connection()
    try:
        targets: list[tuple[str, int, str]] = []
        for table_name, key_name in EMBEDDING_TABLES[:3]:
            rows = conn.execute(
                f"SELECT {key_name} AS record_id, description FROM {table_name} "
                "WHERE description IS NOT NULL AND trim(description) <> '' "
                f"ORDER BY {key_name}"
            ).fetchall()
            targets.extend(
                (table_name, int(row["record_id"]), str(row["description"]))
                for row in rows
            )
        for table_name, key_name in EMBEDDING_TABLES[3:]:
            rows = conn.execute(
                f"SELECT {key_name} AS record_id, column_name, description "
                f"FROM {table_name} ORDER BY {key_name}"
            ).fetchall()
            for row in rows:
                document = column_embedding_document(
                    row["column_name"], row["description"]
                )
                if document:
                    targets.append((table_name, int(row["record_id"]), document))

        documents = [target[2] for target in targets]
        blobs: list[bytes] = []
        for batch in _chunks(documents, clean_batch_size):
            encoded = embed_documents(batch)
            if len(encoded) != len(batch):
                raise ValueError("Embedding encoder returned an incomplete batch")
            blobs.extend(encoded)
        identity = (
            embedding_index_identity_for_blobs(blobs) if blobs else None
        )

        cursor = conn.cursor()
        cursor.execute("BEGIN")
        for table_name, _ in EMBEDDING_TABLES:
            cursor.execute(
                f"UPDATE {table_name} SET description_embedding = NULL"
            )
        cursor.execute(
            "DELETE FROM embedding_index_metadata WHERE index_name = ?",
            (EMBEDDING_INDEX_NAME,),
        )
        grouped: dict[str, list[tuple[bytes, int]]] = {
            table_name: [] for table_name, _ in EMBEDDING_TABLES
        }
        key_names = dict(EMBEDDING_TABLES)
        for (table_name, record_id, _), blob in zip(targets, blobs):
            grouped[table_name].append((blob, record_id))
        for table_name, updates in grouped.items():
            if updates:
                cursor.executemany(
                    f"UPDATE {table_name} SET description_embedding = ? "
                    f"WHERE {key_names[table_name]} = ?",
                    updates,
                )
        if identity is not None:
            _write_embedding_index_metadata(cursor, identity)
        conn.commit()
        return {
            "status": "ok",
            "updated": len(blobs),
            "counts": {
                table_name: len(updates)
                for table_name, updates in grouped.items()
            },
            "index_identity": identity.as_dict() if identity else None,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


__all__ = [
    "EMBEDDING_INDEX_NAME",
    "EMBEDDING_TABLES",
    "EmbeddingIndexCompatibilityError",
    "column_embedding_document",
    "get_embedding_index_metadata",
    "register_embedding_blobs",
    "register_embedding_index",
    "reindex_description_embeddings",
    "validate_embedding_index",
]
