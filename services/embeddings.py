"""Configured local embeddings for document indexing and semantic queries."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Sequence

from dotenv import load_dotenv


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_EMBEDDING_PROFILE = "multilingual-e5-v1"


@dataclass(frozen=True)
class EmbeddingProfile:
    """Exact query/document encoding contract for one embedding index."""

    profile_id: str
    query_prefix: str
    document_prefix: str
    normalize_embeddings: bool


@dataclass(frozen=True)
class EmbeddingIndexIdentity:
    """Persisted identity required before vectors may be compared."""

    model_name: str
    model_revision: str
    profile_id: str
    query_prefix: str
    document_prefix: str
    normalize_embeddings: bool
    dimension: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EMBEDDING_PROFILES = {
    DEFAULT_EMBEDDING_PROFILE: EmbeddingProfile(
        profile_id=DEFAULT_EMBEDDING_PROFILE,
        query_prefix="query: ",
        document_prefix="passage: ",
        normalize_embeddings=True,
    ),
    "plain-normalized-v1": EmbeddingProfile(
        profile_id="plain-normalized-v1",
        query_prefix="",
        document_prefix="",
        normalize_embeddings=True,
    ),
}


def embedding_model_name() -> str:
    """Return configured Sentence Transformers model or the project default."""
    return (os.getenv("EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL


def embedding_model_revision() -> str:
    """Return an optional immutable model revision recorded with the index."""
    return (os.getenv("EMBEDDING_MODEL_REVISION") or "").strip()


def embedding_profile_name() -> str:
    """Return an explicit encoding profile without guessing from a model name."""
    configured = (os.getenv("EMBEDDING_PROFILE") or "").strip()
    if configured:
        if configured not in EMBEDDING_PROFILES:
            choices = ", ".join(sorted(EMBEDDING_PROFILES))
            raise ValueError(
                f"Unknown EMBEDDING_PROFILE {configured!r}; expected one of: {choices}"
            )
        return configured
    if (os.getenv("EMBEDDING_MODEL") or "").strip():
        raise ValueError(
            "EMBEDDING_PROFILE must be set when EMBEDDING_MODEL is configured"
        )
    return DEFAULT_EMBEDDING_PROFILE


def embedding_profile() -> EmbeddingProfile:
    """Return the configured immutable query/document encoding profile."""
    return EMBEDDING_PROFILES[embedding_profile_name()]


@lru_cache(maxsize=8)
def _get_model(model_name: str, model_revision: str, profile_id: str):
    """Load a model under a cache key that includes its full configured profile."""
    del profile_id  # It is intentionally part of the cache key.
    from sentence_transformers import SentenceTransformer

    kwargs = {"revision": model_revision} if model_revision else {}
    return SentenceTransformer(model_name, **kwargs)


def _encode(texts: Sequence[str], *, role: str) -> list[bytes]:
    profile = embedding_profile()
    prefix = profile.query_prefix if role == "query" else profile.document_prefix
    prepared = [f"{prefix}{str(text or '')}" for text in texts]
    model = _get_model(
        embedding_model_name(),
        embedding_model_revision(),
        profile.profile_id,
    )
    vectors = model.encode(
        prepared,
        normalize_embeddings=profile.normalize_embeddings,
    )
    return [vector.astype("float32").tobytes() for vector in vectors]


def embed_documents(texts: Sequence[str]) -> list[bytes]:
    """Encode documents with the configured document-side contract."""
    return _encode(texts, role="document")


def embed_document(text: str) -> bytes:
    """Encode one document with the configured document-side contract."""
    return embed_documents([text])[0]


def embed_query(text: str) -> bytes:
    """Encode one semantic query with the configured query-side contract."""
    return _encode([text], role="query")[0]


def embedding_index_identity(dimension: int) -> EmbeddingIndexIdentity:
    """Build the exact runtime index identity for a known vector dimension."""
    clean_dimension = int(dimension)
    if clean_dimension <= 0:
        raise ValueError("Embedding dimension must be positive")
    profile = embedding_profile()
    return EmbeddingIndexIdentity(
        model_name=embedding_model_name(),
        model_revision=embedding_model_revision(),
        profile_id=profile.profile_id,
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
        normalize_embeddings=profile.normalize_embeddings,
        dimension=clean_dimension,
    )


def embedding_index_identity_for_blobs(
    blobs: Sequence[bytes],
) -> EmbeddingIndexIdentity:
    """Build an identity and reject malformed or mixed-dimension vector blobs."""
    dimensions = set()
    for blob in blobs:
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise TypeError("Embedding must be a bytes-like float32 vector")
        byte_length = len(blob)
        if byte_length <= 0 or byte_length % 4:
            raise ValueError("Embedding blob must contain complete float32 values")
        dimensions.add(byte_length // 4)
    if not dimensions:
        raise ValueError("At least one embedding is required")
    if len(dimensions) != 1:
        raise ValueError("Embedding batch contains mixed vector dimensions")
    return embedding_index_identity(dimensions.pop())


# Compatibility aliases for external callers. Both remain document-side encoders.
def embed_descriptions(texts: Sequence[str]) -> list[bytes]:
    return embed_documents(texts)


def embed_description(text: str) -> bytes:
    return embed_document(text)


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_PROFILE",
    "EMBEDDING_PROFILES",
    "EmbeddingIndexIdentity",
    "EmbeddingProfile",
    "embed_description",
    "embed_descriptions",
    "embed_document",
    "embed_documents",
    "embed_query",
    "embedding_index_identity",
    "embedding_index_identity_for_blobs",
    "embedding_model_name",
    "embedding_model_revision",
    "embedding_profile",
    "embedding_profile_name",
]
