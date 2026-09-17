import sys
from types import SimpleNamespace

import pytest

from services import embeddings
from services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROFILE,
    embedding_model_name,
)


class _FakeVector:
    def __init__(self, value):
        self.value = value

    def astype(self, dtype):
        assert dtype == "float32"
        return self

    def tobytes(self):
        return self.value.encode("utf-8")


def _install_fake_sentence_transformers(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs
            self.calls = []
            created.append(self)

        def encode(self, texts, **kwargs):
            self.calls.append((list(texts), kwargs))
            return [_FakeVector(text) for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    embeddings._get_model.cache_clear()
    return created


def test_embedding_model_name_uses_default_when_env_missing(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert embedding_model_name() == DEFAULT_EMBEDDING_MODEL


def test_embedding_model_name_reads_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    assert embedding_model_name() == "custom/model"


def test_default_e5_profile_uses_distinct_exact_query_and_document_prefixes(
    monkeypatch,
):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL_REVISION", raising=False)
    monkeypatch.delenv("EMBEDDING_PROFILE", raising=False)
    created = _install_fake_sentence_transformers(monkeypatch)

    embeddings.embed_documents(["документ один", "документ два"])
    embeddings.embed_query("поисковый запрос")

    assert embeddings.embedding_profile_name() == DEFAULT_EMBEDDING_PROFILE
    assert len(created) == 1
    assert created[0].calls == [
        (
            ["passage: документ один", "passage: документ два"],
            {"normalize_embeddings": True},
        ),
        (["query: поисковый запрос"], {"normalize_embeddings": True}),
    ]


def test_custom_model_requires_explicit_profile(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "vendor/custom-model")
    monkeypatch.delenv("EMBEDDING_PROFILE", raising=False)

    with pytest.raises(ValueError, match="EMBEDDING_PROFILE must be set"):
        embeddings.embedding_profile_name()


def test_plain_custom_profile_is_not_changed_based_on_model_name(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "vendor/name-containing-e5")
    monkeypatch.setenv("EMBEDDING_PROFILE", "plain-normalized-v1")
    created = _install_fake_sentence_transformers(monkeypatch)

    embeddings.embed_documents(["document"])
    embeddings.embed_query("query")

    assert created[0].calls == [
        (["document"], {"normalize_embeddings": True}),
        (["query"], {"normalize_embeddings": True}),
    ]


def test_model_cache_key_includes_explicit_profile_and_revision(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "vendor/custom-model")
    monkeypatch.setenv("EMBEDDING_MODEL_REVISION", "commit-123")
    monkeypatch.setenv("EMBEDDING_PROFILE", "plain-normalized-v1")
    created = _install_fake_sentence_transformers(monkeypatch)

    embeddings.embed_query("first")
    embeddings.embed_query("second")
    monkeypatch.setenv("EMBEDDING_PROFILE", "multilingual-e5-v1")
    embeddings.embed_query("third")

    assert len(created) == 2
    assert created[0].kwargs == {"revision": "commit-123"}
    assert created[1].kwargs == {"revision": "commit-123"}
