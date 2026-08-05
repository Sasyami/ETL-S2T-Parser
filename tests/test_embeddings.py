import pytest

from services.embeddings import embedding_model_name


def test_embedding_model_name_requires_environment_value(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    with pytest.raises(KeyError, match="EMBEDDING_MODEL"):
        embedding_model_name()


def test_embedding_model_name_reads_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    assert embedding_model_name() == "custom/model"
