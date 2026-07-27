"""Local embeddings for stored descriptions."""

import os
from functools import lru_cache
from typing import Sequence

from dotenv import load_dotenv


load_dotenv()
MODEL_NAME = os.environ["EMBEDDING_MODEL"]


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_descriptions(texts: Sequence[str]) -> list[bytes]:
    vectors = _get_model().encode(
        [f"search_document: {text}" for text in texts],
        normalize_embeddings=True,
    )
    return [vector.astype("float32").tobytes() for vector in vectors]


def embed_description(text: str) -> bytes:
    return embed_descriptions([text])[0]
