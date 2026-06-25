from __future__ import annotations

from app import config
from app.services.embedding_service import EmbeddingService
from app.services.text_utils import stable_hash


_EMBEDDER: EmbeddingService | None = None


def embed_text(text: str) -> list[float]:
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    vectors = _embedder().encode(texts).vectors
    return [vector.astype(float).tolist() for vector in vectors]


def get_embedding_hash(text: str, model_name: str | None = None) -> str:
    model = model_name or config.HF_EMBEDDING_MODEL
    return str(stable_hash(f"{model}\n{text}"))


def embedding_model_name() -> str:
    return _embedder().backend


def _embedder() -> EmbeddingService:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = EmbeddingService()
    return _EMBEDDER
