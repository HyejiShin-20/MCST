from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from app import config
from app.services.text_utils import stable_hash, tokenize


@dataclass
class EmbeddingResult:
    vectors: list[np.ndarray]
    backend: str


class HashingTextEmbedder:
    """Small deterministic fallback when local HuggingFace packages are absent."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in tokenize(text):
            hashed = stable_hash(token)
            index = hashed % self.dimensions
            sign = 1.0 if (hashed >> 1) % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector


class EmbeddingService:
    def __init__(self) -> None:
        self.backend = "hashing-fallback"
        self.model = None
        self.fallback = HashingTextEmbedder()
        self.provider = config.EMBEDDING_PROVIDER
        if self.provider == "openai":
            self.backend = f"openai:{config.OPENAI_EMBEDDING_MODEL}"
        else:
            self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self) -> None:
        if config.HF_LOCAL_FILES_ONLY and not self._local_model_available(
            config.HF_EMBEDDING_MODEL
        ):
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            kwargs = {"local_files_only": config.HF_LOCAL_FILES_ONLY}
            self.model = SentenceTransformer(config.HF_EMBEDDING_MODEL, **kwargs)
            self.backend = f"huggingface:{config.HF_EMBEDDING_MODEL}"
        except Exception:
            self.model = None

    @staticmethod
    def _local_model_available(model_name: str) -> bool:
        model_path = Path(model_name)
        if model_path.exists():
            return True
        cache_roots = [
            os.getenv("HF_HOME"),
            os.getenv("HUGGINGFACE_HUB_CACHE"),
            os.getenv("HF_HUB_CACHE"),
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ]
        cache_name = "models--" + model_name.replace("/", "--")
        for root in cache_roots:
            if not root:
                continue
            candidate = Path(root) / cache_name / "snapshots"
            if candidate.exists() and any(candidate.iterdir()):
                return True
        return False

    def encode(self, texts: Iterable[str]) -> EmbeddingResult:
        text_list = list(texts)
        if self.provider == "openai":
            vectors = self._encode_openai(text_list)
            return EmbeddingResult(vectors, self.backend)
        if self.model is not None:
            vectors = self.model.encode(text_list, normalize_embeddings=True)
            return EmbeddingResult(
                [np.asarray(vector, dtype=np.float32) for vector in vectors],
                self.backend,
            )
        return EmbeddingResult(self.fallback.encode(text_list), self.backend)

    def _encode_openai(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=config.OPENAI_API_KEY or None)
            response = client.embeddings.create(
                model=config.OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            vectors = [
                _normalized_vector(item.embedding)
                for item in sorted(response.data, key=lambda item: item.index)
            ]
            if len(vectors) == len(texts):
                return vectors
        except Exception:
            pass
        return self.fallback.encode(texts)


def _normalized_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(values), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector
