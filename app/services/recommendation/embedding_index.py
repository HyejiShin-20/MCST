from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.embedding_service import EmbeddingService
from app.services.text_utils import stable_hash


@dataclass
class ContentEmbeddingBatch:
    vectors: list[np.ndarray]
    stats: dict[str, int]


class ContentEmbeddingIndex:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._cache: dict[str, tuple[str, int, np.ndarray]] = {}
        self.total_hits = 0
        self.total_misses = 0

    def encode(self, contents: list[dict[str, Any]]) -> ContentEmbeddingBatch:
        vectors: list[np.ndarray | None] = [None] * len(contents)
        miss_indexes: list[int] = []
        miss_texts: list[str] = []
        request_hits = 0
        request_misses = 0

        for index, content in enumerate(contents):
            content_id = str(content["content_id"])
            embedding_text = str(content["embedding_text"])
            text_hash = stable_hash(embedding_text)
            cached = self._cache.get(content_id)
            if cached and cached[0] == self.embedding_service.backend and cached[1] == text_hash:
                vectors[index] = cached[2]
                request_hits += 1
                continue
            miss_indexes.append(index)
            miss_texts.append(embedding_text)
            request_misses += 1

        if miss_texts:
            encoded = self.embedding_service.encode(miss_texts).vectors
            for index, vector in zip(miss_indexes, encoded):
                content = contents[index]
                content_id = str(content["content_id"])
                text_hash = stable_hash(str(content["embedding_text"]))
                self._cache[content_id] = (
                    self.embedding_service.backend,
                    text_hash,
                    vector,
                )
                vectors[index] = vector

        self.total_hits += request_hits
        self.total_misses += request_misses
        completed_vectors: list[np.ndarray] = []
        for vector in vectors:
            if vector is None:
                raise RuntimeError("content embedding batch did not produce all vectors")
            completed_vectors.append(vector)

        return ContentEmbeddingBatch(
            vectors=completed_vectors,
            stats={
                "request_hits": request_hits,
                "request_misses": request_misses,
                "cache_size": len(self._cache),
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
            },
        )
