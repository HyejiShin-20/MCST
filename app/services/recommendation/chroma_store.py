from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app import config
from app.services.embedding_service import EmbeddingService
from app.services.text_utils import stable_hash, tokenize, weighted_overlap


@dataclass
class ChromaSearchResult:
    contents: list[dict[str, Any]]
    stats: dict[str, Any]


class ChromaCandidateStore:
    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._client = None
        self._collection = None
        self._available: bool | None = None

    def search(
        self,
        *,
        query_text: str,
        contents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> ChromaSearchResult:
        if not contents:
            return ChromaSearchResult([], {"backend": "chroma", "available": False})
        collection = self._get_collection()
        if collection is None:
            return ChromaSearchResult(
                contents,
                {
                    "backend": "memory",
                    "available": False,
                    "reason": "chromadb unavailable",
                    "candidate_count": len(contents),
                },
            )

        upserted = self._sync(collection, contents)
        query_vector = self.embedding_service.encode([query_text]).vectors[0]
        result = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k or config.CHROMA_TOP_K, len(contents)),
        )
        ids = [str(item) for item in (result.get("ids") or [[]])[0]]
        by_id = {str(content["content_id"]): content for content in contents}
        selected = [by_id[content_id] for content_id in ids if content_id in by_id]
        if not selected:
            selected = contents
        selected = self._backfill_missing_types(selected, contents, query_text)
        return ChromaSearchResult(
            selected,
            {
                "backend": "chroma",
                "available": True,
                "collection": self._collection_name(),
                "input_count": len(contents),
                "candidate_count": len(selected),
                "upserted": upserted,
                "embedding_backend": self.embedding_service.backend,
            },
        )

    @staticmethod
    def _backfill_missing_types(
        selected: list[dict[str, Any]],
        contents: list[dict[str, Any]],
        query_text: str,
    ) -> list[dict[str, Any]]:
        target_count = max(config.MAX_PER_TYPE * 6, config.DEFAULT_PER_TYPE * 6)
        query_tokens = tokenize(query_text)
        query_lower = query_text.lower()
        selected_ids = {str(content["content_id"]) for content in selected}
        balanced = selected[:]
        content_types = sorted({str(content.get("content_type", "")) for content in contents})
        for content_type in content_types:
            current_count = sum(
                1 for content in balanced if content.get("content_type") == content_type
            )
            if current_count >= target_count:
                continue
            pool = [
                content
                for content in contents
                if content.get("content_type") == content_type
                and str(content["content_id"]) not in selected_ids
            ]
            ranked_pool = sorted(
                pool,
                key=lambda content: ChromaCandidateStore._backfill_score(
                    content, query_tokens, query_lower
                ),
                reverse=True,
            )
            needed = min(target_count - current_count, len(ranked_pool))
            for content in ranked_pool[:needed]:
                balanced.append(content)
                selected_ids.add(str(content["content_id"]))
        return balanced

    @staticmethod
    def _backfill_score(
        content: dict[str, Any],
        query_tokens: list[str],
        query_lower: str,
    ) -> float:
        tags = (
            list(content.get("emotion_tags") or [])
            + list(content.get("topic_tags") or [])
            + list(content.get("mood_tags") or [])
            + list(content.get("recommendation_roles") or [])
        )
        fields = [
            str(content.get("title", "")),
            str(content.get("genre", "")),
            str(content.get("summary", ""))[:700],
            " ".join(str(tag) for tag in tags),
        ]
        search_text = " ".join(fields)
        score = weighted_overlap(query_tokens, tokenize(search_text))
        for tag in tags:
            tag_text = str(tag).strip().lower()
            if tag_text and tag_text in query_lower:
                score += 0.18
        genre = str(content.get("genre", "")).lower()
        if any(cue in query_lower for cue in ["우주", "과학", "천문", "sf"]):
            if any(cue.lower() in search_text.lower() for cue in ["우주", "과학", "SF", "미래"]):
                score += 0.35
        if "코미디" in query_lower and "코미디" in genre:
            score += 0.25
        if "드라마" in query_lower and "드라마" in genre:
            score += 0.18
        return score

    def _get_collection(self) -> Any | None:
        if self._available is False:
            return None
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore

            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            self._collection = self._client.get_or_create_collection(
                self._collection_name(),
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            return self._collection
        except Exception:
            self._available = False
            return None

    def _sync(self, collection: Any, contents: list[dict[str, Any]]) -> int:
        ids = [str(content["content_id"]) for content in contents]
        texts = [str(content["embedding_text"]) for content in contents]
        hashes = [str(stable_hash(text)) for text in texts]
        existing = collection.get(ids=ids, include=["metadatas"])
        existing_hashes = {
            content_id: str(metadata.get("embedding_hash", ""))
            for content_id, metadata in zip(
                existing.get("ids", []),
                existing.get("metadatas", []),
            )
            if metadata
        }
        changed_indexes = [
            index
            for index, content_id in enumerate(ids)
            if existing_hashes.get(content_id) != hashes[index]
        ]
        if not changed_indexes:
            return 0
        changed_texts = [texts[index] for index in changed_indexes]
        embeddings = self.embedding_service.encode(changed_texts).vectors
        collection.upsert(
            ids=[ids[index] for index in changed_indexes],
            embeddings=[vector.tolist() for vector in embeddings],
            documents=changed_texts,
            metadatas=[
                {
                    "content_type": str(contents[index]["content_type"]),
                    "title": str(contents[index]["title"]),
                    "embedding_hash": hashes[index],
                    "embedding_backend": self.embedding_service.backend,
                }
                for index in changed_indexes
            ],
        )
        return len(changed_indexes)

    def _collection_name(self) -> str:
        backend_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", self.embedding_service.backend)
        return f"{config.CHROMA_COLLECTION}_{backend_slug}"[:63]
