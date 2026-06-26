from __future__ import annotations

import importlib.util
import json
from collections import Counter
from typing import Any

from app import config
from app.database import Database
from app.services.analyzer import EmotionAnalyzer
from app.services.memory_service import MemoryService
from app.services.media_transcription import (
    MediaTranscriptionError,
    MediaTranscriptionService,
)
from app.services.link_transcription import LinkTranscriptionError, LinkTranscriptionService
from app.services.recommendation.content_profile import build_content_affect_profile
from app.services.recommender import Recommender
from app.services.text_utils import normalize_text

VALID_FEEDBACK_SIGNALS = {"helpful", "not_helpful", "saved", "skipped", "liked", "disliked", "completed"}
VALID_SAVED_STATUSES = {"planned", "completed", "archived"}


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


class ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class BackendController:
    def __init__(
        self,
        database: Database | None = None,
        analyzer: EmotionAnalyzer | None = None,
        memory_service: MemoryService | None = None,
        recommender: Recommender | None = None,
        media_transcriber: MediaTranscriptionService | None = None,
        link_transcriber: LinkTranscriptionService | None = None,
    ) -> None:
        self.database = database or Database()
        self.analyzer = analyzer or EmotionAnalyzer()
        self.memory_service = memory_service or MemoryService()
        self.recommender = recommender or Recommender()
        self.media_transcriber = media_transcriber or MediaTranscriptionService()
        self.link_transcriber = link_transcriber or LinkTranscriptionService()
        self.database.initialize()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "database": str(self.database.db_path),
            "analysis_backend": self.analyzer.backend,
            "embedding_backend": self.recommender.embedding_service.backend,
            "ocr_provider": config.OCR_PROVIDER,
            "stt_provider": config.STT_PROVIDER,
            "transcription_review_required": config.TRANSCRIPTION_REVIEW_REQUIRED,
            "multimodal_dependencies": {
                "google_cloud_vision": _module_available("google.cloud.vision"),
                "openai": _module_available("openai"),
                "python_multipart": _module_available("multipart"),
                "google_credentials_configured": bool(config.GOOGLE_APPLICATION_CREDENTIALS),
            },
        }

    def login_or_create_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        username = normalize_text(str(payload.get("username") or "demo"))
        display_name = normalize_text(str(payload.get("display_name") or username))
        if not username:
            raise ApiError(400, "username is required")
        user = self.database.login_or_create_user(username, display_name)
        return {"user": user}

    def get_user(self, user_id: int) -> dict[str, Any]:
        user = self.database.get_user(user_id)
        if not user:
            raise ApiError(404, "user not found")
        self.database.touch_user(user_id)
        return {"user": user}

    def create_diary(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = int(payload.get("user_id") or config.DEFAULT_USER_ID)
        input_type = str(payload.get("input_type") or "text")
        text = normalize_text(str(payload.get("text") or payload.get("raw_text") or ""))
        if input_type != "text":
            raise ApiError(400, "MVP에서는 input_type='text'만 실제 분석합니다.")
        if not text:
            raise ApiError(400, "text는 비어 있을 수 없습니다.")

        entry_id = self.database.add_diary_entry(user_id, input_type, text)
        analysis = self.analyzer.analyze(text)
        self.database.save_analysis(entry_id, analysis)
        self._refresh_profile(user_id)
        return {"entry_id": entry_id, "user_id": user_id, "analysis": analysis}

    def transcribe_media_bytes(
        self,
        *,
        user_id: int,
        input_type: str,
        content: bytes,
        filename: str = "",
        mime_type: str = "",
        provider: str | None = None,
    ) -> dict[str, Any]:
        try:
            result = self.media_transcriber.transcribe_bytes(
                input_type=input_type,
                content=content,
                filename=filename,
                mime_type=mime_type,
                provider=provider,
            )
        except MediaTranscriptionError as exc:
            raise ApiError(exc.status_code, exc.message) from exc

        media_id = self.database.save_media_transcription(
            user_id=user_id,
            input_type=result["input_type"],
            provider=result["provider"],
            model_name=result["model_name"],
            source_filename=result["source_filename"],
            source_mime_type=result["source_mime_type"],
            source_size_bytes=result["source_size_bytes"],
            file_sha256=result["file_sha256"],
            candidates=result["candidates"],
            selected_text=result["selected_text"],
            confidence=result["confidence"],
            needs_user_review=result["needs_user_review"],
            status=result["status"],
            metadata=result["metadata"],
        )
        return {"media_transcription": self.database.get_media_transcription(media_id)}

    def transcribe_link(self, *, user_id: int, url: str) -> dict[str, Any]:
        try:
            result = self.link_transcriber.transcribe_url(url)
        except LinkTranscriptionError as exc:
            raise ApiError(exc.status_code, exc.message) from exc

        media_id = self.database.save_media_transcription(
            user_id=user_id,
            input_type=result["input_type"],
            provider=result["provider"],
            model_name=result["model_name"],
            source_filename=result["source_filename"],
            source_mime_type=result["source_mime_type"],
            source_size_bytes=result["source_size_bytes"],
            file_sha256=result["file_sha256"],
            candidates=result["candidates"],
            selected_text=result["selected_text"],
            confidence=result["confidence"],
            needs_user_review=result["needs_user_review"],
            status=result["status"],
            metadata=result["metadata"],
        )
        return {"media_transcription": self.database.get_media_transcription(media_id)}

    def get_media_transcription(self, media_id: int) -> dict[str, Any]:
        media = self.database.get_media_transcription(media_id)
        if not media:
            raise ApiError(404, "media transcription not found")
        return {"media_transcription": media}

    def delete_media_transcription(self, media_id: int) -> dict[str, Any]:
        deleted = self.database.delete_media_transcription(media_id)
        if not deleted:
            raise ApiError(404, "media transcription not found")
        return {"deleted": True, "media_id": media_id}

    def create_diary_from_transcription(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = int(payload.get("user_id") or config.DEFAULT_USER_ID)
        media_items = self._normalize_media_items_payload(payload)
        if not media_items:
            raise ApiError(400, "media_id is required")
        media_records: list[dict[str, Any]] = []
        for media_item in media_items:
            media = self.database.get_media_transcription(media_item["media_id"])
            if not media:
                raise ApiError(404, "media transcription not found")
            if int(media["user_id"]) != user_id:
                raise ApiError(400, "user_id does not match media transcription owner")
            media_records.append({"item": media_item, "media": media})

        selected_text = normalize_text(str(payload.get("selected_text") or ""))
        context_text = normalize_text(str(payload.get("context_text") or ""))
        if not selected_text:
            selected_text = self._combined_media_selected_text(media_records)
        if not selected_text:
            raise ApiError(400, "selected_text is required before analysis")

        primary_media = media_records[0]["media"]
        composed = self._compose_multimodal_entry_text(
            context_text,
            selected_text,
            primary_media,
            prefer_selected_text=len(media_records) > 1,
        )
        # 모달리티 분리: 사용자가 직접 쓴 텍스트/OCR/STT = 1차, 이미지 캡션·분위기 = 2차.
        # 1차만 임베딩 시드(processed_text)로 써서 캡션 노이즈가 추천 쿼리에 새지 않게 한다.
        modalities = self._split_modalities(context_text, media_records)
        primary_text = modalities["primary_text"] or composed["processed_text"]
        secondary_text = modalities["secondary_text"]
        input_type = str(primary_media["input_type"])
        if len(media_records) > 1:
            input_type = "multimodal"
        media_id = int(primary_media["media_id"])
        entry_id = self.database.add_diary_entry(
            user_id=user_id,
            input_type=input_type,
            raw_text=composed["raw_text"],
            processed_text=primary_text,
            source_media_id=media_id,
        )
        analysis = self.analyzer.analyze(primary_text, secondary_text)
        self.database.save_analysis(entry_id, analysis)
        for media_record in media_records:
            item_text = self._media_item_selected_text(
                media_record["item"],
                media_record["media"],
            )
            self.database.confirm_media_transcription(
                int(media_record["media"]["media_id"]),
                item_text or selected_text,
                entry_id=entry_id,
            )
        self._refresh_profile(user_id)

        diary = {"entry_id": entry_id, "user_id": user_id, "analysis": analysis}
        media_ids = [int(record["media"]["media_id"]) for record in media_records]
        response: dict[str, Any] = {
            "media_id": media_id,
            "media_ids": media_ids,
            "selected_text": selected_text,
            "context_text": context_text,
            "raw_text": composed["raw_text"],
            "processed_text": composed["processed_text"],
            "diary": diary,
            "media_transcription": self.database.get_media_transcription(media_id),
            "media_transcriptions": [
                self.database.get_media_transcription(item_id)
                for item_id in media_ids
            ],
        }
        if payload.get("return_recommendations", True):
            response["recommendations"] = self.recommend(
                {
                    "entry_id": entry_id,
                    "content_types": payload.get("content_types")
                    or config.DEFAULT_CONTENT_TYPES,
                    "per_type": payload.get("per_type") or config.DEFAULT_PER_TYPE,
                }
            )
        return response

    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        entry_id = int(payload.get("entry_id") or 0)
        if not entry_id:
            raise ApiError(400, "entry_id가 필요합니다.")
        entry = self.database.get_entry(entry_id)
        if not entry:
            raise ApiError(404, "해당 entry_id의 일기를 찾을 수 없습니다.")
        analysis = self.database.get_analysis(entry_id)
        if not analysis:
            raise ApiError(404, "해당 entry_id의 분석 결과를 찾을 수 없습니다.")

        content_types = payload.get("content_types") or config.DEFAULT_CONTENT_TYPES
        if not isinstance(content_types, list):
            raise ApiError(400, "content_types는 배열이어야 합니다.")
        content_types = [str(item) for item in content_types if item in {"movie", "music", "book"}]
        if not content_types:
            raise ApiError(400, "content_types에는 movie, music, book 중 하나 이상이 필요합니다.")

        per_type = int(payload.get("per_type") or payload.get("top_k_per_type") or config.DEFAULT_PER_TYPE)
        per_type = max(1, min(config.MAX_PER_TYPE, per_type))

        contents = self.database.list_contents(content_types, recommendable_only=True)
        if config.HIDE_SAMPLE_CONTENT_IN_DEMO:
            real_contents = [
                content
                for content in contents
                if not str(content.get("source", "")).startswith("MVP curated sample")
            ]
            if real_contents:
                contents = real_contents
        profile = self.database.get_profile(entry["user_id"])
        recent = self.database.list_recent_analyses(entry["user_id"], limit=10)
        response = self.recommender.recommend(
            entry=entry,
            analysis=analysis,
            contents=contents,
            profile=profile,
            recent_analyses=recent,
            per_type=per_type,
        )
        response["analysis_summary"] = analysis.get("summary")
        response["log_id"] = self.database.save_recommendation_log(
            entry["user_id"], entry_id, response
        )
        return response

    def list_entries(self, user_id: int, limit: int = 50) -> dict[str, Any]:
        return {"user_id": user_id, "entries": self.database.list_entries(user_id, limit)}

    def get_activity(self, user_id: int, limit: int = 50) -> dict[str, Any]:
        user = self.database.get_user(user_id) or self.database.get_user(config.DEFAULT_USER_ID)
        entries = self.database.list_entries(user_id, limit)
        media_transcriptions = self.database.list_media_transcriptions(user_id, limit)
        recommendation_logs = self.database.list_recommendation_logs(user_id, limit)
        saved_items = self.database.list_saved_items(user_id, limit=limit)
        return {
            "user_id": user_id,
            "user": user,
            "entries": entries,
            "media_transcriptions": media_transcriptions,
            "recommendation_logs": recommendation_logs,
            "saved_items": saved_items,
            "summary": {
                "entry_count": len(entries),
                "media_count": len(media_transcriptions),
                "recommendation_count": len(recommendation_logs),
                "saved_count": len(saved_items),
            },
        }

    def save_repository_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = int(payload.get("user_id") or config.DEFAULT_USER_ID)
        content_id = normalize_text(str(payload.get("content_id") or ""))
        if not content_id:
            raise ApiError(400, "content_id is required")

        existing = self.database.get_saved_item_by_content(user_id, content_id)
        content = self.database.get_content(content_id)
        snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {}
        content_type = normalize_text(
            str(payload.get("content_type") or (content or {}).get("content_type") or "")
        )
        if content_type not in {"music", "book", "movie"}:
            raise ApiError(400, "content_type must be one of music, book, movie")

        status = normalize_text(str(payload.get("status") or "planned"))
        if status not in VALID_SAVED_STATUSES:
            raise ApiError(400, "status must be planned, completed, or archived")

        entry_id = self._optional_int(payload.get("entry_id"))
        log_id = self._optional_int(payload.get("log_id"))
        display_tags = payload.get("display_tags") if isinstance(payload.get("display_tags"), list) else []
        saved_item = self.database.save_collection_item(
            user_id=user_id,
            content_id=content_id,
            content_type=content_type,
            title=normalize_text(str(payload.get("title") or (content or {}).get("title") or "")),
            creator=normalize_text(str(payload.get("creator") or (content or {}).get("creator") or "")),
            genre=normalize_text(str(payload.get("genre") or (content or {}).get("genre") or "")),
            source=normalize_text(str(payload.get("source") or (content or {}).get("source") or "")),
            entry_id=entry_id,
            log_id=log_id,
            recommendation_role=normalize_text(str(payload.get("recommendation_role") or "")),
            reason=normalize_text(str(payload.get("reason") or "")),
            score_percent=int(payload.get("score_percent") or 0),
            display_tags=[normalize_text(str(tag)) for tag in display_tags if str(tag).strip()][:8],
            status=status,
            note=normalize_text(str(payload.get("note") or "")),
            snapshot=snapshot,
        )

        if content and not existing:
            self.database.save_feedback(
                user_id=user_id,
                entry_id=entry_id,
                log_id=log_id,
                content_id=content_id,
                content_type=content_type,
                signal="saved",
                rating=None,
                note="saved to repository",
                metadata={"source": "repository", "saved_item_id": saved_item["saved_id"]},
            )
            self._refresh_profile(user_id)

        return {"user_id": user_id, "saved_item": saved_item}

    def list_repository(
        self,
        user_id: int,
        content_type: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        if content_type not in {None, "all", "music", "book", "movie"}:
            raise ApiError(400, "content_type must be all, music, book, or movie")
        if status not in {None, "all", *VALID_SAVED_STATUSES}:
            raise ApiError(400, "status must be all, planned, completed, or archived")
        items = self.database.list_saved_items(
            user_id,
            content_type=content_type,
            status=status,
            limit=limit,
        )
        counts = {
            "total": len(items),
            "planned": sum(1 for item in items if item["status"] == "planned"),
            "completed": sum(1 for item in items if item["status"] == "completed"),
            "archived": sum(1 for item in items if item["status"] == "archived"),
        }
        return {"user_id": user_id, "saved_items": items, "counts": counts}

    def update_repository_item(self, saved_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = int(payload.get("user_id") or config.DEFAULT_USER_ID)
        status = payload.get("status")
        note = payload.get("note")
        if status is not None:
            status = normalize_text(str(status))
            if status not in VALID_SAVED_STATUSES:
                raise ApiError(400, "status must be planned, completed, or archived")
        if note is not None:
            note = normalize_text(str(note))
        saved_item = self.database.update_saved_item(
            saved_id=saved_id,
            user_id=user_id,
            status=status,
            note=note,
        )
        if not saved_item:
            raise ApiError(404, "saved item not found")
        return {"user_id": user_id, "saved_item": saved_item}

    def delete_repository_item(self, saved_id: int, user_id: int) -> dict[str, Any]:
        deleted = self.database.delete_saved_item(saved_id, user_id)
        if not deleted:
            raise ApiError(404, "saved item not found")
        self._refresh_profile(user_id)
        return {"user_id": user_id, "saved_id": saved_id, "deleted": True}

    def analyze_saved_taste(self, user_id: int, use_openai: bool = True) -> dict[str, Any]:
        saved_items = self.database.list_saved_items(user_id, limit=500)
        music_count = sum(1 for item in saved_items if item["content_type"] == "music")
        book_count = sum(1 for item in saved_items if item["content_type"] == "book")
        movie_count = sum(1 for item in saved_items if item["content_type"] == "movie")
        total = len(saved_items)

        tag_counter: Counter[str] = Counter()
        role_counter: Counter[str] = Counter()
        creator_counter: Counter[str] = Counter()
        valence_values: list[float] = []
        energy_values: list[float] = []
        for item in saved_items:
            for tag in item.get("display_tags", []):
                if tag:
                    tag_counter[str(tag)] += 1
            if item.get("creator"):
                creator_counter[str(item["creator"])] += 1
            role = item.get("recommendation_role") or item.get("snapshot", {}).get("recommendation_role")
            if role:
                role_counter[str(role)] += 1
            affect = item.get("snapshot", {}).get("content_affect_profile", {})
            if isinstance(affect, dict):
                valence = self._optional_float(affect.get("valence"))
                energy = self._optional_float(affect.get("energy"))
                if valence is not None:
                    valence_values.append(valence)
                if energy is not None:
                    energy_values.append(energy)

        type_counts = {"music": music_count, "book": book_count, "movie": movie_count}
        content_ratio = {
            content_type: {
                "count": count,
                "percent": round((count / total) * 100) if total else 0,
            }
            for content_type, count in type_counts.items()
            if count or content_type in {"music", "book"}
        }
        top_tags = [tag for tag, _ in tag_counter.most_common(8)]
        top_roles = [role for role, _ in role_counter.most_common(5)]
        top_creators = [creator for creator, _ in creator_counter.most_common(5)]
        avg_valence = round(sum(valence_values) / len(valence_values), 3) if valence_values else None
        avg_energy = round(sum(energy_values) / len(energy_values), 3) if energy_values else None
        dominant_type = max(type_counts, key=type_counts.get) if total else "none"

        taste_summary = {
            "headline": self._taste_headline(top_tags, dominant_type, total),
            "top_tags": top_tags,
            "top_creators": top_creators,
            "saved_count": total,
        }
        emotion_patterns = {
            "dominant_moods": top_tags[:5],
            "recommendation_roles": top_roles,
            "average_valence": avg_valence,
            "average_energy": avg_energy,
            "summary": self._emotion_pattern_summary(top_tags, top_roles, avg_valence, avg_energy),
        }
        recommendation_direction = {
            "primary_content_type": dominant_type,
            "focus_keywords": top_tags[:6],
            "next_direction": self._next_direction(dominant_type, top_tags, top_roles),
            "balance_note": self._balance_note(type_counts),
        }
        deterministic_report = self._deterministic_taste_report(
            taste_summary,
            emotion_patterns,
            content_ratio,
            recommendation_direction,
        )
        generated_report = {
            "source": "deterministic",
            "text": deterministic_report,
            "error": "",
        }
        if use_openai and total:
            openai_text, openai_error = self._generate_taste_report(
                {
                    "taste_summary": taste_summary,
                    "emotion_patterns": emotion_patterns,
                    "content_ratio": content_ratio,
                    "recommendation_direction": recommendation_direction,
                }
            )
            if openai_text:
                generated_report = {
                    "source": "openai",
                    "model": config.OPENAI_EMOTION_MODEL,
                    "text": openai_text,
                    "fallback_text": deterministic_report,
                    "error": "",
                }
            elif openai_error:
                generated_report["error"] = openai_error

        return {
            "user_id": user_id,
            "basis": "saved_items_only",
            "saved_count": total,
            "taste_summary": taste_summary,
            "emotion_patterns": emotion_patterns,
            "content_ratio": content_ratio,
            "recommendation_direction": recommendation_direction,
            "generated_report": generated_report,
            "saved_items_preview": saved_items[:12],
        }

    def get_profile(self, user_id: int) -> dict[str, Any]:
        profile = self.database.get_profile(user_id)
        if not profile:
            return {"user_id": user_id, "profile": None}
        return {"user_id": user_id, "profile": profile}

    def add_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        content_id = normalize_text(str(payload.get("content_id") or ""))
        signal = normalize_text(str(payload.get("signal") or ""))
        if not content_id:
            raise ApiError(400, "content_id가 필요합니다.")
        if signal not in VALID_FEEDBACK_SIGNALS:
            raise ApiError(
                400,
                "signal은 helpful, not_helpful, saved, skipped, liked, disliked, completed 중 하나여야 합니다.",
            )

        content = self.database.get_content(content_id)
        if not content:
            raise ApiError(404, "해당 content_id의 콘텐츠를 찾을 수 없습니다.")

        entry_id = self._optional_int(payload.get("entry_id"))
        entry = self.database.get_entry(entry_id) if entry_id else None
        if entry_id and not entry:
            raise ApiError(404, "해당 entry_id의 일기를 찾을 수 없습니다.")

        payload_user_id = self._optional_int(payload.get("user_id"))
        user_id = int(entry["user_id"]) if entry else int(payload_user_id or config.DEFAULT_USER_ID)
        if payload_user_id and payload_user_id != user_id:
            raise ApiError(400, "user_id와 entry_id의 소유자가 일치하지 않습니다.")

        log_id = self._optional_int(payload.get("log_id"))
        rating = self._optional_float(payload.get("rating"))
        if rating is not None:
            rating = max(1.0, min(5.0, rating))
        note = normalize_text(str(payload.get("note") or ""))
        content_snapshot = {
            "content_id": content["content_id"],
            "content_type": content["content_type"],
            "title": content["title"],
            "creator": content["creator"],
            "genre": content["genre"],
            "summary": content["summary"],
            "emotion_tags": content["emotion_tags"],
            "topic_tags": content["topic_tags"],
        }
        affect_profile = build_content_affect_profile(content)
        metadata = {
            "signal": signal,
            "content": content_snapshot,
            "content_affect_profile": affect_profile,
        }
        feedback_id = self.database.save_feedback(
            user_id=user_id,
            entry_id=entry_id,
            log_id=log_id,
            content_id=content_id,
            content_type=content["content_type"],
            signal=signal,
            rating=rating,
            note=note,
            metadata=metadata,
        )
        self._refresh_profile(user_id)
        return {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "entry_id": entry_id,
            "content_id": content_id,
            "signal": signal,
            "rating": rating,
            "profile": self.database.get_profile(user_id),
        }

    def list_feedback(self, user_id: int, limit: int = 100) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "feedback_events": self.database.list_feedback_events(user_id, limit),
        }

    def list_contents(self, content_type: str | None = None) -> dict[str, Any]:
        content_types = [content_type] if content_type in {"movie", "music", "book"} else None
        return {"contents": self.database.list_contents(content_types)}

    def get_content(self, content_id: str) -> dict[str, Any]:
        content = self.database.get_content(content_id)
        if not content:
            raise ApiError(404, "content not found")
        return {"content": content}

    def delete_entry(self, entry_id: int) -> dict[str, Any]:
        entry = self.database.get_entry(entry_id)
        deleted = self.database.delete_entry(entry_id)
        if not deleted:
            raise ApiError(404, "삭제할 일기를 찾을 수 없습니다.")
        if entry:
            self._refresh_profile(int(entry["user_id"]))
        return {"deleted": True, "entry_id": entry_id}

    def reset_memory(self, user_id: int) -> dict[str, Any]:
        self.database.reset_memory(user_id)
        return {
            "user_id": user_id,
            "memory_reset": True,
            "note": "장기 기억 프로필과 추천 피드백 기억을 초기화했습니다. 원문 일기와 분석 결과는 유지됩니다.",
        }

    def delete_user_data(self, user_id: int) -> dict[str, Any]:
        self.database.delete_user_data(user_id)
        return {
            "user_id": user_id,
            "deleted": True,
            "note": "원문 일기, 분석 결과, 추천 로그, 장기 기억 프로필을 삭제했습니다.",
        }

    def _refresh_profile(self, user_id: int) -> None:
        analyses = self.database.list_recent_analyses(user_id, limit=100)
        feedback_events = self.database.list_feedback_events(user_id, limit=200)
        if analyses or feedback_events:
            profile = self.memory_service.build_profile(user_id, analyses, feedback_events)
            self.database.upsert_profile(user_id, profile)
        else:
            self.database.reset_memory(user_id)

    @staticmethod
    def _normalize_media_items_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        media_items: list[dict[str, Any]] = []
        seen: set[int] = set()

        def add_item(media_id: int, selected_text: str = "") -> None:
            if not media_id or media_id in seen:
                return
            seen.add(media_id)
            media_items.append(
                {
                    "media_id": media_id,
                    "selected_text": normalize_text(selected_text),
                }
            )

        raw_items = payload.get("media_items")
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                try:
                    media_id = int(raw_item.get("media_id") or 0)
                except (TypeError, ValueError):
                    continue
                add_item(media_id, str(raw_item.get("selected_text") or ""))

        try:
            primary_media_id = int(payload.get("media_id") or 0)
        except (TypeError, ValueError):
            primary_media_id = 0
        if primary_media_id and primary_media_id not in seen:
            media_items.insert(
                0,
                {
                    "media_id": primary_media_id,
                    "selected_text": "",
                },
            )
        return media_items

    @staticmethod
    def _media_item_selected_text(
        media_item: dict[str, Any],
        media: dict[str, Any],
    ) -> str:
        selected_text = normalize_text(str(media_item.get("selected_text") or ""))
        if selected_text:
            return selected_text
        selected_text = normalize_text(str(media.get("selected_text") or ""))
        if selected_text:
            return selected_text
        candidates = media.get("candidates") or []
        if candidates:
            return normalize_text(str(candidates[0].get("text") or ""))
        return ""

    def _combined_media_selected_text(self, media_records: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for media_record in media_records:
            media = media_record["media"]
            selected_text = self._media_item_selected_text(
                media_record["item"],
                media,
            )
            if not selected_text:
                continue
            media_id = media.get("media_id")
            input_type = media.get("input_type") or "media"
            parts.append(f"[{input_type} #{media_id}]\n{selected_text}")
        return "\n\n".join(parts).strip()

    def _split_modalities(
        self,
        context_text: str,
        media_records: list[dict[str, Any]],
    ) -> dict[str, str]:
        """입력을 1차(사용자 작성/OCR/STT)와 2차(이미지 AI 추론)로 분리한다.

        - 직접 작성 기록, 손글씨 OCR, 음성/링크 변환 = 1차(감정 분석 주신호, 임베딩 시드).
        - 이미지 캡션/장면/분위기 = 2차(낮은 비중 보조 신호, 임베딩 시드에서 제외).
        1차가 비어 있으면(예: 이미지 캡션만 있는 기록) 2차를 1차로 승격한다.
        """
        primary_parts: list[str] = []
        secondary_parts: list[str] = []
        if context_text:
            primary_parts.append(context_text)
        for record in media_records:
            media = record["media"]
            metadata = media.get("metadata") or {}
            input_type = str(media.get("input_type") or "")
            selected = self._media_item_selected_text(record["item"], media)
            ocr_text = normalize_text(str(metadata.get("ocr_text") or ""))
            caption = normalize_text(str(metadata.get("image_caption") or ""))
            scene = normalize_text(str(metadata.get("image_scene") or ""))
            atmosphere = normalize_text(str(metadata.get("image_atmosphere") or ""))
            if input_type == "image":
                if ocr_text:
                    primary_parts.append(ocr_text)
                visual = " ".join(part for part in [caption, scene, atmosphere] if part)
                if visual:
                    secondary_parts.append(visual)
                elif not ocr_text and selected:
                    # 구조화 메타데이터가 없으면 selected_text(대개 캡션)는 2차로 취급
                    secondary_parts.append(selected)
            else:
                if selected:
                    primary_parts.append(selected)
        primary_text = "\n".join(
            dict.fromkeys(part for part in primary_parts if part)
        ).strip()
        secondary_text = "\n".join(
            dict.fromkeys(part for part in secondary_parts if part)
        ).strip()
        if not primary_text:
            primary_text = secondary_text
            secondary_text = ""
        return {"primary_text": primary_text, "secondary_text": secondary_text}

    @staticmethod
    def _compose_multimodal_entry_text(
        context_text: str,
        selected_text: str,
        media: dict[str, Any],
        prefer_selected_text: bool = False,
    ) -> dict[str, str]:
        metadata = media.get("metadata") or {}
        input_type = str(media.get("input_type") or "")
        ocr_text = normalize_text(str(metadata.get("ocr_text") or ""))
        caption = normalize_text(str(metadata.get("image_caption") or ""))
        scene = normalize_text(str(metadata.get("image_scene") or ""))
        atmosphere = normalize_text(str(metadata.get("image_atmosphere") or ""))
        media_primary = selected_text if prefer_selected_text else (ocr_text or selected_text)
        parts: list[str] = []
        if context_text:
            parts.append(f"직접 작성 기록:\n{context_text}")
        if selected_text:
            parts.append(f"첨부 변환 텍스트:\n{selected_text}")
        raw_text = "\n\n".join(parts) or selected_text

        weighted_parts: list[str] = []
        if context_text:
            weighted_parts.extend(
                [
                    "직접 작성 기록(우선 분석):",
                    context_text,
                    context_text,
                ]
            )
        if media_primary:
            label = "OCR/음성/링크 변환 텍스트(우선 분석)" if input_type != "image" or ocr_text else "첨부 변환 텍스트(참고)"
            weighted_parts.extend([label + ":", media_primary])
        if input_type == "image" and not ocr_text:
            visual_context = " ".join(part for part in [caption, scene, atmosphere] if part)
            if visual_context:
                weighted_parts.extend(["이미지 장면/분위기 참고(낮은 비중):", visual_context])
        processed_text = "\n".join(part for part in weighted_parts if part).strip()
        return {
            "raw_text": raw_text.strip() or selected_text,
            "processed_text": processed_text or raw_text.strip() or selected_text,
        }

    @staticmethod
    def _taste_headline(top_tags: list[str], dominant_type: str, total: int) -> str:
        if not total:
            return "아직 저장된 취향 데이터가 없습니다."
        type_label = {"music": "음악", "book": "도서", "movie": "영화"}.get(dominant_type, "콘텐츠")
        if top_tags:
            return f"{type_label} 중심으로 {', '.join(top_tags[:3])} 결의 콘텐츠를 자주 저장했습니다."
        return f"{type_label} 저장 비중이 가장 높습니다."

    @staticmethod
    def _emotion_pattern_summary(
        top_tags: list[str],
        top_roles: list[str],
        avg_valence: float | None,
        avg_energy: float | None,
    ) -> str:
        if not top_tags and not top_roles:
            return "저장 항목이 부족해 감정 패턴을 아직 확정하기 어렵습니다."
        valence_text = "중립적"
        if avg_valence is not None:
            if avg_valence >= 0.62:
                valence_text = "긍정적"
            elif avg_valence <= 0.38:
                valence_text = "차분하거나 낮은 정서"
        energy_text = "중간 에너지"
        if avg_energy is not None:
            if avg_energy >= 0.62:
                energy_text = "활동적인 에너지"
            elif avg_energy <= 0.38:
                energy_text = "낮고 안정적인 에너지"
        tag_text = ", ".join(top_tags[:3]) if top_tags else "저장 선호"
        role_text = ", ".join(top_roles[:2]) if top_roles else "개인화 추천"
        return f"{tag_text} 태그가 반복되며, {valence_text} 분위기와 {energy_text} 쪽으로 기울어 있습니다. 추천 역할은 {role_text} 흐름이 두드러집니다."

    @staticmethod
    def _next_direction(dominant_type: str, top_tags: list[str], top_roles: list[str]) -> str:
        type_label = {"music": "바로 듣기 좋은 음악", "book": "천천히 읽을 도서", "movie": "감상형 영상"}.get(
            dominant_type,
            "문화 콘텐츠",
        )
        tag_text = ", ".join(top_tags[:3]) if top_tags else "최근 저장한 분위기"
        role_text = ", ".join(top_roles[:2]) if top_roles else "회복과 몰입"
        return f"다음 추천은 {tag_text}를 유지하되, {role_text} 목적에 맞는 {type_label}을 우선 탐색하는 것이 좋습니다."

    @staticmethod
    def _balance_note(type_counts: dict[str, int]) -> str:
        music = type_counts.get("music", 0)
        book = type_counts.get("book", 0)
        if music > book * 2 and music >= 4:
            return "음악 저장이 많으므로 비슷한 감정선의 도서를 보조 추천하면 확장성이 좋아집니다."
        if book > music * 2 and book >= 4:
            return "도서 저장이 많으므로 같은 주제의 음악을 함께 추천하면 데모 체감이 좋아집니다."
        return "음악과 도서 비중이 크게 치우치지 않아 분야별 추천 균형을 유지하기 좋습니다."

    @staticmethod
    def _deterministic_taste_report(
        taste_summary: dict[str, Any],
        emotion_patterns: dict[str, Any],
        content_ratio: dict[str, Any],
        recommendation_direction: dict[str, Any],
    ) -> str:
        ratio_text = ", ".join(
            f"{key}:{value['percent']}%"
            for key, value in content_ratio.items()
        ) or "저장 데이터 없음"
        return (
            f"{taste_summary['headline']} "
            f"감정 패턴은 {emotion_patterns['summary']} "
            f"분야 비율은 {ratio_text}입니다. "
            f"{recommendation_direction['next_direction']} "
            f"{recommendation_direction['balance_note']}"
        )

    @staticmethod
    def _generate_taste_report(payload: dict[str, Any]) -> tuple[str, str]:
        if not config.OPENAI_API_KEY:
            return "", "OPENAI_API_KEY is not configured"
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(
                api_key=config.OPENAI_API_KEY or None,
                timeout=config.OPENAI_ANALYSIS_TIMEOUT_SECONDS,
            )
            response = client.responses.create(
                model=config.OPENAI_EMOTION_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You write concise Korean taste-analysis reports "
                                    "for a culture recommendation demo. Use only the "
                                    "provided saved-item analysis. Do not invent titles."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "다음 JSON을 바탕으로 취향 요약, 감정 패턴, 분야 비율, "
                                    "다음 추천 방향을 4문장 이내로 자연스럽게 설명해줘.\n"
                                    f"{json.dumps(payload, ensure_ascii=False)}"
                                ),
                            }
                        ],
                    },
                ],
            )
            return str(getattr(response, "output_text", "") or "").strip(), ""
        except Exception as exc:
            return "", str(exc)[:300]

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        return float(value)


_controller: BackendController | None = None


def get_controller() -> BackendController:
    global _controller
    if _controller is None:
        _controller = BackendController()
    return _controller
