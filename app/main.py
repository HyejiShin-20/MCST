from __future__ import annotations

import base64
import binascii
import importlib.util
from typing import Any

from app.controllers import ApiError, get_controller


controller = get_controller()

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    from app import config

    class DiaryRequest(BaseModel):
        user_id: int = 1
        input_type: str = "text"
        text: str

    class RecommendationRequest(BaseModel):
        entry_id: int
        content_types: list[str] = Field(default_factory=lambda: list(config.DEFAULT_CONTENT_TYPES))
        per_type: int = 4

    class FeedbackRequest(BaseModel):
        user_id: int | None = None
        entry_id: int | None = None
        log_id: int | None = None
        content_id: str
        signal: str
        rating: float | None = None
        note: str = ""

    class MediaBase64Request(BaseModel):
        user_id: int = 1
        input_type: str
        filename: str = ""
        mime_type: str = ""
        data_base64: str
        provider: str | None = None

    class LinkTranscriptionRequest(BaseModel):
        user_id: int = 1
        url: str

    class TranscriptionDiaryRequest(BaseModel):
        user_id: int = 1
        media_id: int | None = None
        selected_text: str = ""
        context_text: str = ""
        media_items: list[dict[str, Any]] = Field(default_factory=list)
        content_types: list[str] = Field(default_factory=lambda: list(config.DEFAULT_CONTENT_TYPES))
        per_type: int = 4
        return_recommendations: bool = True

    class UserRequest(BaseModel):
        username: str = "demo"
        display_name: str = ""

    class RepositorySaveRequest(BaseModel):
        user_id: int = 1
        content_id: str
        content_type: str
        title: str = ""
        creator: str = ""
        genre: str = ""
        source: str = ""
        entry_id: int | None = None
        log_id: int | None = None
        recommendation_role: str = ""
        reason: str = ""
        score_percent: int = 0
        display_tags: list[str] = Field(default_factory=list)
        status: str = "planned"
        note: str = ""
        snapshot: dict[str, Any] = Field(default_factory=dict)

    class RepositoryUpdateRequest(BaseModel):
        user_id: int = 1
        status: str | None = None
        note: str | None = None

    app = FastAPI(
        title="Emotion Daily Culture Recommender API",
        version="0.1.0",
        description="감정·일상 기반 문화 콘텐츠 추천 MVP 백엔드",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def handle(callback: Any) -> Any:
        try:
            return callback()
        except ApiError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    def decode_base64_media(data_base64: str) -> bytes:
        try:
            return base64.b64decode(data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="data_base64 is invalid") from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return controller.health()

    @app.post("/api/users")
    def login_or_create_user(payload: UserRequest) -> dict[str, Any]:
        return handle(lambda: controller.login_or_create_user(payload.model_dump()))

    @app.post("/api/users/login")
    def login_user(payload: UserRequest) -> dict[str, Any]:
        return handle(lambda: controller.login_or_create_user(payload.model_dump()))

    @app.get("/api/users/{user_id}")
    def get_user(user_id: int) -> dict[str, Any]:
        return handle(lambda: controller.get_user(user_id))

    @app.post("/api/diary")
    def create_diary(payload: DiaryRequest) -> dict[str, Any]:
        return handle(lambda: controller.create_diary(payload.model_dump()))

    @app.post("/api/media/transcribe-base64")
    def transcribe_media_base64(payload: MediaBase64Request) -> dict[str, Any]:
        content = decode_base64_media(payload.data_base64)
        return handle(
            lambda: controller.transcribe_media_bytes(
                user_id=payload.user_id,
                input_type=payload.input_type,
                content=content,
                filename=payload.filename,
                mime_type=payload.mime_type,
                provider=payload.provider,
            )
        )

    @app.post("/api/link/transcribe")
    def transcribe_link(payload: LinkTranscriptionRequest) -> dict[str, Any]:
        return handle(
            lambda: controller.transcribe_link(
                user_id=payload.user_id,
                url=payload.url,
            )
        )

    if importlib.util.find_spec("multipart") is not None:
        from fastapi import File, Form, UploadFile

        @app.post("/api/media/transcribe")
        async def transcribe_media(
            input_type: str = Form(...),
            user_id: int = Form(1),
            provider: str | None = Form(None),
            file: UploadFile = File(...),
        ) -> dict[str, Any]:
            content = await file.read()
            return handle(
                lambda: controller.transcribe_media_bytes(
                    user_id=user_id,
                    input_type=input_type,
                    content=content,
                    filename=file.filename or "",
                    mime_type=file.content_type or "",
                    provider=provider,
                )
            )
    else:

        @app.post("/api/media/transcribe")
        def transcribe_media_missing_dependency() -> dict[str, Any]:
            raise HTTPException(
                status_code=503,
                detail="python-multipart is required for multipart media uploads.",
            )

    @app.post("/api/diary/from-transcription")
    def create_diary_from_transcription(
        payload: TranscriptionDiaryRequest,
    ) -> dict[str, Any]:
        return handle(lambda: controller.create_diary_from_transcription(payload.model_dump()))

    @app.get("/api/media/transcriptions/{media_id}")
    def get_media_transcription(media_id: int) -> dict[str, Any]:
        return handle(lambda: controller.get_media_transcription(media_id))

    @app.delete("/api/media/transcriptions/{media_id}")
    def delete_media_transcription(media_id: int) -> dict[str, Any]:
        return handle(lambda: controller.delete_media_transcription(media_id))

    @app.post("/api/recommend")
    def recommend(payload: RecommendationRequest) -> dict[str, Any]:
        return handle(lambda: controller.recommend(payload.model_dump()))

    @app.get("/api/entries")
    def list_entries(user_id: int = 1, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        return controller.list_entries(user_id, limit)

    @app.get("/api/activity")
    def get_activity(user_id: int = 1, limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        return controller.get_activity(user_id, limit)

    @app.get("/api/repository")
    def list_repository(
        user_id: int = 1,
        content_type: str | None = None,
        status: str | None = None,
        limit: int = Query(default=200, ge=1, le=500),
    ) -> dict[str, Any]:
        return handle(lambda: controller.list_repository(user_id, content_type, status, limit))

    @app.post("/api/repository/save")
    def save_repository_item(payload: RepositorySaveRequest) -> dict[str, Any]:
        return handle(lambda: controller.save_repository_item(payload.model_dump()))

    @app.patch("/api/repository/{saved_id}")
    def update_repository_item(
        saved_id: int,
        payload: RepositoryUpdateRequest,
    ) -> dict[str, Any]:
        return handle(lambda: controller.update_repository_item(saved_id, payload.model_dump()))

    @app.delete("/api/repository/{saved_id}")
    def delete_repository_item(saved_id: int, user_id: int = 1) -> dict[str, Any]:
        return handle(lambda: controller.delete_repository_item(saved_id, user_id))

    @app.get("/api/taste-analysis")
    def analyze_saved_taste(user_id: int = 1, use_openai: bool = True) -> dict[str, Any]:
        return handle(lambda: controller.analyze_saved_taste(user_id, use_openai))

    @app.get("/api/profile/{user_id}")
    def get_profile(user_id: int) -> dict[str, Any]:
        return controller.get_profile(user_id)

    @app.post("/api/feedback")
    def add_feedback(payload: FeedbackRequest) -> dict[str, Any]:
        return handle(lambda: controller.add_feedback(payload.model_dump()))

    @app.get("/api/feedback")
    def list_feedback(user_id: int = 1, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return controller.list_feedback(user_id, limit)

    @app.get("/api/content")
    def list_contents(content_type: str | None = None) -> dict[str, Any]:
        return controller.list_contents(content_type)

    @app.get("/api/content/{content_id}")
    def get_content(content_id: str) -> dict[str, Any]:
        return handle(lambda: controller.get_content(content_id))

    @app.delete("/api/diary/{entry_id}")
    def delete_entry(entry_id: int) -> dict[str, Any]:
        return handle(lambda: controller.delete_entry(entry_id))

    @app.post("/api/users/{user_id}/memory/reset")
    def reset_memory(user_id: int) -> dict[str, Any]:
        return controller.reset_memory(user_id)

    @app.delete("/api/users/{user_id}/data")
    def delete_user_data(user_id: int) -> dict[str, Any]:
        return controller.delete_user_data(user_id)

except Exception:
    from app.mini_asgi import MiniASGI

    app = MiniASGI(controller)
