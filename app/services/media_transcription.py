from __future__ import annotations

import hashlib
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any

from app import config
from app.services.text_utils import normalize_text


IMAGE_MIME_PREFIX = "image/"
AUDIO_MIME_PREFIX = "audio/"
DEFAULT_IMAGE_EXTENSION = ".png"
DEFAULT_AUDIO_EXTENSION = ".wav"


class MediaTranscriptionError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "MEDIA_TRANSCRIPTION_ERROR",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type


class MediaTranscriptionService:
    def transcribe_bytes(
        self,
        *,
        input_type: str,
        content: bytes,
        filename: str = "",
        mime_type: str = "",
        provider: str | None = None,
    ) -> dict[str, Any]:
        media_type = normalize_text(input_type).lower()
        if media_type not in {"image", "audio"}:
            raise MediaTranscriptionError(
                400,
                "input_type must be either 'image' or 'audio'.",
                "UNSUPPORTED_INPUT_TYPE",
            )
        if not content:
            raise MediaTranscriptionError(400, "Uploaded media is empty.", "EMPTY_MEDIA")
        max_bytes = config.MEDIA_MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise MediaTranscriptionError(
                413,
                f"Uploaded media is larger than {config.MEDIA_MAX_UPLOAD_MB} MB.",
                "MEDIA_TOO_LARGE",
            )

        selected_provider = provider or (
            config.OCR_PROVIDER if media_type == "image" else config.STT_PROVIDER
        )
        temp_path = self._write_temp_file(media_type, content, filename)
        temp_deleted = False
        try:
            if media_type == "image":
                result = self._transcribe_image(content, selected_provider, mime_type)
            else:
                result = self._transcribe_audio(temp_path, selected_provider)
        finally:
            temp_deleted = self._delete_temp_file(temp_path)

        candidates = result["candidates"]
        selected_text = candidates[0]["text"] if candidates else ""
        confidence = max(
            [float(candidate.get("confidence") or 0.0) for candidate in candidates] or [0.0]
        )
        return {
            "input_type": media_type,
            "provider": result["provider"],
            "model_name": result["model_name"],
            "source_filename": filename or temp_path.name,
            "source_mime_type": mime_type,
            "source_size_bytes": len(content),
            "file_sha256": hashlib.sha256(content).hexdigest(),
            "candidates": candidates,
            "selected_text": selected_text,
            "confidence": round(confidence, 3),
            "needs_user_review": config.TRANSCRIPTION_REVIEW_REQUIRED,
            "status": "candidate_ready" if selected_text else "empty_result",
            "metadata": {
                **result.get("metadata", {}),
                "temp_file_deleted": temp_deleted,
                "temp_file_retained": not temp_deleted,
            },
        }

    def _transcribe_image(
        self,
        content: bytes,
        provider: str,
        mime_type: str = "",
    ) -> dict[str, Any]:
        if provider != "google":
            raise MediaTranscriptionError(
                400,
                f"Unsupported OCR provider: {provider}",
                "UNSUPPORTED_OCR_PROVIDER",
            )
        try:
            from google.cloud import vision  # type: ignore
        except Exception as exc:
            raise MediaTranscriptionError(
                503,
                "google-cloud-vision is not installed. Install requirements first.",
                "GOOGLE_VISION_SDK_MISSING",
            ) from exc

        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=content)
            image_context = vision.ImageContext(
                language_hints=config.GOOGLE_VISION_LANGUAGE_HINTS
            )
            feature = config.GOOGLE_VISION_FEATURE.upper()
            if feature == "TEXT_DETECTION":
                response = client.text_detection(image=image, image_context=image_context)
            else:
                response = client.document_text_detection(
                    image=image,
                    image_context=image_context,
                )
        except Exception as exc:
            raise MediaTranscriptionError(
                502,
                f"Google Vision OCR request failed: {exc}",
                "GOOGLE_VISION_REQUEST_FAILED",
            ) from exc
        if response.error.message:
            raise MediaTranscriptionError(
                502,
                f"Google Vision OCR failed: {response.error.message}",
                "GOOGLE_VISION_ERROR",
            )

        text = ""
        if response.full_text_annotation and response.full_text_annotation.text:
            text = response.full_text_annotation.text
        elif response.text_annotations:
            text = response.text_annotations[0].description
        confidence = self._google_vision_confidence(response)
        model_name = f"google-vision:{feature}"
        caption = self._caption_image(content, mime_type or "image/png")
        combined_text = self._combined_image_text(text, caption)
        return {
            "provider": "google",
            "model_name": model_name,
            "candidates": [
                self._candidate(
                    rank=1,
                    text=combined_text,
                    confidence=confidence,
                    provider="google",
                    model_name=model_name,
                )
            ],
            "metadata": {
                "language_hints": config.GOOGLE_VISION_LANGUAGE_HINTS,
                "feature": feature,
                "project": config.GOOGLE_CLOUD_PROJECT,
                "ocr_text": normalize_text(text),
                "image_caption": caption,
            },
        }

    def _caption_image(self, content: bytes, mime_type: str) -> dict[str, Any]:
        if not config.OPENAI_IMAGE_CAPTION_ENABLED or not config.OPENAI_API_KEY:
            return {"enabled": False}
        try:
            from openai import OpenAI  # type: ignore

            data_url = f"data:{mime_type};base64," + base64.b64encode(content).decode("ascii")
            client = OpenAI(
                api_key=config.OPENAI_API_KEY or None,
                timeout=config.OPENAI_IMAGE_CAPTION_TIMEOUT_SECONDS,
            )
            prompt = (
                "Analyze this diary-related image. Return compact JSON only with: "
                "objects(list), scene(string), emotional_atmosphere(list), "
                "inferred_context(string), caption_text_for_analysis(string), "
                "confidence(number), uncertainty(list). Focus on objects, scene, "
                "and emotional mood. Do not invent private facts."
            )
            response = client.responses.create(
                model=config.OPENAI_VISION_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": config.OPENAI_VISION_DETAIL,
                            },
                        ],
                    }
                ],
            )
            parsed = _parse_json(_response_text(response))
            if parsed:
                parsed["enabled"] = True
                parsed["provider"] = "openai"
                parsed["model"] = config.OPENAI_VISION_MODEL
                return parsed
            return {
                "enabled": True,
                "provider": "openai",
                "model": config.OPENAI_VISION_MODEL,
                "raw_text": _response_text(response)[:800],
            }
        except Exception as exc:
            return {
                "enabled": True,
                "provider": "openai",
                "model": config.OPENAI_VISION_MODEL,
                "error": str(exc)[:300],
            }

    @staticmethod
    def _combined_image_text(ocr_text: str, caption: dict[str, Any]) -> str:
        parts: list[str] = []
        clean_ocr = normalize_text(ocr_text)
        if clean_ocr:
            parts.append(f"OCR text: {clean_ocr}")
        caption_text = normalize_text(str(caption.get("caption_text_for_analysis") or ""))
        if caption_text:
            parts.append(f"Image scene: {caption_text}")
        elif caption.get("scene"):
            parts.append(f"Image scene: {normalize_text(str(caption.get('scene')))}")
        atmosphere = caption.get("emotional_atmosphere") or []
        if atmosphere:
            parts.append("Image emotional atmosphere: " + ", ".join(map(str, atmosphere)))
        inferred = normalize_text(str(caption.get("inferred_context") or ""))
        if inferred:
            parts.append(f"Image inferred context: {inferred}")
        return "\n".join(parts)

    def _transcribe_audio(self, temp_path: Path, provider: str) -> dict[str, Any]:
        if provider != "openai":
            raise MediaTranscriptionError(
                400,
                f"Unsupported STT provider: {provider}",
                "UNSUPPORTED_STT_PROVIDER",
            )
        if not config.OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
            raise MediaTranscriptionError(
                503,
                "OPENAI_API_KEY is not configured.",
                "OPENAI_API_KEY_MISSING",
            )
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise MediaTranscriptionError(
                503,
                "openai is not installed. Install requirements first.",
                "OPENAI_SDK_MISSING",
            ) from exc

        model_name = config.OPENAI_TRANSCRIPTION_MODEL or "gpt-4o-mini-transcribe"
        client = OpenAI(api_key=config.OPENAI_API_KEY or None)
        try:
            with temp_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio_file,
                    language=config.OPENAI_TRANSCRIPTION_LANGUAGE or "ko",
                )
        except Exception as exc:
            raise MediaTranscriptionError(
                502,
                f"OpenAI STT request failed: {exc}",
                "OPENAI_STT_REQUEST_FAILED",
            ) from exc
        text = self._response_text(response)
        return {
            "provider": "openai",
            "model_name": model_name,
            "candidates": [
                self._candidate(
                    rank=1,
                    text=text,
                    confidence=0.0,
                    provider="openai",
                    model_name=model_name,
                )
            ],
            "metadata": {
                "language": config.OPENAI_TRANSCRIPTION_LANGUAGE or "ko",
            },
        }

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        text = getattr(response, "text", None)
        if text is not None:
            return str(text)
        if isinstance(response, dict):
            return str(response.get("text") or "")
        try:
            data = response.model_dump()
            return str(data.get("text") or "")
        except Exception:
            return ""

    @staticmethod
    def _candidate(
        *,
        rank: int,
        text: str,
        confidence: float,
        provider: str,
        model_name: str,
    ) -> dict[str, Any]:
        return {
            "rank": rank,
            "text": normalize_text(text),
            "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
            "provider": provider,
            "model_name": model_name,
        }

    @staticmethod
    def _google_vision_confidence(response: Any) -> float:
        values: list[float] = []
        annotation = getattr(response, "full_text_annotation", None)
        for page in getattr(annotation, "pages", []) or []:
            for block in getattr(page, "blocks", []) or []:
                if getattr(block, "confidence", 0):
                    values.append(float(block.confidence))
                for paragraph in getattr(block, "paragraphs", []) or []:
                    if getattr(paragraph, "confidence", 0):
                        values.append(float(paragraph.confidence))
                    for word in getattr(paragraph, "words", []) or []:
                        if getattr(word, "confidence", 0):
                            values.append(float(word.confidence))
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _write_temp_file(media_type: str, content: bytes, filename: str) -> Path:
        config.MEDIA_TMP_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix
        if not suffix:
            suffix = DEFAULT_IMAGE_EXTENSION if media_type == "image" else DEFAULT_AUDIO_EXTENSION
        temp_path = config.MEDIA_TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
        temp_path.write_bytes(content)
        return temp_path

    @staticmethod
    def _delete_temp_file(path: Path) -> bool:
        if not config.MEDIA_DELETE_UPLOADS:
            return False
        try:
            path.unlink(missing_ok=True)
            return not path.exists()
        except Exception:
            return False


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    try:
        data = response.model_dump()
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(response)


def _parse_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
