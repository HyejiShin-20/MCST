from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from typing import Any
from urllib import request
from urllib.parse import urlparse

from app import config
from app.services.text_utils import normalize_text


class LinkTranscriptionError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "LINK_TRANSCRIPTION_ERROR",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type


class LinkTranscriptionService:
    def transcribe_url(self, url: str) -> dict[str, Any]:
        clean_url = normalize_text(url)
        parsed = urlparse(clean_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LinkTranscriptionError(400, "http 또는 https 링크를 입력해 주세요.", "INVALID_URL")

        html, content_type = self._download(clean_url)
        extracted = extract_readable_text(html)
        if not extracted:
            raise LinkTranscriptionError(422, "링크에서 분석할 텍스트를 찾지 못했습니다.", "EMPTY_LINK_TEXT")

        selected_text = extracted[: config.LINK_MAX_TEXT_CHARS]
        sha = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()
        model_name = "stdlib-html-link-extractor"
        return {
            "input_type": "link",
            "provider": "link",
            "model_name": model_name,
            "source_filename": clean_url,
            "source_mime_type": content_type,
            "source_size_bytes": len(html.encode("utf-8")),
            "file_sha256": sha,
            "candidates": [
                {
                    "rank": 1,
                    "text": selected_text,
                    "confidence": 0.72,
                    "provider": "link",
                    "model_name": model_name,
                }
            ],
            "selected_text": selected_text,
            "confidence": 0.72,
            "needs_user_review": True,
            "status": "candidate_ready",
            "metadata": {
                "url": clean_url,
                "content_type": content_type,
                "extractor": model_name,
                "max_bytes": config.LINK_MAX_BYTES,
            },
        }

    @staticmethod
    def _download(url: str) -> tuple[str, str]:
        req = request.Request(
            url,
            headers={
                "User-Agent": "EmotionCultureRecommender/0.1",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        try:
            with request.urlopen(req, timeout=config.LINK_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("content-type", "")
                raw = response.read(config.LINK_MAX_BYTES + 1)
        except Exception as exc:
            raise LinkTranscriptionError(502, f"링크를 읽지 못했습니다: {exc}", "LINK_FETCH_FAILED") from exc

        if len(raw) > config.LINK_MAX_BYTES:
            raise LinkTranscriptionError(413, "링크 본문이 너무 큽니다.", "LINK_TOO_LARGE")

        charset = "utf-8"
        match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
        if match:
            charset = match.group(1).strip("\"'")
        try:
            return raw.decode(charset, errors="replace"), content_type
        except LookupError:
            return raw.decode("utf-8", errors="replace"), content_type


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "canvas"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "canvas"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = normalize_text(data)
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return normalize_text(" ".join(self._chunks))


def extract_readable_text(raw: str) -> str:
    if "<html" not in raw.lower() and "<body" not in raw.lower():
        return normalize_text(raw)
    parser = ReadableHTMLParser()
    parser.feed(raw)
    text = parser.text()
    text = re.sub(r"\s+", " ", text)
    return normalize_text(text)
