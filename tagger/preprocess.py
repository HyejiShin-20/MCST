from __future__ import annotations

import re
from typing import Any

from app.services.text_utils import normalize_text as base_normalize_text


BRACKET_RE = re.compile(r"\[[^\]]+\]|\([^\)]+\)")
NOISE_RE = re.compile(r"(copyright|all rights reserved|작사|작곡|편곡)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return base_normalize_text(text)


def preprocess_lyrics(lyrics: str) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in (lyrics or "").splitlines():
        line = BRACKET_RE.sub(" ", raw_line)
        line = normalize_text(line)
        if not line or NOISE_RE.search(line):
            continue
        if len(line) <= 2:
            continue
        normalized_key = line.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        lines.append(line)
    return normalize_text(" ".join(lines))


def preprocess_description(text: str) -> str:
    cleaned = BRACKET_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return normalize_text(cleaned)


def build_tagging_text(content: dict[str, Any]) -> tuple[str, str]:
    title = content.get("title", "")
    creator = content.get("creator", "")
    genre = content.get("genre", "")
    content_type = content.get("content_type", "")
    summary = content.get("summary") or content.get("raw_description") or ""
    raw_lyrics = content.get("raw_lyrics") or ""

    if content_type == "music" and raw_lyrics:
        processed_text = preprocess_lyrics(raw_lyrics)
    else:
        processed_text = preprocess_description(
            content.get("processed_description") or summary
        )

    tagging_text = "\n".join(
        part
        for part in [
            f"제목: {title}",
            f"창작자: {creator}",
            f"유형: {content_type}",
            f"장르: {genre}",
            f"내용: {processed_text}",
        ]
        if part.strip()
    )
    return processed_text, normalize_text(tagging_text)


def text_quality_score(text: str) -> float:
    length = len(normalize_text(text))
    if length >= 160:
        return 1.0
    if length >= 100:
        return 0.82
    if length >= 60:
        return 0.64
    if length >= 30:
        return 0.42
    return 0.15
