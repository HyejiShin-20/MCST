from __future__ import annotations

from typing import Any

from app import config
from app.services.embedding_service import EmbeddingService
from app.services.text_utils import cosine_similarity
from tagger.tag_config import (
    EMOTION_TAGS_NEGATIVE,
    EMOTION_TAGS_NEUTRAL,
    EMOTION_TAGS_POSITIVE,
    MOOD_TAGS,
    RECOMMENDATION_ROLES,
    TAG_PROTOTYPES,
    THEME_TAGS,
)


_EMBEDDER: EmbeddingService | None = None
_PROTOTYPE_VECTORS: dict[str, Any] | None = None


def tag_by_prototype(text: str, top_k: int = 18) -> dict[str, dict[str, float]]:
    embedder = _embedder()
    prototype_vectors = _prototype_vectors(embedder)
    text_vector = embedder.encode([text]).vectors[0]
    scored = sorted(
        (
            (tag, cosine_similarity(text_vector, vector))
            for tag, vector in prototype_vectors.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    result = _empty_result()
    for tag, score in scored:
        bucket = _bucket_for_tag(tag)
        if bucket:
            result[bucket][tag] = round(float(score), 4)
    return result


def merge_tags(
    keyword_result: dict[str, dict[str, float]],
    prototype_result: dict[str, dict[str, float]],
) -> dict[str, Any]:
    merged = _empty_result()
    for bucket in merged:
        tags = set(keyword_result.get(bucket, {})) | set(prototype_result.get(bucket, {}))
        has_keyword_bucket = bool(keyword_result.get(bucket))
        for tag in tags:
            keyword_score = keyword_result.get(bucket, {}).get(tag, 0.0)
            prototype_score = prototype_result.get(bucket, {}).get(tag, 0.0)
            prototype_weight = 0.18 if has_keyword_bucket and not keyword_score else 0.38
            keyword_weight = 1.0 - prototype_weight
            score = keyword_score * keyword_weight + prototype_score * prototype_weight
            if keyword_score and prototype_score:
                score += 0.06
            merged[bucket][tag] = round(min(1.0, score), 4)
    return {
        "scores": merged,
        "keyword": keyword_result,
        "prototype": prototype_result,
    }


def calculate_confidence(merged_result: dict[str, Any]) -> float:
    scores: list[float] = []
    keyword_strength = 0.0
    for bucket, bucket_scores in merged_result["scores"].items():
        values = sorted(bucket_scores.values(), reverse=True)
        scores.extend(values[:3])
        if merged_result["keyword"].get(bucket):
            keyword_strength = max(keyword_strength, max(merged_result["keyword"][bucket].values()))
    if not scores:
        return 0.35
    scores.sort(reverse=True)
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = max(0.0, top1 - top2)
    bucket_coverage = sum(
        1 for bucket_scores in merged_result["scores"].values() if bucket_scores
    ) / 4
    confidence = 0.5 * top1 + 0.28 * keyword_strength + 0.14 * margin + 0.08 * bucket_coverage
    return round(max(0.25, min(0.96, confidence)), 4)


def finalize_tags(
    merged_result: dict[str, Any],
    *,
    content_type: str,
    text_quality: float,
) -> dict[str, Any]:
    scores = merged_result["scores"]
    if not any(scores[bucket] for bucket in scores):
        scores["emotion_tags"]["평범함"] = 0.45
        scores["theme_tags"]["일상"] = 0.45
        scores["mood_tags"]["저부담"] = 0.45
        scores["recommendation_roles"]["발견"] = 0.45

    emotion_tags = _top(scores["emotion_tags"], 4)
    theme_tags = _top(scores["theme_tags"], 6)
    mood_tags = _top(scores["mood_tags"], 5)
    roles = _top(scores["recommendation_roles"], 5)
    if not roles:
        roles = _infer_roles(emotion_tags, theme_tags, mood_tags)
    confidence = calculate_confidence(merged_result)
    if text_quality < 0.45:
        confidence = round(confidence * 0.78, 4)

    numeric = _numeric_profile(
        scores=scores,
        content_type=content_type,
        emotion_tags=emotion_tags,
        theme_tags=theme_tags,
        mood_tags=mood_tags,
        roles=roles,
    )
    too_heavy = numeric["arousal"] >= 0.78 and numeric["valence"] <= -0.3
    dark_flag = any(tag in emotion_tags + mood_tags for tag in ["우울", "어두움", "강렬함", "긴장감"])
    background_friendly = any(tag in mood_tags + roles for tag in ["배경친화", "저부담", "배경", "유지"])
    is_recommendable = (
        confidence >= config.MIN_RECOMMEND_CONFIDENCE
        and text_quality >= 0.35
        and not too_heavy
    )
    return {
        "emotion_tags": emotion_tags,
        "theme_tags": theme_tags,
        "mood_tags": mood_tags,
        "recommendation_roles": roles,
        "valence": numeric["valence"],
        "arousal": numeric["arousal"],
        "intensity": numeric["intensity"],
        "energy": numeric["energy"],
        "cognitive_load": numeric["cognitive_load"],
        "tag_confidence": confidence,
        "dark_flag": dark_flag,
        "too_heavy_flag": too_heavy,
        "background_friendly": background_friendly,
        "is_recommendable": is_recommendable,
    }


def _embedder() -> EmbeddingService:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = EmbeddingService()
    return _EMBEDDER


def _prototype_vectors(embedder: EmbeddingService) -> dict[str, Any]:
    global _PROTOTYPE_VECTORS
    if _PROTOTYPE_VECTORS is None:
        tags = list(TAG_PROTOTYPES)
        vectors = embedder.encode([TAG_PROTOTYPES[tag] for tag in tags]).vectors
        _PROTOTYPE_VECTORS = dict(zip(tags, vectors))
    return _PROTOTYPE_VECTORS


def _empty_result() -> dict[str, dict[str, float]]:
    return {
        "emotion_tags": {},
        "theme_tags": {},
        "mood_tags": {},
        "recommendation_roles": {},
    }


def _bucket_for_tag(tag: str) -> str | None:
    if tag in EMOTION_TAGS_NEGATIVE or tag in EMOTION_TAGS_POSITIVE or tag in EMOTION_TAGS_NEUTRAL:
        return "emotion_tags"
    if tag in THEME_TAGS:
        return "theme_tags"
    if tag in MOOD_TAGS:
        return "mood_tags"
    if tag in RECOMMENDATION_ROLES:
        return "recommendation_roles"
    return None


def _top(scores: dict[str, float], limit: int) -> list[str]:
    return [
        tag
        for tag, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= 0.34
    ][:limit]


def _infer_roles(
    emotion_tags: list[str], theme_tags: list[str], mood_tags: list[str]
) -> list[str]:
    signals = set(emotion_tags) | set(theme_tags) | set(mood_tags)
    roles: list[str] = []

    role_rules = [
        ("공감", {"슬픔", "외로움", "상처", "허무함", "후회", "실망", "쓸쓸함", "상실", "그리움"}),
        ("안정", {"불안", "분노", "긴장감", "강렬함", "저부담", "평온"}),
        ("회복", {"무기력", "회복", "위로", "자기돌봄", "휴식", "고립"}),
        ("전환", {"분노", "무료함", "기분전환", "여행"}),
        ("축하", {"뿌듯함", "성취", "자신감", "감동"}),
        ("음미", {"기쁨", "즐거움", "감사", "안도감", "따뜻함", "잔잔함", "만족"}),
        ("증폭", {"기쁨", "즐거움", "설렘", "활력", "밝음", "신남"}),
        ("영감", {"희망", "자신감", "경외", "미래", "도전", "성장"}),
        ("탐색", {"호기심", "우주", "과학", "기술", "예술", "여행", "자연"}),
        ("학습", {"학습", "과학", "우주", "기술", "지식"}),
        ("몰입", {"우주", "과학", "예술", "음악", "영화", "도서"}),
        ("유지", {"평범함", "잔잔함", "일상", "평온", "배경친화"}),
        ("배경", {"저부담", "배경친화", "휴식", "집", "음식", "느림"}),
        ("발견", {"무료함", "가벼운 관심", "가벼움", "일상"}),
        ("가벼운 성찰", {"성찰", "자기이해", "기억", "시간", "일상"}),
        ("재도전", {"진로", "발표", "학교", "자존감", "실패", "도전"}),
    ]
    for role, cues in role_rules:
        if signals & cues:
            roles.append(role)
    return list(dict.fromkeys(roles or ["발견"]))[:5]


def _numeric_profile(
    *,
    scores: dict[str, dict[str, float]],
    content_type: str,
    emotion_tags: list[str],
    theme_tags: list[str],
    mood_tags: list[str],
    roles: list[str],
) -> dict[str, float]:
    positive = sum(scores["emotion_tags"].get(tag, 0.0) for tag in EMOTION_TAGS_POSITIVE)
    negative = sum(scores["emotion_tags"].get(tag, 0.0) for tag in EMOTION_TAGS_NEGATIVE)
    neutral = sum(scores["emotion_tags"].get(tag, 0.0) for tag in EMOTION_TAGS_NEUTRAL)
    valence = max(-1.0, min(1.0, (positive - negative) / max(1.0, positive + negative + neutral)))
    active_tags = {"신남", "축하", "영감", "강렬함", "긴장감", "도전", "설렘", "증폭"}
    low_tags = {"차분함", "아늑함", "느림", "저부담", "배경", "유지", "평범함"}
    arousal = 0.35
    arousal += 0.12 * len(set(mood_tags + roles + emotion_tags) & active_tags)
    arousal -= 0.07 * len(set(mood_tags + roles + emotion_tags) & low_tags)
    intensity = min(1.0, max(scores["emotion_tags"].values() or [0.35]) + abs(valence) * 0.16)
    energy = arousal + max(0.0, valence) * 0.16
    if "무기력" in emotion_tags:
        energy -= 0.18
    cognitive_load = 0.42
    if content_type == "book":
        cognitive_load += 0.16
    if content_type == "music":
        cognitive_load -= 0.14
    if set(theme_tags) & {"과학", "우주", "기술", "학습", "미래"}:
        cognitive_load += 0.12
    if set(mood_tags) & {"저부담", "배경친화", "가벼움"}:
        cognitive_load -= 0.12
    return {
        "valence": round(valence, 4),
        "arousal": round(max(0.0, min(1.0, arousal)), 4),
        "intensity": round(max(0.0, min(1.0, intensity)), 4),
        "energy": round(max(0.0, min(1.0, energy)), 4),
        "cognitive_load": round(max(0.0, min(1.0, cognitive_load)), 4),
    }
