from __future__ import annotations

from typing import Any

from app.services.recommendation.rules import (
    HIGH_TENSION_TAGS,
    LOW_TENSION_TAGS,
    RESTORATIVE_TAGS,
)


POSITIVE_TAGS = {
    "기쁨",
    "희망",
    "설렘",
    "따뜻함",
    "감동",
    "평온",
    "경외",
}

DARK_TAGS = {
    "우울",
    "절망",
    "비극",
    "고립",
    "상실",
    "폭력",
    "복수",
    "두려움",
    "압박",
    "불안",
    "후회",
    "외로움",
    "쓸쓸함",
}

ACTIVATING_TAGS = {
    "도전",
    "재출발",
    "성장",
    "탐험",
    "여행",
    "설렘",
    "희망",
    "기쁨",
    "자기표현",
    "우주",
    "미래",
    "호기심",
}

REFLECTIVE_TAGS = {
    "성찰",
    "자기이해",
    "기억",
    "시간",
    "삶의 의미",
    "정체성",
    "선택",
}

RELATIONAL_TAGS = {
    "관계",
    "사랑",
    "가족",
    "이별",
    "연결",
    "돌봄",
    "공감",
}

COGNITIVE_LOAD_TAGS = {
    "과학",
    "지식",
    "역사",
    "사회",
    "언어",
    "철학",
    "미래",
    "SF",
    "정체성",
}

LOW_LOAD_TAGS = {
    "휴식",
    "평온",
    "고요",
    "명상",
    "일상",
    "음악",
    "따뜻함",
    "기쁨",
}


def enrich_content_with_affect_profile(content: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(content)
    enriched["affect_profile"] = build_content_affect_profile(content)
    return enriched


def build_content_affect_profile(content: dict[str, Any]) -> dict[str, Any]:
    tags = (
        set(content.get("emotion_tags", []))
        | set(content.get("topic_tags", []))
        | set(content.get("mood_tags", []))
        | set(content.get("recommendation_roles", []))
    )
    content_type = content.get("content_type", "")
    genre = str(content.get("genre", ""))

    tension_level = _clamp(
        0.18
        + _coverage(tags, HIGH_TENSION_TAGS) * 0.55
        + _coverage(tags, DARK_TAGS) * 0.16
        - _coverage(tags, LOW_TENSION_TAGS) * 0.22
    )
    darkness_level = _clamp(
        0.1
        + _coverage(tags, DARK_TAGS) * 0.58
        + _coverage(tags, {"상실", "고립", "우울", "절망", "비극"}) * 0.18
        - _coverage(tags, RESTORATIVE_TAGS) * 0.14
    )
    recovery_potential = _clamp(
        0.22
        + _coverage(tags, RESTORATIVE_TAGS) * 0.46
        + _coverage(tags, POSITIVE_TAGS) * 0.24
        + _coverage(tags, {"자기돌봄", "재출발", "자존감", "성장"}) * 0.18
        - darkness_level * 0.1
    )
    calming_effect = _clamp(
        0.22
        + _coverage(tags, LOW_TENSION_TAGS) * 0.52
        + _coverage(tags, {"휴식", "명상", "고요", "일상"}) * 0.18
        - tension_level * 0.2
    )
    activation_level = _clamp(
        0.2
        + _coverage(tags, ACTIVATING_TAGS) * 0.55
        + _coverage(tags, {"도전", "재출발", "자기표현"}) * 0.18
        - calming_effect * 0.08
    )
    reflection_depth = _clamp(
        0.18
        + _coverage(tags, REFLECTIVE_TAGS) * 0.52
        + _coverage(tags, RELATIONAL_TAGS) * 0.12
    )
    cognitive_load = _metadata_or(
        content,
        "cognitive_load",
        _cognitive_load(tags, content_type, genre),
    )
    emotional_intensity = _clamp(
        0.18
        + tension_level * 0.34
        + darkness_level * 0.24
        + activation_level * 0.18
        + reflection_depth * 0.1
    )

    return {
        "dominant_tone": _dominant_tone(
            recovery_potential=recovery_potential,
            calming_effect=calming_effect,
            tension_level=tension_level,
            darkness_level=darkness_level,
            activation_level=activation_level,
            reflection_depth=reflection_depth,
        ),
        "tension_level": round(_metadata_or(content, "arousal", tension_level), 4),
        "darkness_level": round(darkness_level, 4),
        "recovery_potential": round(recovery_potential, 4),
        "calming_effect": round(calming_effect, 4),
        "activation_level": round(_metadata_or(content, "energy", activation_level), 4),
        "reflection_depth": round(reflection_depth, 4),
        "cognitive_load": round(cognitive_load, 4),
        "emotional_intensity": round(_metadata_or(content, "intensity", emotional_intensity), 4),
        "regulation_effects": _regulation_effects(
            recovery_potential=recovery_potential,
            calming_effect=calming_effect,
            activation_level=activation_level,
            reflection_depth=reflection_depth,
            darkness_level=darkness_level,
        ),
        "recommended_when": _recommended_when(
            recovery_potential=recovery_potential,
            calming_effect=calming_effect,
            activation_level=activation_level,
            reflection_depth=reflection_depth,
            cognitive_load=cognitive_load,
        ),
        "avoid_when": _avoid_when(
            tension_level=tension_level,
            darkness_level=darkness_level,
            cognitive_load=cognitive_load,
        ),
    }


def _cognitive_load(tags: set[str], content_type: str, genre: str) -> float:
    score = 0.36 + _coverage(tags, COGNITIVE_LOAD_TAGS) * 0.36
    if content_type == "book":
        score += 0.16
    elif content_type == "music":
        score -= 0.16
    elif content_type == "movie":
        score += 0.04

    if any(keyword in genre for keyword in ["과학", "인문", "SF"]):
        score += 0.1
    if any(keyword in genre for keyword in ["연주곡", "발라드", "코미디"]):
        score -= 0.08

    score -= _coverage(tags, LOW_LOAD_TAGS) * 0.14
    return _clamp(score)


def _metadata_or(content: dict[str, Any], key: str, fallback: float) -> float:
    metadata = content.get("tag_metadata") or {}
    value = metadata.get(key)
    if isinstance(value, (int, float)):
        return _clamp(float(value))
    return fallback


def _dominant_tone(
    *,
    recovery_potential: float,
    calming_effect: float,
    tension_level: float,
    darkness_level: float,
    activation_level: float,
    reflection_depth: float,
) -> list[str]:
    tone_scores = {
        "회복형": recovery_potential,
        "안정형": calming_effect,
        "활력형": activation_level,
        "사색형": reflection_depth,
        "고긴장형": tension_level,
        "어두운 정서": darkness_level,
        "양가형": min(recovery_potential, darkness_level) + 0.08,
    }
    return [
        tone
        for tone, score in sorted(tone_scores.items(), key=lambda item: item[1], reverse=True)
        if score >= 0.38
    ][:3]


def _regulation_effects(
    *,
    recovery_potential: float,
    calming_effect: float,
    activation_level: float,
    reflection_depth: float,
    darkness_level: float,
) -> list[str]:
    effects: list[str] = []
    if calming_effect >= 0.48:
        effects.append("긴장 완화")
    if recovery_potential >= 0.48:
        effects.append("회복/위로")
    if activation_level >= 0.48:
        effects.append("행동 에너지")
    if reflection_depth >= 0.48:
        effects.append("감정 정리")
    if darkness_level >= 0.5:
        effects.append("어려운 감정 직면")
    return effects or ["가벼운 전환"]


def _recommended_when(
    *,
    recovery_potential: float,
    calming_effect: float,
    activation_level: float,
    reflection_depth: float,
    cognitive_load: float,
) -> list[str]:
    states: list[str] = []
    if calming_effect >= 0.5:
        states.append("불안/긴장을 낮추고 싶을 때")
    if recovery_potential >= 0.5:
        states.append("위로와 회복감이 필요할 때")
    if activation_level >= 0.5:
        states.append("다음 행동의 에너지가 필요할 때")
    if reflection_depth >= 0.5:
        states.append("감정을 차분히 정리하고 싶을 때")
    if cognitive_load <= 0.38:
        states.append("부담 낮은 콘텐츠가 필요할 때")
    return states or ["일상적인 문화 탐색을 하고 싶을 때"]


def _avoid_when(
    *,
    tension_level: float,
    darkness_level: float,
    cognitive_load: float,
) -> list[str]:
    states: list[str] = []
    if tension_level >= 0.62:
        states.append("불안이 높고 긴장 완화가 먼저 필요할 때")
    if darkness_level >= 0.58:
        states.append("무기력/우울이 깊어 침잠을 피해야 할 때")
    if cognitive_load >= 0.68:
        states.append("인지적 여유가 거의 없을 때")
    return states


def _coverage(tags: set[str], targets: set[str]) -> float:
    if not targets:
        return 0.0
    return min(1.0, len(tags & targets) / 3)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
