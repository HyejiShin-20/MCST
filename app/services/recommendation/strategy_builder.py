from __future__ import annotations

from app.schemas.recommendation import RecommendationSlot
from app.services.recommendation.rules import DEFAULT_ROLE_WEIGHTS, ROLE_PURPOSES, ROLE_WEIGHTS


class StrategyBuilder:
    def build_slots(self, analysis: dict, per_type: int) -> list[RecommendationSlot]:
        strategy = analysis.get("recommendation_strategy", {})
        desired = analysis.get("desired_support", [])
        intent = analysis.get("situation", {}).get("intent", "")
        roles = intent_roles(intent, desired) or list(strategy.get("emotional_arc", []))
        if "동기부여" in desired and "재도전" not in roles:
            roles.append("재도전")
        if "확장" in desired and "확장" not in roles:
            roles.append("확장")
        if "탐색" in desired and "가벼운 탐색" not in roles:
            roles.append("가벼운 탐색")
        for role in [
            "전환",
            "증폭",
            "축하",
            "음미",
            "영감",
            "유지",
            "발견",
            "배경",
            "가벼운 성찰",
            "학습",
            "몰입",
        ]:
            if role in desired and role not in roles:
                roles.append(role)
        if not roles:
            roles = ["가벼운 탐색"]
        while len(roles) < per_type:
            roles.append(roles[-1])
        return [
            RecommendationSlot(
                role=role,
                purpose=ROLE_PURPOSES.get(role, f"{role} 방향의 정서적 보조"),
                target_effect=target_effect(role),
                weights=ROLE_WEIGHTS.get(role, DEFAULT_ROLE_WEIGHTS),
            )
            for role in roles[:per_type]
        ]


def target_effect(role: str) -> str:
    mapping = {
        "안전": "위험 감정 완화",
        "공감": "이해받음",
        "안정": "긴장 완화",
        "회복": "회복",
        "재도전": "자기효능감",
        "자기효능감": "자기효능감",
        "호기심": "호기심",
        "몰입": "전환",
        "확장": "확장",
        "가벼운 탐색": "가벼운 몰입",
        "전환": "기분 전환",
        "증폭": "좋은 기분 확장",
        "축하": "성취감 축하",
        "음미": "좋은 여운",
        "영감": "다음 가능성",
        "유지": "잔잔한 유지",
        "발견": "새 취향 발견",
        "배경": "저부담 동행",
        "가벼운 성찰": "일상 정리",
        "탐색": "관심사 탐색",
        "학습": "알아가는 즐거움",
    }
    return mapping.get(role, role)


def intent_roles(intent: str, desired: list[str]) -> list[str]:
    if intent == "interest_based":
        return ["탐색", "학습", "몰입", "영감"]
    if intent == "neutral_daily_based":
        if "무료함" in desired:
            return ["발견", "가벼운 성찰", "배경", "유지"]
        return ["유지", "배경", "발견", "가벼운 성찰"]
    if intent == "positive_emotion_based":
        if any(item in desired for item in ["회복", "안정"]) and "축하" in desired:
            return ["음미", "회복", "축하", "유지"]
        if "축하" in desired:
            return ["축하", "음미", "영감", "증폭"]
        return ["증폭", "음미", "영감", "발견"]
    if intent == "mixed_emotion_based" and ("축하" in desired or "음미" in desired):
        return ["축하", "음미", "안정", "영감"]
    return []
