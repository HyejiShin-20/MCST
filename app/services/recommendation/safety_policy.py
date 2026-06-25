from __future__ import annotations

from app.schemas.recommendation import SafetyDecision
from app.services.recommendation.content_profile import build_content_affect_profile
from app.services.recommendation.rules import (
    AMPLIFYING_NEGATIVE_TAGS,
    HIGH_TENSION_TAGS,
    LOW_TENSION_TAGS,
    RESTORATIVE_TAGS,
    VULNERABLE_EMOTIONS,
)


class SafetyPolicy:
    def evaluate(self, analysis: dict, content: dict) -> SafetyDecision:
        emotions = set(analysis.get("emotions", []))
        content_tags = set(content.get("emotion_tags", [])) | set(content.get("topic_tags", []))
        affect_profile = content.get("affect_profile") or build_content_affect_profile(content)
        risk_level = analysis.get("situation", {}).get("risk_level", "low")
        vulnerable = bool(emotions & VULNERABLE_EMOTIONS)
        amplifying = bool(content_tags & AMPLIFYING_NEGATIVE_TAGS)
        restorative = bool(content_tags & RESTORATIVE_TAGS)
        high_tension = bool(content_tags & HIGH_TENSION_TAGS)
        low_tension = bool(content_tags & LOW_TENSION_TAGS)
        tension_level = affect_profile.get("tension_level", 0.0)
        darkness_level = affect_profile.get("darkness_level", 0.0)
        recovery_potential = affect_profile.get("recovery_potential", 0.0)
        calming_effect = affect_profile.get("calming_effect", 0.0)
        cognitive_load = affect_profile.get("cognitive_load", 0.0)

        reasons: list[str] = []
        title = str(content.get("title", ""))
        strong_distress_title = any(
            cue in title for cue in ["응급실", "죽고", "눈물", "절망", "우울"]
        )
        if content.get("content_type") == "music" and strong_distress_title and not vulnerable:
            return SafetyDecision(
                allowed=False,
                safety_score=0.0,
                penalty=1.0,
                reasons=["현재 입력과 맞지 않는 강한 부정 제목 신호"],
            )

        if risk_level == "high" and amplifying and recovery_potential < 0.45:
            return SafetyDecision(
                allowed=False,
                safety_score=0.0,
                penalty=1.0,
                reasons=["위험 신호가 높고 부정 감정을 강화할 가능성이 큼"],
            )

        score = 0.72
        penalty = 0.0
        if vulnerable and amplifying and recovery_potential < 0.5:
            score -= 0.35
            penalty += 0.22
            reasons.append("취약 감정에서 부정 정서를 강화할 수 있음")
        if "불안" in emotions and (high_tension or tension_level >= 0.58) and not low_tension:
            amount = 0.16 if calming_effect >= 0.45 else 0.24
            score -= amount
            penalty += 0.14
            reasons.append("불안 상태에서 긴장도가 높을 수 있음")
        if "분노" in emotions and ("분노" in content_tags or "폭력" in content_tags):
            score -= 0.2
            penalty += 0.14
            reasons.append("분노 상태에서 공격성을 강화할 수 있음")
        if "무기력" in emotions and (amplifying or darkness_level >= 0.52) and recovery_potential < 0.5:
            score -= 0.2
            penalty += 0.12
            reasons.append("무기력 상태에서 침잠을 강화할 수 있음")
        if vulnerable and cognitive_load >= 0.72:
            score -= 0.12
            penalty += 0.08
            reasons.append("현재 정서에서는 인지 부담이 높을 수 있음")
        if vulnerable and recovery_potential >= 0.5:
            score += 0.2
            reasons.append("회복/위로 태그가 있어 취약 감정 완화에 적합")
        elif restorative or recovery_potential >= 0.5:
            score += 0.1
        if calming_effect >= 0.52 and "불안" in emotions:
            score += 0.08
            reasons.append("차분한 정서 전환 가능성이 있음")

        score = max(0.0, min(1.0, score))
        return SafetyDecision(
            allowed=True,
            safety_score=round(score, 4),
            penalty=round(penalty, 4),
            reasons=reasons,
        )
