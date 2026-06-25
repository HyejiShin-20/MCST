from __future__ import annotations

from typing import Any

from app.schemas.analysis import EvidenceDetail, Situation
from app.services.analysis_rules import (
    DISTRESS_EMOTIONS,
    NEUTRAL_EMOTIONS,
    POSITIVE_EMOTIONS,
    TOPIC_KEYWORDS,
)
from app.services.emotion_detector import EmotionDetector, intensity_label
from app.services.text_utils import tokenize


class ContextDetector:
    def detect(
        self,
        text: str,
        emotions: list[str],
        emotion_scores: dict[str, float],
        risk_flags: list[str],
        sentences: list[str],
        desired_support: list[str],
    ) -> dict[str, Any]:
        joined = " ".join(tokenize(text))
        topic_counts = EmotionDetector.weighted_counts(text, joined, TOPIC_KEYWORDS)
        topics = [name for name, score in topic_counts.most_common() if score > 0][:6]
        if not topics:
            topics = ["일상"]
            topic_counts["일상"] = 1.0

        topic_details = EmotionDetector.details(
            topics, topic_counts, TOPIC_KEYWORDS, sentences
        )
        context_keywords = self.context_keywords(text, emotions, topics, desired_support)
        situation = self.situation(
            text=text,
            emotions=emotions,
            topics=topics,
            desired_support=desired_support,
            risk_flags=risk_flags,
            emotion_scores=emotion_scores,
        )
        return {
            "topics": topics,
            "topic_details": topic_details,
            "context_keywords": context_keywords,
            "situation": situation,
        }

    @staticmethod
    def context_keywords(
        text: str, emotions: list[str], topics: list[str], desired_support: list[str]
    ) -> list[str]:
        keywords = list(
            dict.fromkeys(
                activity_keywords(text)
                + topics[:4]
                + desired_support[:4]
            )
        )
        if any(item in emotions for item in ["슬픔", "불안", "무기력", "외로움", "창피함"]):
            keywords.append("정서적 회복")
        if "피로" in emotions and any(item in emotions for item in ["뿌듯함", "기쁨", "안도감"]):
            keywords.append("성취 후 회복")
        if any(item in topics for item in ["발표", "시험", "진로"]):
            keywords.append("성과 압박")
        if any(item in topics for item in ["이별", "관계"]):
            keywords.append("관계 정리")
        if "성취" in topics and "불안" in emotions:
            keywords.append("성취 이후 부담")
        return list(dict.fromkeys(keywords))

    @staticmethod
    def situation(
        text: str,
        emotions: list[str],
        topics: list[str],
        desired_support: list[str],
        risk_flags: list[str],
        emotion_scores: dict[str, float],
    ) -> Situation:
        event_name = primary_event(topics, emotions)
        vulnerability = max(
            emotion_scores.get("sadness", 0.0),
            emotion_scores.get("anxiety", 0.0),
            emotion_scores.get("fatigue", 0.0),
            emotion_scores.get("shame", 0.0),
            emotion_scores.get("anger", 0.0),
        )
        positive = max(
            emotion_scores.get("hope", 0.0),
            emotion_scores.get("joy", 0.0),
            emotion_scores.get("pride", 0.0),
            emotion_scores.get("relief", 0.0),
            emotion_scores.get("curiosity", 0.0),
            emotion_scores.get("calm", 0.0),
        )
        neutral = max(
            emotion_scores.get("ordinary", 0.0),
            emotion_scores.get("boredom", 0.0),
            emotion_scores.get("neutral", 0.0),
        )
        intensity = min(1.0, max(vulnerability, positive) + length_intensity(text))
        risk_level = "high" if risk_flags else ("medium" if vulnerability >= 0.55 else "low")
        valence = valence_label(vulnerability, positive, emotions)
        arousal = "high" if any(item in emotions for item in ["불안", "분노", "설렘"]) else "low"
        uncertainty = uncertainty_reasons(emotions, topics)
        confidence = confidence_score(emotions, topics, uncertainty, risk_flags)
        return Situation(
            primary_event=event_name,
            intent=intent(topics, desired_support, emotions, vulnerability, positive, neutral),
            needs=needs(desired_support),
            stressors=[
                topic
                for topic in topics
                if topic in {"발표", "시험", "진로", "관계", "이별", "자존감"}
            ],
            risk_level=risk_level,
            valence=valence,
            arousal=arousal,
            intensity=round(intensity, 3),
            intensity_label=intensity_label(intensity),
            support_priority=desired_support,
            confidence=confidence,
            uncertainty_reasons=uncertainty,
        )


def primary_event(topics: list[str], emotions: list[str] | None = None) -> str:
    emotions = emotions or []
    if "이별" in topics:
        return "관계 상실/이별"
    if "관계" in topics and any(item in emotions for item in ["기쁨", "감사", "안도감", "평온"]):
        return "관계 속 연결/즐거움"
    if "관계" in topics:
        return "관계 갈등 또는 연결 욕구"
    if "발표" in topics:
        return "평가 상황 이후의 자신감 저하"
    if "시험" in topics:
        return "성과 압박과 시험 스트레스"
    if "성취" in topics and "진로" in topics:
        return "성취 이후 다음 단계 부담"
    if "성취" in topics:
        return "성취와 여운"
    if any(item in topics for item in ["집", "음식", "휴식", "일상"]) and any(
        item in emotions for item in ["기쁨", "즐거움", "감사", "안도감", "평온", "뿌듯함"]
    ):
        return "여유로운 긍정 일상"
    if "집" in topics or "음식" in topics or "휴식" in topics:
        return "잔잔한 일상"
    if "일상" in topics:
        return "일상 감정 기록"
    if "진로" in topics:
        return "진로/미래 탐색"
    if "우주" in topics:
        return "우주와 탐험에 대한 새 관심"
    if "여행" in topics:
        return "일상 이탈과 여행 욕구"
    return "일상 감정 기록"


def activity_keywords(text: str) -> list[str]:
    patterns = [
        ("코딩 프로젝트", ["코딩 프로젝트", "코딩", "개발"]),
        ("운동", ["운동", "헬스", "러닝", "조깅", "필라테스", "요가"]),
        ("반려동물 산책", ["강아지", "반려견", "반려동물", "토리", "산책"]),
        ("공모전 작업", ["공모전"]),
        ("집에서 보낸 시간", ["집에 와", "집에와", "우리집", "집"]),
    ]
    found: list[str] = []
    for label, cues in patterns:
        if any(cue in text for cue in cues):
            found.append(label)
    return found


def intent(
    topics: list[str],
    desired_support: list[str],
    emotions: list[str],
    vulnerability: float,
    positive: float,
    neutral: float,
) -> str:
    if "안전" in desired_support:
        return "negative_emotion_based"
    has_distress = bool(set(emotions) & DISTRESS_EMOTIONS)
    has_positive = bool(set(emotions) & POSITIVE_EMOTIONS)
    has_neutral = bool(set(emotions) & NEUTRAL_EMOTIONS)
    if has_distress and has_positive:
        return "mixed_emotion_based"
    if has_distress:
        return "negative_emotion_based"
    if has_positive and positive >= max(0.25, vulnerability * 0.85, neutral * 0.72):
        return "positive_emotion_based"
    if has_neutral and neutral >= max(0.2, vulnerability * 0.8) and not any(
        topic in topics for topic in ["우주", "과학", "여행", "진로", "학습"]
    ):
        return "neutral_daily_based"
    if any(item in desired_support for item in ["학습", "확장", "탐색", "몰입"]) and any(
        topic in topics for topic in ["우주", "과학", "여행", "예술", "학습", "진로"]
    ):
        return "interest_based"
    if has_neutral or any(item in desired_support for item in ["유지", "발견", "배경"]):
        return "neutral_daily_based"
    return "neutral_daily_based"


def needs(desired_support: list[str]) -> list[str]:
    mapping = {
        "안전": "위험 감정 완화",
        "공감": "감정이 이해받는 느낌",
        "안정": "과흥분과 불안을 낮추기",
        "회복": "다시 움직일 수 있는 정서적 여지",
        "동기부여": "자기효능감과 재도전 감각",
        "확장": "관심사를 넓히는 지적 자극",
        "탐색": "부담 없는 문화 탐색",
        "전환": "기분을 가볍게 환기하기",
        "증폭": "좋은 기분을 밝게 이어가기",
        "음미": "좋은 여운을 천천히 머무르게 하기",
        "축하": "잘 해낸 일을 가볍게 축하하기",
        "영감": "다음 가능성을 넓히기",
        "유지": "잔잔한 하루의 흐름 유지",
        "발견": "부담 없는 새 취향 발견",
        "배경": "쉬는 시간에 곁에 둘 저부담 콘텐츠",
        "가벼운 성찰": "일상을 가볍게 돌아보기",
        "학습": "관심 주제를 더 알아가기",
        "몰입": "관심사에 깊게 들어가기",
    }
    return [mapping[item] for item in desired_support if item in mapping]


def valence_label(vulnerability: float, positive: float, emotions: list[str]) -> str:
    has_distress = bool(set(emotions) & DISTRESS_EMOTIONS)
    has_positive = bool(set(emotions) & POSITIVE_EMOTIONS)
    if has_distress and has_positive:
        return "mixed"
    if has_positive and vulnerability >= 0.25 and positive >= vulnerability * 0.75:
        return "mixed"
    if vulnerability >= 0.35:
        return "negative"
    if positive > 0.4:
        return "positive"
    return "neutral"


def uncertainty_reasons(emotions: list[str], topics: list[str]) -> list[str]:
    reasons: list[str] = []
    if bool(set(emotions) & DISTRESS_EMOTIONS) and bool(set(emotions) & POSITIVE_EMOTIONS):
        reasons.append("상반된 감정 신호가 함께 나타남")
    if len(emotions) >= 4:
        reasons.append("감정 후보가 많아 우선순위 판단이 필요함")
    if len(topics) >= 5:
        reasons.append("상황 단서가 여러 갈래로 분산됨")
    return reasons


def confidence_score(
    emotions: list[str], topics: list[str], uncertainty: list[str], risk_flags: list[str]
) -> float:
    score = 0.72
    if emotions and topics:
        score += 0.12
    if risk_flags:
        score += 0.04
    score -= 0.08 * len(uncertainty)
    return round(max(0.35, min(0.95, score)), 3)


def length_intensity(text: str) -> float:
    if len(text) >= 120:
        return 0.12
    if len(text) >= 60:
        return 0.08
    return 0.03
