from __future__ import annotations

from app.schemas.analysis import EmotionalGoal, RecommendationStrategy, Situation
from app.services.analysis_rules import (
    DISTRESS_EMOTIONS,
    EMOTIONAL_ARCS,
    POSITIVE_EMOTIONS,
    SUPPORT_PRIORITY,
    SUPPORT_RULES,
)


class SupportStrategyPlanner:
    def desired_support(
        self, emotions: list[str], topics: list[str], risk_flags: list[str]
    ) -> list[str]:
        signals = set(emotions) | set(topics)
        if risk_flags:
            signals.add("risk")
        emotion_weight = {
            emotion: max(0.35, 1.0 - index * 0.16)
            for index, emotion in enumerate(emotions)
        }
        topic_weight = {
            topic: max(0.28, 0.58 - index * 0.06)
            for index, topic in enumerate(topics)
        }
        if risk_flags:
            emotion_weight["risk"] = 1.4

        ranked: list[tuple[float, int, str]] = []
        for priority, label in enumerate(SUPPORT_PRIORITY):
            matches = signals & SUPPORT_RULES.get(label, set())
            if not matches:
                continue
            score = max(
                emotion_weight.get(signal, topic_weight.get(signal, 0.35))
                for signal in matches
            )
            if label in {"안전", "공감", "안정", "회복"} and matches & DISTRESS_EMOTIONS:
                score += 0.16
            if label in {"증폭", "음미", "축하", "영감"} and matches & POSITIVE_EMOTIONS:
                score += 0.14
            if label in {"유지", "배경", "가벼운 성찰"} and not (set(emotions) & (DISTRESS_EMOTIONS | POSITIVE_EMOTIONS)):
                score += 0.1
            ranked.append((score, -priority, label))

        ranked.sort(reverse=True)
        support = [label for _, _, label in ranked]
        if not support:
            support.append("탐색")
        return support

    def emotional_goal(
        self, emotions: list[str], desired_support: list[str]
    ) -> EmotionalGoal:
        desired: list[str] = []
        for support in desired_support:
            if support == "안전":
                desired.extend(["안전", "안정"])
            elif support == "공감":
                desired.extend(["이해받음"])
            elif support == "안정":
                desired.extend(["차분함", "긴장 완화"])
            elif support == "회복":
                desired.extend(["회복", "다시 움직일 여지"])
            elif support == "전환":
                desired.extend(["가벼운 환기", "기분 전환"])
            elif support == "증폭":
                desired.extend(["밝은 에너지", "즐거운 확장"])
            elif support == "축하":
                desired.extend(["작은 축하", "성취감 유지"])
            elif support == "음미":
                desired.extend(["좋은 여운", "차분한 만족"])
            elif support == "영감":
                desired.extend(["가능성", "다음 동기"])
            elif support == "유지":
                desired.extend(["잔잔한 흐름", "부담 없는 안정"])
            elif support == "발견":
                desired.extend(["가벼운 새로움", "취향 발견"])
            elif support == "배경":
                desired.extend(["저부담", "편안한 동행"])
            elif support == "가벼운 성찰":
                desired.extend(["가벼운 정리", "잔잔한 사유"])
            elif support == "동기부여":
                desired.extend(["자기효능감", "재도전"])
            elif support == "확장":
                desired.extend(["호기심", "확장"])
            elif support == "탐색":
                desired.extend(["가벼운 몰입"])
            elif support == "학습":
                desired.extend(["알아가는 즐거움", "탐구"])
            elif support == "몰입":
                desired.extend(["깊은 몰입", "관심사 확장"])
        transition = list(dict.fromkeys([*emotions[:2], *desired[:3]]))
        return EmotionalGoal(
            current_state=emotions[:3],
            desired_state=list(dict.fromkeys(desired))[:5],
            transition=transition,
        )

    def strategy(
        self,
        situation: Situation,
        desired_support: list[str],
        emotions: list[str],
        topics: list[str],
    ) -> RecommendationStrategy:
        first = desired_support[0] if desired_support else "탐색"
        avoid = ["감정 악화", "과도하게 우울한 분위기", "폭력적 전개"]
        if "분노" in emotions:
            avoid.append("공격성이나 복수심을 강화하는 콘텐츠")
        if "불안" in emotions:
            avoid.append("긴장도가 높은 콘텐츠")
        if "무기력" in emotions:
            avoid.append("지나치게 느리고 침잠하는 콘텐츠")
        if situation.risk_level == "high":
            avoid.extend(["상실을 미화하는 콘텐츠", "절망을 강화하는 콘텐츠"])
        if situation.intent == "positive_emotion_based":
            avoid.append("좋은 기분을 갑자기 무겁게 꺾는 콘텐츠")
        if situation.intent == "neutral_daily_based":
            avoid.extend(["과잉 해석을 부르는 무거운 콘텐츠", "인지 부담이 큰 콘텐츠"])
        if situation.intent == "mixed_emotion_based":
            avoid.append("에너지를 한쪽으로 과하게 밀어붙이는 콘텐츠")
        alternatives = self.alternatives(emotions, topics, situation)
        confidence = max(0.35, min(0.95, situation.confidence - 0.04 * len(alternatives)))
        return RecommendationStrategy(
            mode=situation.intent,
            emotional_arc=EMOTIONAL_ARCS.get(first, EMOTIONAL_ARCS["탐색"]),
            prioritize=list(dict.fromkeys(desired_support + situation.needs)),
            avoid=list(dict.fromkeys(avoid)),
            confidence=round(confidence, 3),
            alternative_interpretations=alternatives,
        )

    @staticmethod
    def alternatives(
        emotions: list[str], topics: list[str], situation: Situation
    ) -> list[str]:
        alternatives: list[str] = []
        if bool(set(emotions) & DISTRESS_EMOTIONS) and bool(set(emotions) & POSITIVE_EMOTIONS):
            alternatives.append("복합 감정: 긍정 사건과 부담이 공존")
        if situation.intent == "neutral_daily_based":
            alternatives.append("큰 사건보다 하루의 낮은 강도와 생활 맥락이 중심")
        if situation.intent == "positive_emotion_based" and "성취" in topics:
            alternatives.append("성취감과 안도감이 함께 남은 상태일 수 있음")
        if "관계" in topics and "분노" in emotions:
            alternatives.append("분노 표현보다 관계 정리 욕구일 수 있음")
        if "휴식" in topics and "무기력" in emotions:
            alternatives.append("휴식 욕구와 무기력 신호 구분 필요")
        if situation.primary_event == "일상 감정 기록" and len(emotions) >= 3:
            alternatives.append("명확한 사건보다 정서 상태 기록에 가까움")
        return alternatives[:3]

    @staticmethod
    def nuanced_emotion(
        emotions: list[str], topics: list[str], situation: Situation, goal: EmotionalGoal
    ) -> str:
        current = ", ".join(emotions[:3])
        desired = ", ".join(goal.desired_state[:3]) or "부담 없는 탐색"
        if situation.intent == "positive_emotion_based":
            return (
                f"{situation.primary_event} 뒤에 {current}이 남아 있고, "
                f"그 좋은 결을 {desired} 쪽으로 이어가려는 흐름이 보입니다."
            )
        if situation.intent == "neutral_daily_based":
            return (
                f"{situation.primary_event}처럼 큰 감정 변화보다 생활의 온도가 중심인 기록이며, "
                f"{desired} 정도의 낮은 부담이 잘 맞는 상태로 해석됩니다."
            )
        if situation.intent == "interest_based":
            return (
                f"{situation.primary_event}에서 {current}과 관심사가 함께 드러나며, "
                f"{desired} 방향으로 넓혀가고 싶은 기록에 가깝습니다."
            )
        if situation.valence == "mixed":
            return (
                f"{situation.primary_event} 속에서 {current}이 함께 나타나며, "
                f"현재 감정을 단순히 해소하기보다 {desired} 상태로 이동하려는 욕구가 보입니다."
            )
        return (
            f"{situation.primary_event} 상황에서 {current} 감정이 두드러지고, "
            f"문화 콘텐츠를 통해 {desired}을 얻고 싶은 상태로 해석됩니다."
        )
