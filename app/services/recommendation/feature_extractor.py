from __future__ import annotations

from typing import Any

from app.schemas.recommendation import CandidateFeatures, SafetyDecision
from app.services.analysis_rules import DISTRESS_EMOTIONS, NEUTRAL_EMOTIONS, POSITIVE_EMOTIONS
from app.services.recommendation.content_profile import build_content_affect_profile
from app.services.recommendation.rules import (
    LOW_TENSION_TAGS,
    RESTORATIVE_TAGS,
    ROLE_TAGS,
    SITUATION_TAGS,
)
from app.services.text_utils import cosine_similarity, weighted_overlap


class FeatureExtractor:
    def extract(
        self,
        *,
        role: str,
        query_vector: Any,
        content_vector: Any,
        analysis: dict,
        content: dict,
        recent_terms: list[str],
        profile_terms: list[str],
        negative_profile_terms: list[str],
        safety: SafetyDecision,
    ) -> CandidateFeatures:
        affect_profile = content.get("affect_profile") or build_content_affect_profile(content)
        content_terms = (
            content["topic_tags"]
            + content["emotion_tags"]
            + content.get("mood_tags", [])
            + content.get("recommendation_roles", [])
            + affect_profile.get("dominant_tone", [])
            + affect_profile.get("regulation_effects", [])
        )
        role_fit = weighted_overlap(ROLE_TAGS.get(role, [role]), content_terms)
        input_priority_fit = self.input_priority_fit(analysis, content, affect_profile)
        features = CandidateFeatures(
            semantic_similarity=cosine_similarity(query_vector, content_vector),
            emotion_match=weighted_overlap(
                target_emotions_for_match(analysis), content["emotion_tags"]
            ),
            topic_match=weighted_overlap(
                analysis.get("topics", []) + analysis.get("context_keywords", []),
                content["topic_tags"],
            ),
            situation_fit=self.situation_fit(analysis, content),
            support_fit=self.support_fit(analysis, content),
            affect_alignment=self.affect_alignment(analysis, content, affect_profile, role),
            emotion_regulation=self.emotion_regulation(
                analysis, content, affect_profile, role, safety
            ),
            recovery_potential=affect_profile["recovery_potential"],
            calm_fit=self.calm_fit(analysis, affect_profile, role),
            cognitive_load_fit=self.cognitive_load_fit(analysis, affect_profile, role),
            input_priority_fit=input_priority_fit,
            valence_fit=self.valence_fit(analysis, content, affect_profile),
            tag_confidence_fit=self.tag_confidence_fit(content),
            context_mismatch_penalty=self.context_mismatch_penalty(
                analysis, content, affect_profile, input_priority_fit
            ),
            role_fit=role_fit,
            safety_score=safety.safety_score,
            personal_fit=weighted_overlap(profile_terms, content_terms),
            preference_penalty=weighted_overlap(negative_profile_terms, content_terms),
            recent_context=weighted_overlap(recent_terms, content_terms),
            novelty=self.novelty(content, recent_terms, profile_terms),
            tension_penalty=self.tension_penalty(analysis, content, affect_profile),
            darkness_penalty=self.darkness_penalty(analysis, affect_profile),
        )
        return features

    @staticmethod
    def input_priority_fit(analysis: dict, content: dict, affect_profile: dict) -> float:
        emotions = set(analysis.get("emotions", []))
        valence = analysis.get("structured_emotion", {}).get("valence") or analysis.get("valence")
        content_terms = searchable_terms(content)
        content_text = content_search_text(content, content_terms)
        negative_or_tired = bool(emotions & (DISTRESS_EMOTIONS | {"피로"})) or valence == "negative"
        positive = bool(emotions & POSITIVE_EMOTIONS) and not negative_or_tired

        if negative_or_tired:
            target_terms: list[str] = []
            target_terms.extend(analysis.get("desired_support", []))
            target_terms.extend(analysis.get("emotional_goal", {}).get("desired_state", []))
            target_terms.extend(["회복", "위로", "휴식", "안정", "평온", "잠", "집", "저부담", "잔잔함", "안도감", "자기돌봄"])
            overlap = weighted_overlap(target_terms, content_terms)
            text_fit = text_cue_fit(
                content_text,
                [
                    "회복",
                    "위로",
                    "휴식",
                    "쉬",
                    "잠",
                    "편안",
                    "괜찮",
                    "안정",
                    "잔잔",
                    "따뜻",
                    "희망",
                    "힘든",
                    "나아가",
                    "응원",
                    "다시",
                ],
            )
            recovery = affect_profile.get("recovery_potential", 0.0)
            calming = affect_profile.get("calming_effect", 0.0)
            return round(max(overlap, text_fit, recovery * 0.68 + calming * 0.16), 4)

        if positive:
            target_terms = []
            target_terms.extend(analysis.get("emotions", []))
            target_terms.extend(analysis.get("topics", []))
            target_terms.extend(analysis.get("desired_support", []))
            target_terms.extend(analysis.get("emotional_goal", {}).get("desired_state", []))
            target_terms.extend(["축하", "음미", "기쁨", "즐거움", "성취", "희망", "밝음", "신남", "평온", "여유", "따뜻함"])
            overlap = weighted_overlap(target_terms, content_terms)
            activation = affect_profile.get("activation_level", 0.0)
            calming = affect_profile.get("calming_effect", 0.0)
            text_fit = text_cue_fit(
                content_text,
                ["기쁨", "즐거움", "행복", "희망", "밝", "웃", "편안", "따뜻", "여유", "응원"],
            )
            positive_calm = bool(emotions & {"평온", "안도감"}) or valence in {"positive", "positive_mixed"}
            affect_fit = activation * 0.48
            if positive_calm:
                affect_fit = max(affect_fit, calming * 0.38 + affect_profile.get("recovery_potential", 0.0) * 0.22)
            return round(max(overlap, text_fit, affect_fit), 4)

        target_terms = []
        target_terms.extend(analysis.get("topics", []))
        target_terms.extend(analysis.get("context_keywords", []))
        target_terms.extend(["일상", "저부담", "평온", "잔잔함", "발견", "가벼운 성찰"])
        overlap = weighted_overlap(target_terms, content_terms)
        calming = affect_profile.get("calming_effect", 0.0)
        load_fit = 1.0 - affect_profile.get("cognitive_load", 0.0)
        return round(max(overlap, calming * 0.35 + load_fit * 0.25), 4)

    @staticmethod
    def context_mismatch_penalty(
        analysis: dict,
        content: dict,
        affect_profile: dict,
        input_priority_fit: float,
    ) -> float:
        emotions = set(analysis.get("emotions", []))
        valence = analysis.get("structured_emotion", {}).get("valence") or analysis.get("valence")
        terms = set(searchable_terms(content))
        analysis_terms = set(analysis.get("emotions", []))
        analysis_terms.update(analysis.get("topics", []))
        analysis_terms.update(analysis.get("context_keywords", []))
        analysis_terms.update(analysis.get("structured_emotion", {}).get("context_keywords", []))
        content_text = content_search_text(content, list(terms))
        negative_or_tired = bool(emotions & (DISTRESS_EMOTIONS | {"피로"})) or valence == "negative"
        romantic_terms = {"사랑", "연애", "이별", "그리움", "연인", "고백", "설렘"}
        romantic_context = bool(analysis_terms & romantic_terms)
        penalty = 0.0

        if not romantic_context:
            if terms & {"이별", "그리움", "연인", "고백"}:
                penalty = max(penalty, 0.14)
            if _has_any(content_text, ["연인", "고백", "헤어", "이별", "잘가요", "눈물"]):
                penalty = max(penalty, 0.16)

        if negative_or_tired:
            content_emotions = set(content.get("emotion_tags", []))
            non_neutral_emotions = content_emotions - NEUTRAL_EMOTIONS
            recovery = affect_profile.get("recovery_potential", 0.0)
            if not non_neutral_emotions and input_priority_fit < 0.26 and recovery < 0.38:
                penalty = max(penalty, 0.11)
            if terms & {"설렘", "즐거움", "신남"} and input_priority_fit < 0.32:
                penalty = max(penalty, 0.1)
            if "피로" in emotions and terms & {"무기력", "우울", "고립", "상실"} and recovery < 0.55:
                penalty = max(penalty, 0.08)
        elif valence in {"positive", "positive_mixed"} or emotions & POSITIVE_EMOTIONS:
            positive_terms = {"기쁨", "즐거움", "희망", "평온", "따뜻함", "음미", "증폭", "축하", "영감", "활력형", "회복형", "안정형"}
            content_positive = bool(terms & positive_terms)
            content_valence = content.get("tag_metadata", {}).get("valence")
            tag_confidence = content.get("tag_confidence", 0.62)
            if not isinstance(content_valence, (int, float)):
                content_valence = 0.0
            try:
                tag_confidence = float(tag_confidence)
            except (TypeError, ValueError):
                tag_confidence = 0.62
            if not content_positive and input_priority_fit < 0.26:
                penalty = max(penalty, 0.1)
            if not content_positive and input_priority_fit < 0.36 and tag_confidence < 0.68:
                penalty = max(penalty, 0.16)
            if float(content_valence) < 0.08 and input_priority_fit < 0.38:
                penalty = max(penalty, 0.13)
            if terms & {"우울", "상실", "고립", "불안", "갈등"} and not (analysis_terms & terms):
                penalty = max(penalty, 0.12)

        return round(min(0.26, penalty), 4)

    @staticmethod
    def valence_fit(analysis: dict, content: dict, affect_profile: dict) -> float:
        valence = analysis.get("structured_emotion", {}).get("valence") or analysis.get("valence")
        emotions = set(analysis.get("emotions", []))
        content_valence = content.get("tag_metadata", {}).get("valence")
        if not isinstance(content_valence, (int, float)):
            content_valence = 0.0
        content_valence = max(-1.0, min(1.0, float(content_valence)))
        recovery = affect_profile.get("recovery_potential", 0.0)
        calming = affect_profile.get("calming_effect", 0.0)

        if valence in {"positive", "positive_mixed"} or emotions & POSITIVE_EMOTIONS:
            return round(max(0.0, min(1.0, (content_valence + 0.35) / 1.35)), 4)
        if valence == "negative" or emotions & DISTRESS_EMOTIONS:
            return round(max((content_valence + 1.0) / 2.0, recovery * 0.82, calming * 0.58), 4)
        return round(max(0.0, min(1.0, 1.0 - abs(content_valence) * 0.72)), 4)

    @staticmethod
    def tag_confidence_fit(content: dict) -> float:
        confidence = content.get("tag_confidence", 0.62)
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            value = 0.62
        return round(max(0.0, min(1.0, value)), 4)

    @staticmethod
    def situation_fit(analysis: dict, content: dict) -> float:
        situation = analysis.get("situation", {})
        primary_event = situation.get("primary_event", "일상 감정 기록")
        target_tags = list(SITUATION_TAGS.get(primary_event, []))
        target_tags.extend(situation.get("stressors", []))
        target_tags.extend(analysis.get("topics", []))
        return weighted_overlap(target_tags, searchable_terms(content))

    @staticmethod
    def support_fit(analysis: dict, content: dict) -> float:
        target_tags: list[str] = []
        target_tags.extend(analysis.get("desired_support", []))
        target_tags.extend(analysis.get("recommendation_strategy", {}).get("emotional_arc", []))
        target_tags.extend(analysis.get("emotional_goal", {}).get("desired_state", []))
        return weighted_overlap(target_tags, searchable_terms(content))

    @staticmethod
    def emotion_regulation(
        analysis: dict,
        content: dict,
        affect_profile: dict,
        role: str,
        safety: SafetyDecision,
    ) -> float:
        content_terms = set(searchable_terms(content))
        desired_state = analysis.get("emotional_goal", {}).get("desired_state", [])
        desired_overlap = weighted_overlap(desired_state, searchable_terms(content))
        restorative = bool(content_terms & RESTORATIVE_TAGS)
        low_tension = bool(content_terms & LOW_TENSION_TAGS)
        recovery = affect_profile.get("recovery_potential", 0.0)
        calming = affect_profile.get("calming_effect", 0.0)
        activation = affect_profile.get("activation_level", 0.0)
        base = 0.28 + desired_overlap * 0.24 + recovery * 0.18 + calming * 0.12
        if role in {"안정", "안전"} and low_tension:
            base += 0.22
        if role in {"회복", "재도전"} and restorative:
            base += 0.2
        if role in {"재도전", "자기효능감", "확장"}:
            base += activation * 0.1
        if role in {"증폭", "축하", "영감"}:
            base += activation * 0.16
        if role in {"음미", "유지", "배경", "가벼운 성찰"}:
            base += calming * 0.12
        if role == "공감":
            base += weighted_overlap(analysis.get("emotions", []), content["emotion_tags"]) * 0.2
        base = base * (0.75 + safety.safety_score * 0.25)
        return max(0.0, min(1.0, base))

    @staticmethod
    def affect_alignment(
        analysis: dict,
        content: dict,
        affect_profile: dict,
        role: str,
    ) -> float:
        target_terms: list[str] = []
        target_terms.extend(analysis.get("desired_support", []))
        target_terms.extend(analysis.get("recommendation_strategy", {}).get("emotional_arc", []))
        target_terms.extend(analysis.get("emotional_goal", {}).get("desired_state", []))
        profile_terms: list[str] = []
        profile_terms.extend(affect_profile.get("dominant_tone", []))
        profile_terms.extend(affect_profile.get("regulation_effects", []))
        profile_terms.extend(content.get("emotion_tags", []))
        profile_terms.extend(content.get("topic_tags", []))
        profile_terms.extend(content.get("mood_tags", []))
        profile_terms.extend(content.get("recommendation_roles", []))
        score = weighted_overlap(target_terms, profile_terms)
        if role in {"안전", "안정"}:
            score = max(score, affect_profile.get("calming_effect", 0.0))
        if role == "회복":
            score = max(score, affect_profile.get("recovery_potential", 0.0))
        if role in {"재도전", "자기효능감", "확장", "증폭", "축하", "영감", "탐색", "학습"}:
            score = max(score, affect_profile.get("activation_level", 0.0))
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def calm_fit(analysis: dict, affect_profile: dict, role: str) -> float:
        calming = affect_profile.get("calming_effect", 0.0)
        tension = affect_profile.get("tension_level", 0.0)
        emotions = set(analysis.get("emotions", []))
        high_arousal = bool(emotions & {"불안", "분노", "창피함", "두려움", "압박감"})
        if role in {"안전", "안정", "가벼운 탐색", "유지", "배경", "음미", "가벼운 성찰"}:
            return round(max(calming, 1.0 - tension), 4)
        if high_arousal:
            return round(calming * 0.75 + (1.0 - tension) * 0.25, 4)
        return round(calming * 0.6, 4)

    @staticmethod
    def cognitive_load_fit(analysis: dict, affect_profile: dict, role: str) -> float:
        load = affect_profile.get("cognitive_load", 0.0)
        emotions = set(analysis.get("emotions", []))
        vulnerable = bool(emotions & {"슬픔", "불안", "무기력", "피로", "외로움", "창피함", "후회"})
        if role in {"확장", "호기심", "탐색", "학습", "몰입"} and not vulnerable:
            return round(0.45 + load * 0.45, 4)
        if role in {"유지", "발견", "배경", "가벼운 성찰"}:
            return round(max(0.0, min(1.0, 1.0 - load)), 4)
        if role in {"재도전", "자기효능감"} and not {"무기력", "불안"} & emotions:
            return round(0.55 + (1.0 - abs(load - 0.52)) * 0.25, 4)
        return round(max(0.0, min(1.0, 1.0 - load)), 4)

    @staticmethod
    def novelty(content: dict, recent_terms: list[str], profile_terms: list[str]) -> float:
        overlap = weighted_overlap(recent_terms + profile_terms, searchable_terms(content))
        return round(max(0.0, min(1.0, 1.0 - overlap)), 4)

    @staticmethod
    def tension_penalty(analysis: dict, content: dict, affect_profile: dict | None = None) -> float:
        emotions = set(analysis.get("emotions", []))
        terms = set(searchable_terms(content))
        analysis_terms = set(analysis.get("emotions", []))
        analysis_terms.update(analysis.get("topics", []))
        analysis_terms.update(analysis.get("context_keywords", []))
        analysis_terms.update(analysis.get("structured_emotion", {}).get("context_keywords", []))
        affect_profile = affect_profile or build_content_affect_profile(content)
        tension_level = affect_profile.get("tension_level", 0.0)
        title_text = str(content.get("title", ""))
        body_text = " ".join(
            [
                title_text,
                str(content.get("summary", ""))[:800],
            ]
        )
        content_text = " ".join([body_text, " ".join(terms)])
        penalty = 0.0
        if "불안" in emotions and terms & {"긴장", "압박", "생존", "두려움"}:
            penalty = max(penalty, 0.16)
        if "무기력" in emotions and terms & {"우울", "고립", "상실"}:
            penalty = max(penalty, 0.12)
        if "분노" in emotions and terms & {"분노", "폭력", "복수"}:
            penalty = max(penalty, 0.14)
        if "불안" in emotions and tension_level >= 0.54:
            penalty = max(penalty, min(0.2, (tension_level - 0.46) * 0.32))
        relationship_terms = {"사랑", "연애", "이별", "그리움", "연인", "고백"}
        romantic_context = bool(analysis_terms & relationship_terms)
        primary_event = str(analysis.get("situation", {}).get("primary_event", ""))
        if "관계 상실" in primary_event or "이별" in primary_event:
            romantic_context = True
        if not romantic_context:
            farewell_cue = _has_any(
                content_text,
                ["잘가요", "안녕", "이별", "헤어", "떠나", "그리움", "작별"],
            )
            romance_cue = _has_any(content_text, ["연인", "고백", "사랑"])
            lonely_cue = _has_any(content_text, ["외롭", "쓸쓸"])
            strong_title_mismatch = _has_any(
                title_text,
                ["연인", "고백", "이별", "잘가요", "외롭", "응급실", "눈물"],
            )
            if content.get("content_type") == "music" and strong_title_mismatch:
                penalty = max(penalty, 0.34)
            if terms & {"이별", "그리움", "연인", "고백"} or farewell_cue:
                penalty = max(penalty, 0.22)
            elif content.get("content_type") == "music" and romance_cue:
                penalty = max(penalty, 0.16)
            elif content.get("content_type") == "music" and terms & {"사랑", "연애", "설렘"}:
                penalty = max(penalty, 0.11)
            elif terms & {"연애", "설렘"}:
                penalty = max(penalty, 0.08)
            if lonely_cue and not (analysis_terms & {"외로움", "쓸쓸함", "고립"}):
                penalty = max(penalty, 0.22)
        if "우주" in analysis_terms and not _has_any(
            body_text,
            ["우주", "천문", "행성", "별자리", "우주비행", "탐험", "정재승", "코스모스"],
        ):
            penalty = max(penalty, 0.42)
        elif not (analysis_terms & {"우주", "과학", "학습", "호기심", "탐험"}):
            if terms & {"우주", "과학"}:
                penalty = max(penalty, 0.14)
        if "불안" not in emotions and terms & {"불안", "긴장감", "압박"}:
            penalty = max(penalty, 0.07)
        if "분노" not in emotions and terms & {"분노", "강렬함"}:
            penalty = max(penalty, 0.09)
        if not (emotions & {"분노", "상처", "슬픔", "불안"}) and terms & {"갈등", "상처"}:
            penalty = max(penalty, 0.13)
        if "피로" in emotions and terms & {"우울", "고립", "상실"}:
            penalty = max(penalty, 0.08)
        return round(penalty, 4)

    @staticmethod
    def darkness_penalty(analysis: dict, affect_profile: dict) -> float:
        emotions = set(analysis.get("emotions", []))
        darkness = affect_profile.get("darkness_level", 0.0)
        recovery = affect_profile.get("recovery_potential", 0.0)
        vulnerable = bool(emotions & {"슬픔", "무기력", "외로움", "후회", "불안"})
        if not vulnerable or darkness < 0.48 or recovery >= 0.58:
            return 0.0
        return round(min(0.18, (darkness - 0.42) * 0.32), 4)


def searchable_terms(content: dict) -> list[str]:
    return (
        content.get("topic_tags", [])
        + content.get("emotion_tags", [])
        + content.get("mood_tags", [])
        + content.get("recommendation_roles", [])
    )


def target_emotions_for_match(analysis: dict) -> list[str]:
    emotions = [str(emotion) for emotion in analysis.get("emotions", []) if str(emotion).strip()]
    emotion_set = set(emotions)
    if emotion_set & (DISTRESS_EMOTIONS | {"피로"}):
        focused = [emotion for emotion in emotions if emotion not in NEUTRAL_EMOTIONS]
        return focused or emotions
    return emotions


def content_search_text(content: dict, terms: list[str]) -> str:
    return " ".join(
        [
            str(content.get("title", "")),
            str(content.get("summary", ""))[:900],
            str(content.get("genre", "")),
            " ".join(terms),
        ]
    )


def text_cue_fit(text: str, cues: list[str]) -> float:
    if not text:
        return 0.0
    matches = sum(1 for cue in cues if cue and cue in text)
    return round(min(1.0, matches / max(3, len(cues) * 0.45)), 4)


def _has_any(text: str, cues: list[str]) -> bool:
    return any(cue in text for cue in cues)
