from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.text_utils import top_items


POSITIVE_SIGNALS = {"helpful", "saved", "liked", "completed"}
NEGATIVE_SIGNALS = {"not_helpful", "skipped", "disliked"}


class MemoryService:
    def build_profile(
        self,
        user_id: int,
        analyses: list[dict[str, Any]],
        feedback_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        emotions: list[str] = []
        topics: list[str] = []
        supports: list[str] = []

        for analysis in analyses:
            emotions.extend(analysis.get("emotions", []))
            topics.extend(analysis.get("topics", []))
            topics.extend(analysis.get("context_keywords", []))
            supports.extend(analysis.get("desired_support", []))

        feedback_profile = self._feedback_profile(feedback_events or [])
        preferred_tone = self._preferred_tone(supports, emotions)
        preferred_tone = top_items(
            preferred_tone + feedback_profile["positive_tones"],
            6,
        )
        return {
            "user_id": user_id,
            "entry_count": len(analyses),
            "feedback_count": feedback_profile["positive_count"] + feedback_profile["negative_count"],
            "long_term_emotion_pattern": top_items(emotions, 6),
            "long_term_interest_profile": top_items(topics, 8),
            "preferred_content_tone": preferred_tone,
            "disliked_content_tone": top_items(
                feedback_profile["negative_tones"]
                + ["과도하게 우울한 분위기", "폭력적 전개", "감정 악화"],
                6,
            ),
            "content_feedback_profile": feedback_profile,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _preferred_tone(supports: list[str], emotions: list[str]) -> list[str]:
        tones: list[str] = []
        if "회복" in supports:
            tones.append("잔잔한 회복")
        if "공감" in supports:
            tones.append("따뜻한 위로")
        if "동기부여" in supports:
            tones.append("현실적인 성장")
        if "확장" in supports:
            tones.append("호기심을 넓히는 탐험")
        if "평온" in emotions:
            tones.append("차분한 사유")
        return tones or ["일상에 부담 없이 스며드는 콘텐츠"]

    @staticmethod
    def _feedback_profile(feedback_events: list[dict[str, Any]]) -> dict[str, Any]:
        positive_emotions: list[str] = []
        positive_topics: list[str] = []
        positive_tones: list[str] = []
        positive_types: list[str] = []
        negative_emotions: list[str] = []
        negative_topics: list[str] = []
        negative_tones: list[str] = []
        negative_types: list[str] = []
        positive_count = 0
        negative_count = 0

        for event in feedback_events:
            signal = event.get("signal", "")
            rating = event.get("rating")
            metadata = event.get("metadata", {})
            content = metadata.get("content", {})
            affect_profile = metadata.get("content_affect_profile", {})
            is_positive = signal in POSITIVE_SIGNALS or (
                isinstance(rating, (int, float)) and rating >= 4
            )
            is_negative = signal in NEGATIVE_SIGNALS or (
                isinstance(rating, (int, float)) and rating <= 2
            )
            if not is_positive and not is_negative:
                continue

            emotion_tags = content.get("emotion_tags", [])
            topic_tags = content.get("topic_tags", [])
            tones = affect_profile.get("dominant_tone", [])
            content_type = content.get("content_type") or event.get("content_type")

            if is_positive:
                positive_count += 1
                positive_emotions.extend(emotion_tags)
                positive_topics.extend(topic_tags)
                positive_tones.extend(tones)
                if content_type:
                    positive_types.append(content_type)
            if is_negative:
                negative_count += 1
                negative_emotions.extend(emotion_tags)
                negative_topics.extend(topic_tags)
                negative_tones.extend(tones)
                if content_type:
                    negative_types.append(content_type)

        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_emotion_tags": top_items(positive_emotions, 8),
            "positive_topic_tags": top_items(positive_topics, 10),
            "positive_tones": top_items(positive_tones, 6),
            "preferred_content_types": top_items(positive_types, 3),
            "negative_emotion_tags": top_items(negative_emotions, 8),
            "negative_topic_tags": top_items(negative_topics, 10),
            "negative_tones": top_items(negative_tones, 6),
            "avoided_content_types": top_items(negative_types, 3),
        }
