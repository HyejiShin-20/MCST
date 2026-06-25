from __future__ import annotations

from app.schemas.recommendation import CandidateFeatures, RecommendationSlot, SafetyDecision


class SlotScorer:
    def score(
        self,
        slot: RecommendationSlot,
        features: CandidateFeatures,
        safety: SafetyDecision,
    ) -> float:
        values = features.model_dump()
        score = 0.0
        for key, weight in slot.weights.items():
            score += values.get(key, 0.0) * weight
        score += min(0.08, features.recent_context * 0.03)
        score += min(0.14, features.input_priority_fit * 0.12)
        score += min(0.07, features.valence_fit * 0.05)
        score += min(0.04, features.tag_confidence_fit * 0.035)
        if features.input_priority_fit < 0.22 and features.emotion_match < 0.08 and features.support_fit < 0.1:
            score -= 0.09
        if features.valence_fit < 0.34 and features.input_priority_fit < 0.32:
            score -= 0.06
        if features.tag_confidence_fit < 0.66 and features.input_priority_fit < 0.34:
            score -= 0.035
        score -= min(0.12, features.preference_penalty * 0.12)
        score -= features.tension_penalty
        score -= features.context_mismatch_penalty
        score -= features.darkness_penalty
        score -= safety.penalty
        return round(max(0.0, min(1.0, score)), 4)
