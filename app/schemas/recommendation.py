from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationSlot(BaseModel):
    role: str
    purpose: str
    target_effect: str
    weights: dict[str, float]


class CandidateFeatures(BaseModel):
    semantic_similarity: float = 0.0
    emotion_match: float = 0.0
    topic_match: float = 0.0
    situation_fit: float = 0.0
    support_fit: float = 0.0
    affect_alignment: float = 0.0
    emotion_regulation: float = 0.0
    recovery_potential: float = 0.0
    calm_fit: float = 0.0
    cognitive_load_fit: float = 0.0
    input_priority_fit: float = 0.0
    valence_fit: float = 0.0
    tag_confidence_fit: float = 0.0
    context_mismatch_penalty: float = 0.0
    role_fit: float = 0.0
    safety_score: float = 0.0
    personal_fit: float = 0.0
    preference_penalty: float = 0.0
    recent_context: float = 0.0
    novelty: float = 0.0
    tension_penalty: float = 0.0
    darkness_penalty: float = 0.0
    diversity_adjustment: float = 0.0


class SafetyDecision(BaseModel):
    allowed: bool = True
    safety_score: float = 1.0
    penalty: float = 0.0
    reasons: list[str] = Field(default_factory=list)
