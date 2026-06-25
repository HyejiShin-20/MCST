from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceDetail(BaseModel):
    label: str
    score: float
    intensity: str
    evidence: list[str] = Field(default_factory=list)


class EmotionalGoal(BaseModel):
    current_state: list[str] = Field(default_factory=list)
    desired_state: list[str] = Field(default_factory=list)
    transition: list[str] = Field(default_factory=list)


class Situation(BaseModel):
    primary_event: str
    intent: str
    needs: list[str] = Field(default_factory=list)
    stressors: list[str] = Field(default_factory=list)
    risk_level: str
    valence: str
    arousal: str
    intensity: float
    intensity_label: str
    support_priority: list[str] = Field(default_factory=list)
    confidence: float
    uncertainty_reasons: list[str] = Field(default_factory=list)


class RecommendationStrategy(BaseModel):
    mode: str
    emotional_arc: list[str] = Field(default_factory=list)
    prioritize: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    confidence: float
    alternative_interpretations: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    emotions: list[str]
    emotion_scores: dict[str, float]
    emotion_details: list[EvidenceDetail]
    topics: list[str]
    topic_details: list[EvidenceDetail]
    context_keywords: list[str]
    desired_support: list[str]
    situation: Situation
    recommendation_strategy: RecommendationStrategy
    emotional_goal: EmotionalGoal
    nuanced_emotion: str
    confidence: float
    uncertainty: list[str] = Field(default_factory=list)
    summary: str
    risk_flags: list[str] = Field(default_factory=list)
    backend: str
    model_signals: list[dict[str, Any]] = Field(default_factory=list)

