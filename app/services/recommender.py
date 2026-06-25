from __future__ import annotations

from typing import Any

from app import config
from app.schemas.recommendation import RecommendationSlot
from app.services.embedding_service import EmbeddingService
from app.services.recommendation.candidate_retriever import CandidateRetriever
from app.services.recommendation.chroma_store import ChromaCandidateStore
from app.services.recommendation.content_profile import enrich_content_with_affect_profile
from app.services.recommendation.embedding_index import ContentEmbeddingIndex
from app.services.recommendation.explanation_builder import ExplanationBuilder
from app.services.recommendation.feature_extractor import FeatureExtractor
from app.services.recommendation.portfolio_builder import PortfolioBuilder
from app.services.recommendation.safety_policy import SafetyPolicy
from app.services.recommendation.slot_scorer import SlotScorer
from app.services.recommendation.strategy_builder import StrategyBuilder
from app.services.text_utils import weighted_overlap


class Recommender:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        strategy_builder: StrategyBuilder | None = None,
        candidate_retriever: CandidateRetriever | None = None,
        safety_policy: SafetyPolicy | None = None,
        feature_extractor: FeatureExtractor | None = None,
        slot_scorer: SlotScorer | None = None,
        portfolio_builder: PortfolioBuilder | None = None,
        explanation_builder: ExplanationBuilder | None = None,
        content_embedding_index: ContentEmbeddingIndex | None = None,
        chroma_store: ChromaCandidateStore | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.strategy_builder = strategy_builder or StrategyBuilder()
        self.candidate_retriever = candidate_retriever or CandidateRetriever()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.slot_scorer = slot_scorer or SlotScorer()
        self.portfolio_builder = portfolio_builder or PortfolioBuilder()
        self.explanation_builder = explanation_builder or ExplanationBuilder()
        self.content_embedding_index = content_embedding_index or ContentEmbeddingIndex(
            self.embedding_service
        )
        self.chroma_store = chroma_store or ChromaCandidateStore(self.embedding_service)

    def recommend(
        self,
        entry: dict[str, Any],
        analysis: dict[str, Any],
        contents: list[dict[str, Any]],
        profile: dict[str, Any] | None,
        recent_analyses: list[dict[str, Any]],
        per_type: int = 4,
    ) -> dict[str, Any]:
        slots = self.strategy_builder.build_slots(analysis, per_type)
        query_text = self._query_text(entry, analysis)
        raw_candidates = self.candidate_retriever.retrieve(contents)
        retrieval_stats: dict[str, Any] = {
            "backend": "memory",
            "input_count": len(raw_candidates),
            "candidate_count": len(raw_candidates),
        }
        if config.RETRIEVAL_BACKEND == "chroma":
            chroma_result = self.chroma_store.search(
                query_text=query_text,
                contents=raw_candidates,
                top_k=max(config.CHROMA_TOP_K, per_type * len(set(content["content_type"] for content in raw_candidates))),
            )
            raw_candidates = chroma_result.contents
            retrieval_stats = chroma_result.stats
        candidates = [
            enrich_content_with_affect_profile(content)
            for content in raw_candidates
        ]
        query_vector = self.embedding_service.encode([query_text]).vectors[0]
        content_embedding_batch = self.content_embedding_index.encode(candidates)
        content_vectors = content_embedding_batch.vectors
        recent_terms = self._recent_terms(recent_analyses)
        profile_terms = self._profile_terms(profile)
        negative_profile_terms = self._negative_profile_terms(profile)

        scored_items: list[dict[str, Any]] = []
        blocked_count = 0
        for content, content_vector in zip(candidates, content_vectors):
            safety = self.safety_policy.evaluate(analysis, content)
            if not safety.allowed:
                blocked_count += 1
                continue
            slot_scores: dict[str, float] = {}
            slot_features: dict[str, dict[str, float]] = {}
            target_shifts: dict[str, dict[str, list[str]]] = {}
            for slot in slots:
                features = self.feature_extractor.extract(
                    role=slot.role,
                    query_vector=query_vector,
                    content_vector=content_vector,
                    analysis=analysis,
                    content=content,
                    recent_terms=recent_terms,
                    profile_terms=profile_terms,
                    negative_profile_terms=negative_profile_terms,
                    safety=safety,
                )
                slot_scores[slot.role] = self.slot_scorer.score(slot, features, safety)
                slot_features[slot.role] = features.model_dump()
                target_shifts[slot.role] = self._target_shift(slot.role, analysis)

            representative_role = max(slot_scores, key=slot_scores.get)
            representative_features = slot_features[representative_role]
            base_score = max(slot_scores.values()) if slot_scores else 0.0
            scored_items.append(
                {
                    "content_id": content["content_id"],
                    "content_type": content["content_type"],
                    "type_label": self._type_label(content["content_type"]),
                    "title": content["title"],
                    "creator": content["creator"],
                    "genre": content["genre"],
                    "source": content["source"],
                    "base_score": round(base_score, 4),
                    "score": round(base_score, 4),
                    "score_percent": round(base_score * 100),
                    "score_components": representative_features,
                    "slot_scores": slot_scores,
                    "slot_features": slot_features,
                    "target_shifts": target_shifts,
                    "safety_reasons": safety.reasons,
                    "content_affect_profile": content["affect_profile"],
                    "emotion_tags": content["emotion_tags"],
                    "topic_tags": content["topic_tags"],
                    "summary": content["summary"],
                }
            )

        grouped = self.portfolio_builder.build(scored_items, slots, per_type)
        for recommendations in grouped.values():
            for item in recommendations:
                item["display_tags"] = self._display_tags(analysis, item)
                item["reason"] = self.explanation_builder.build(analysis, item)
                item.pop("slot_scores", None)
                item.pop("slot_features", None)
                item.pop("target_shifts", None)

        return {
            "entry_id": entry["entry_id"],
            "user_id": entry["user_id"],
            "embedding_backend": self.embedding_service.backend,
            "retrieval_backend": retrieval_stats.get("backend", "memory"),
            "algorithm_profile": self._algorithm_profile(
                slots, blocked_count, content_embedding_batch.stats, retrieval_stats
            ),
            "recommendation_strategy": analysis.get("recommendation_strategy", {}),
            "recommendation_roles": [slot.role for slot in slots],
            "recommendation_slots": [slot.model_dump() for slot in slots],
            "recommendations": grouped,
        }

    @staticmethod
    def _query_text(entry: dict[str, Any], analysis: dict[str, Any]) -> str:
        situation = analysis.get("situation", {})
        strategy = analysis.get("recommendation_strategy", {})
        emotional_goal = analysis.get("emotional_goal", {})
        structured = analysis.get("structured_emotion", {})
        context_keywords = list(
            dict.fromkeys(
                structured.get("context_keywords", [])
                or analysis.get("context_keywords", [])
            )
        )[:6]
        needs = list(situation.get("needs", []))[:3]
        desired_state = list(emotional_goal.get("desired_state", []))[:4]
        emotional_arc = list(strategy.get("emotional_arc", []))[:4]
        parts = [
            entry["processed_text"],
            "핵심 감정: "
            + str(structured.get("dominant_emotion") or ", ".join(analysis.get("emotions", [])[:1])),
            "보조 감정: " + ", ".join(structured.get("sub_emotions", analysis.get("emotions", [])[1:4])),
            "에너지: " + str(structured.get("energy_level", "")),
            "감정 근거: " + evidence_text(analysis.get("emotion_details", [])),
            "주제: " + ", ".join(analysis.get("topics", [])),
            "실제 활동/맥락: " + ", ".join(context_keywords),
            "상황: " + str(situation.get("primary_event", "")),
            "의도: " + str(situation.get("intent", "")),
            "필요: " + ", ".join(needs),
            "목표 정서: " + ", ".join(desired_state),
            "추천 흐름: " + ", ".join(emotional_arc),
            "복합 감정 설명: " + str(analysis.get("nuanced_emotion", "")),
        ]
        return "\n".join(parts)

    @staticmethod
    def _recent_terms(recent_analyses: list[dict[str, Any]]) -> list[str]:
        terms: list[str] = []
        for analysis in recent_analyses[:5]:
            terms.extend(analysis.get("emotions", []))
            terms.extend(analysis.get("topics", []))
            terms.extend(analysis.get("context_keywords", []))
            terms.extend(analysis.get("desired_support", []))
            terms.extend(analysis.get("emotional_goal", {}).get("desired_state", []))
        return terms

    @staticmethod
    def _profile_terms(profile: dict[str, Any] | None) -> list[str]:
        if not profile:
            return []
        terms: list[str] = []
        terms.extend(profile.get("long_term_emotion_pattern", []))
        terms.extend(profile.get("long_term_interest_profile", []))
        terms.extend(profile.get("preferred_content_tone", []))
        feedback_profile = profile.get("content_feedback_profile", {})
        terms.extend(feedback_profile.get("positive_emotion_tags", []))
        terms.extend(feedback_profile.get("positive_topic_tags", []))
        terms.extend(feedback_profile.get("positive_tones", []))
        terms.extend(feedback_profile.get("preferred_content_types", []))
        return terms

    @staticmethod
    def _negative_profile_terms(profile: dict[str, Any] | None) -> list[str]:
        if not profile:
            return []
        terms: list[str] = []
        terms.extend(profile.get("disliked_content_tone", []))
        feedback_profile = profile.get("content_feedback_profile", {})
        terms.extend(feedback_profile.get("negative_emotion_tags", []))
        terms.extend(feedback_profile.get("negative_topic_tags", []))
        terms.extend(feedback_profile.get("negative_tones", []))
        terms.extend(feedback_profile.get("avoided_content_types", []))
        return terms

    @staticmethod
    def _target_shift(role: str, analysis: dict[str, Any]) -> dict[str, list[str]]:
        goal = analysis.get("emotional_goal", {})
        current = goal.get("current_state") or analysis.get("emotions", [])[:3]
        role_targets = {
            "안전": ["안전", "안정"],
            "공감": ["이해받음", "정서적 완충"],
            "안정": ["차분함", "긴장 완화"],
            "회복": ["회복", "정서적 여지"],
            "재도전": ["자기효능감", "재도전"],
            "자기효능감": ["자기효능감"],
            "호기심": ["호기심", "확장"],
            "몰입": ["몰입", "전환"],
            "확장": ["확장", "탐색"],
            "가벼운 탐색": ["가벼운 몰입"],
            "전환": ["기분 전환", "가벼움"],
            "증폭": ["좋은 기분 확장"],
            "축하": ["성취감 축하"],
            "음미": ["좋은 여운", "차분한 만족"],
            "영감": ["다음 가능성", "동기"],
            "유지": ["잔잔한 유지"],
            "발견": ["새 취향 발견"],
            "배경": ["저부담 동행"],
            "가벼운 성찰": ["일상 정리"],
            "탐색": ["관심사 탐색"],
            "학습": ["알아가는 즐거움"],
        }
        return {"from": current, "to": role_targets.get(role, [role])}

    @staticmethod
    def _type_label(content_type: str) -> str:
        return {"movie": "영화", "music": "음악", "book": "도서"}.get(
            content_type, content_type
        )

    @staticmethod
    def _display_tags(analysis: dict[str, Any], item: dict[str, Any]) -> list[str]:
        analysis_terms = set(analysis.get("emotions", []))
        analysis_terms.update(analysis.get("topics", []))
        analysis_terms.update(analysis.get("context_keywords", []))
        analysis_terms.update(
            analysis.get("structured_emotion", {}).get("context_keywords", [])
        )
        situation = analysis.get("situation", {})
        analysis_terms.update(situation.get("needs", []))
        analysis_terms.update(analysis.get("emotional_goal", {}).get("desired_state", []))
        primary_event = str(situation.get("primary_event", ""))
        romantic_context = bool(
            analysis_terms
            & {"사랑", "연애", "이별", "그리움", "연인", "고백", "설렘"}
        ) or "이별" in primary_event or "관계 상실" in primary_event

        hidden_unless_matched = {
            "분노",
            "불안",
            "무기력",
            "슬픔",
            "외로움",
            "상처",
            "갈등",
            "그리움",
            "이별",
            "연애",
            "연인",
            "고백",
        }
        tags: list[str] = []
        candidates: list[str] = []
        candidates.extend(item.get("emotion_tags", []))
        candidates.extend(item.get("topic_tags", []))
        candidates.extend(item.get("content_affect_profile", {}).get("dominant_tone", []))
        candidates.append(item.get("genre", ""))
        candidates.append(item.get("recommendation_role", ""))

        for tag in candidates:
            tag = str(tag or "").strip()
            if not tag:
                continue
            if tag == "사랑" and not romantic_context:
                continue
            if tag == "설렘" and tag not in analysis_terms and not romantic_context:
                continue
            if tag in hidden_unless_matched and tag not in analysis_terms:
                continue
            if tag not in tags:
                tags.append(tag)
            if len(tags) >= 4:
                break

        if len(tags) < 3:
            for tag in [item.get("recommendation_role"), *analysis.get("topics", [])[:2], "저부담"]:
                tag = str(tag or "").strip()
                if tag and tag not in tags:
                    tags.append(tag)
                if len(tags) >= 4:
                    break
        return tags[:4]

    @staticmethod
    def _algorithm_profile(
        slots: list[RecommendationSlot],
        blocked_count: int,
        cache_stats: dict[str, int],
        retrieval_stats: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "pipeline": [
                "strategy_builder",
                "candidate_retriever",
                "chroma_candidate_store" if retrieval_stats.get("backend") == "chroma" else "memory_candidate_store",
                "content_affect_profiler",
                "content_embedding_index",
                "safety_policy",
                "feature_extractor",
                "slot_scorer",
                "portfolio_builder",
                "explanation_builder",
            ],
            "slot_weights": {slot.role: slot.weights for slot in slots},
            "blocked_by_safety": blocked_count,
            "candidate_retrieval": retrieval_stats,
            "content_embedding_cache": cache_stats,
            "principles": [
                "추천은 top-k 점수순이 아니라 역할 기반 포트폴리오로 구성",
                "감정 일치보다 현재 상태에서 적절한 감정 조절 효과를 더 중시",
                "콘텐츠별 긴장도/회복 가능성/인지 부담을 분리해 안전성과 적합도를 계산",
                "콘텐츠 임베딩은 content_id와 embedding_text 해시 기준으로 재사용",
                "위험 신호와 감정별 avoid 조건은 점수 계산 전에 안전 정책으로 처리",
                "슬롯별 가중치를 다르게 적용해 공감/안정/회복/재도전을 구분",
                "데이터셋에 있는 콘텐츠만 추천",
            ],
        }


def evidence_text(details: list[dict[str, Any]]) -> str:
    evidence: list[str] = []
    for detail in details[:3]:
        evidence.extend(detail.get("evidence", [])[:1])
    return " / ".join(evidence)
