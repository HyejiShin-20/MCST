from __future__ import annotations

from app.schemas.recommendation import RecommendationSlot


class PortfolioBuilder:
    def build(
        self,
        scored_items: list[dict],
        slots: list[RecommendationSlot],
        per_type: int,
    ) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {"movie": [], "music": [], "book": []}
        for content_type in grouped:
            candidates = [
                item for item in scored_items if item["content_type"] == content_type
            ]
            grouped[content_type] = self._select_for_type(candidates, slots, per_type)
        return grouped

    def _select_for_type(
        self,
        candidates: list[dict],
        slots: list[RecommendationSlot],
        per_type: int,
    ) -> list[dict]:
        selected: list[dict] = []
        remaining = candidates[:]
        for slot in slots:
            if not remaining or len(selected) >= per_type:
                break
            ranked = []
            for item in remaining:
                slot_score = item["slot_scores"].get(slot.role, 0.0)
                adjustment = diversity_adjustment(item, selected)
                ranked.append((slot_score + adjustment, adjustment, item))
            ranked.sort(key=lambda value: value[0], reverse=True)
            final_score, adjustment, selected_item = ranked[0]
            minimum_items = min(3, per_type)
            if len(selected) >= minimum_items and final_score < quality_floor(slot.role):
                break
            remaining.remove(selected_item)
            item = dict(selected_item)
            item["recommendation_role"] = slot.role
            item["expected_effect"] = slot.purpose
            item["target_emotional_shift"] = selected_item["target_shifts"].get(slot.role)
            item["rank_adjustment"] = round(adjustment, 4)
            item["score"] = round(max(0.0, min(1.0, final_score)), 4)
            item["raw_score_percent"] = round(item["score"] * 100)
            item["score_percent"] = display_score_percent(item["score"])
            item["score_components"] = dict(item["slot_features"][slot.role])
            item["score_components"]["diversity_adjustment"] = round(adjustment, 4)
            selected.append(item)
        return selected


def diversity_adjustment(item: dict, selected: list[dict]) -> float:
    if not selected:
        return 0.0
    adjustment = 0.0
    item_topics = set(item.get("topic_tags", []))
    for existing in selected:
        if item.get("genre") == existing.get("genre"):
            adjustment -= 0.03
        existing_topics = set(existing.get("topic_tags", []))
        denominator = max(1, min(len(item_topics), len(existing_topics)))
        overlap = len(item_topics & existing_topics) / denominator
        adjustment -= min(0.055, overlap * 0.04)
        if item.get("recommendation_role") == existing.get("recommendation_role"):
            adjustment -= 0.02
    return max(-0.14, adjustment)


def display_score_percent(score: float) -> int:
    return round(max(0.0, min(96.0, score * 100.0)))


def quality_floor(role: str) -> float:
    if role in {"발견", "가벼운 성찰", "배경", "유지"}:
        return 0.34
    if role in {"학습", "탐색", "몰입", "확장"}:
        return 0.35
    return 0.36
