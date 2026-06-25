from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.schemas.analysis import EvidenceDetail
from app.services.analysis_rules import (
    DISTRESS_EMOTIONS,
    EMOTION_KEYWORDS,
    EMOTION_RANK_BIAS,
    INTENSIFIERS,
    NEUTRAL_EMOTIONS,
    POSITIVE_EMOTIONS,
    RISK_KEYWORDS,
    SENTENCE_RE,
    SOFTENERS,
)
from app.services.text_utils import normalize_text, tokenize


# 2차(이미지 캡션/분위기 등 AI 추론) 텍스트가 감정 카운트에 기여하는 비중.
# 사용자가 직접 쓴 1차 텍스트보다 항상 낮게 둔다.
SECONDARY_EMOTION_WEIGHT = 0.5
# 부정 정서로 취급할 라벨(중립/긍정 충돌 판단용). DISTRESS + 피로.
NEGATIVE_EMOTIONS = DISTRESS_EMOTIONS | {"피로"}


class EmotionDetector:
    def detect(self, text: str, secondary_text: str = "") -> dict[str, Any]:
        """1차(사용자 작성) 텍스트와 2차(이미지/첨부 추론) 텍스트를 분리해 감정을 검출한다.

        - 2차 텍스트는 SECONDARY_EMOTION_WEIGHT 비중으로만 반영.
        - 모달리티 충돌(예: 사용자는 분노인데 이미지 분위기는 평온)은 fuse_modalities에서 해소.
        secondary_text가 비어 있으면 기존 단일 텍스트 동작과 동일하다(하위호환).
        """
        primary = normalize_text(text)
        secondary = normalize_text(secondary_text)
        joined = " ".join(tokenize(primary))
        primary_counts = self.weighted_counts(primary, joined, EMOTION_KEYWORDS)
        secondary_counts: Counter[str] = Counter()
        if secondary:
            secondary_joined = " ".join(tokenize(secondary))
            secondary_counts = self.weighted_counts(
                secondary, secondary_joined, EMOTION_KEYWORDS
            )

        counts, modality_conflict = self.fuse_modalities(
            primary_counts, secondary_counts, SECONDARY_EMOTION_WEIGHT
        )

        combined = primary if not secondary else f"{primary}\n{secondary}"
        sentences = self.sentences(combined)
        emotions = self.rank_emotions(counts)[:5]
        if not emotions:
            emotions = ["평온"]
            counts["평온"] = 1.0
        details = self.details(emotions, counts, EMOTION_KEYWORDS, sentences)
        return {
            "emotions": emotions,
            "emotion_counts": counts,
            "emotion_scores": self.emotion_scores(counts),
            "emotion_details": details,
            "risk_flags": [keyword for keyword in RISK_KEYWORDS if keyword in combined],
            "sentences": sentences,
            "modality_conflict": modality_conflict,
        }

    @staticmethod
    def fuse_modalities(
        primary: Counter[str],
        secondary: Counter[str],
        secondary_weight: float,
    ) -> tuple[Counter[str], bool]:
        """1차/2차 감정 카운트를 가중 합산하고 모달리티 충돌을 해소한다.

        Rule 1: 2차에만 존재하고 1차 정서 방향과 반대되는 감정은 제거한다.
                (예: 사용자 텍스트는 부정인데 이미지에서만 온 긍정 감정)
        Rule 2: 최상위 감정이 고통(distress) 계열이면 중립 감정(평온/평범함/무료함)을 제거한다.
                (강한 부정 상황에서 '그냥/calm' 같은 약한 중립 신호는 노이즈)
        반환: (융합 카운트, 충돌 발생 여부)
        """
        fused: Counter[str] = Counter()
        for label, value in primary.items():
            fused[label] += float(value)
        for label, value in secondary.items():
            fused[label] += float(value) * secondary_weight

        conflict = False
        if not fused:
            return fused, conflict

        primary_positive = sum(
            float(primary[label]) for label in primary if label in POSITIVE_EMOTIONS
        )
        primary_negative = sum(
            float(primary[label]) for label in primary if label in NEGATIVE_EMOTIONS
        )
        if primary_positive > primary_negative:
            primary_valence = "positive"
        elif primary_negative > primary_positive:
            primary_valence = "negative"
        else:
            primary_valence = "neutral"

        # Rule 1: 2차 전용 + 1차와 반대 방향 감정 제거
        if primary_valence in {"positive", "negative"}:
            for label in list(fused):
                if float(primary.get(label, 0.0)) > 0:
                    continue  # 1차에 근거가 있으면 유지(복합감정 보존)
                if primary_valence == "positive" and label in NEGATIVE_EMOTIONS:
                    del fused[label]
                    conflict = True
                elif primary_valence == "negative" and label in POSITIVE_EMOTIONS:
                    del fused[label]
                    conflict = True

        # Rule 2: 최상위 감정이 고통 계열이면 중립 감정 제거
        ranked = EmotionDetector.rank_emotions(fused)
        if ranked and ranked[0] in DISTRESS_EMOTIONS:
            for label in list(fused):
                if label in NEUTRAL_EMOTIONS:
                    del fused[label]
                    conflict = True

        return fused, conflict

    @staticmethod
    def sentences(text: str) -> list[str]:
        sentences = [normalize_text(match.group(0)) for match in SENTENCE_RE.finditer(text)]
        return [sentence for sentence in sentences if sentence] or [text]

    @staticmethod
    def weighted_counts(
        text: str, joined_tokens: str, keyword_map: dict[str, list[str]]
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        lowered = text.lower()
        raw_tokens = set(tokenize_raw(lowered))
        intensity_bonus = 1.0 + min(0.8, 0.16 * sum(1 for cue in INTENSIFIERS if cue in text))
        softener = 1.0 - min(0.35, 0.12 * sum(1 for cue in SOFTENERS if cue in text))
        weight = max(0.45, intensity_bonus * softener)
        for label, keywords in keyword_map.items():
            hits = sum(
                1
                for keyword in keywords
                if keyword_hit(lowered, raw_tokens, joined_tokens, keyword)
            )
            if hits:
                counts[label] = round(hits * weight, 3)
        return counts

    @staticmethod
    def rank_emotions(counter: Counter[str]) -> list[str]:
        ranked = [
            label
            for label, _ in sorted(
                (
                    (label, float(value) * EMOTION_RANK_BIAS.get(label, 1.0))
                    for label, value in counter.items()
                    if value > 0
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        distress = [label for label in ranked if label in DISTRESS_EMOTIONS]
        if not distress:
            return ranked
        distress_score = sum(
            float(counter[label]) * EMOTION_RANK_BIAS.get(label, 1.0)
            for label in distress
        )
        positive_score = sum(
            float(counter[label]) * EMOTION_RANK_BIAS.get(label, 1.0)
            for label in ranked
            if label in POSITIVE_EMOTIONS
        )
        if positive_score >= distress_score * 0.9:
            return ranked
        supportive = [label for label in ranked if label not in DISTRESS_EMOTIONS]
        return distress + supportive

    @staticmethod
    def details(
        labels: list[str],
        counts: Counter[str],
        keyword_map: dict[str, list[str]],
        sentences: list[str],
    ) -> list[EvidenceDetail]:
        max_count = max(counts.values()) if counts else 1.0
        details: list[EvidenceDetail] = []
        for label in labels:
            evidence = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_tokens = set(tokenize_raw(sentence_lower))
                sentence_joined = " ".join(tokenize(sentence_lower))
                if any(
                    keyword_hit(sentence_lower, sentence_tokens, sentence_joined, keyword)
                    for keyword in keyword_map.get(label, [])
                ):
                    evidence.append(sentence)
            raw_score = float(counts[label])
            normalized = min(1.0, raw_score / max_count)
            details.append(
                EvidenceDetail(
                    label=label,
                    score=round(normalized, 3),
                    intensity=intensity_label(normalized),
                    evidence=evidence[:2],
                )
            )
        return details

    @staticmethod
    def emotion_scores(counter: Counter[str]) -> dict[str, float]:
        total = max(1.0, float(sum(counter.values())))

        def score(keys: list[str]) -> float:
            value = sum(float(counter[key]) for key in keys)
            return round(min(1.0, value / total), 3)

        return {
            "sadness": score(["슬픔", "외로움"]),
            "anxiety": score(["불안"]),
            "anger": score(["분노"]),
            "fatigue": score(["피로", "무기력"]),
            "shame": score(["창피함"]),
            "regret": score(["후회"]),
            "hope": score(["희망", "기쁨", "설렘", "자신감"]),
            "joy": score(["기쁨"]),
            "pride": score(["뿌듯함", "자신감"]),
            "gratitude": score(["감사"]),
            "relief": score(["안도감"]),
            "curiosity": score(["호기심"]),
            "calm": score(["평온"]),
            "ordinary": score(["평범함"]),
            "boredom": score(["무료함"]),
            "neutral": score(list(NEUTRAL_EMOTIONS)),
        }


def intensity_label(value: float) -> str:
    if value >= 0.75:
        return "높음"
    if value >= 0.45:
        return "중간"
    return "낮음"


def tokenize_raw(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text)


def keyword_hit(
    text: str, raw_tokens: set[str], joined_tokens: str, keyword: str
) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    if " " in keyword:
        return keyword in text
    if len(keyword) == 1:
        return any(
            token == keyword or (token.startswith(keyword) and len(token) <= 3)
            for token in raw_tokens
        )
    return any(token == keyword or token.startswith(keyword) for token in raw_tokens)
