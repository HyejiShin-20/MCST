from __future__ import annotations

from typing import Any

from app import config
from app.schemas.analysis import AnalysisResult
from app.services.context_detector import ContextDetector
from app.services.emotion_detector import EmotionDetector
from app.services.openai_analysis import OpenAIAnalysisEnhancer, merge_openai_analysis
from app.services.analysis_rules import DISTRESS_EMOTIONS, EMOTIONAL_ARCS, POSITIVE_EMOTIONS
from app.services.support_strategy import SupportStrategyPlanner
from app.services.text_utils import normalize_text


class EmotionAnalyzer:
    def __init__(
        self,
        emotion_detector: EmotionDetector | None = None,
        context_detector: ContextDetector | None = None,
        strategy_planner: SupportStrategyPlanner | None = None,
        openai_enhancer: OpenAIAnalysisEnhancer | None = None,
    ) -> None:
        self.backend = "keyword-fallback"
        self.pipeline = None
        self.emotion_detector = emotion_detector or EmotionDetector()
        self.context_detector = context_detector or ContextDetector()
        self.strategy_planner = strategy_planner or SupportStrategyPlanner()
        self.openai_enhancer = openai_enhancer or OpenAIAnalysisEnhancer()
        self._try_load_hf_pipeline()

    def _try_load_hf_pipeline(self) -> None:
        if not config.HF_EMOTION_MODEL:
            return
        try:
            from transformers import pipeline  # type: ignore

            self.pipeline = pipeline(
                "text-classification",
                model=config.HF_EMOTION_MODEL,
                tokenizer=config.HF_EMOTION_MODEL,
                top_k=None,
                model_kwargs={"local_files_only": config.HF_LOCAL_FILES_ONLY},
            )
            self.backend = f"huggingface:{config.HF_EMOTION_MODEL}"
        except Exception:
            self.pipeline = None

    def analyze(self, text: str, secondary_text: str = "") -> dict[str, Any]:
        """일기 분석.

        text: 사용자가 직접 작성한 1차 텍스트(+ OCR/STT 등 사용자 본인 발화).
        secondary_text: 이미지 캡션/장면/분위기 등 AI가 추론한 2차 텍스트(낮은 비중).
        secondary_text가 비면 단일 텍스트 분석과 동일하게 동작한다.
        """
        normalized = normalize_text(text)
        secondary = normalize_text(secondary_text)
        # 토픽/상황 단서는 이미지 맥락도 활용하되, 감정은 모달리티 분리로 검출한다.
        combined = normalized if not secondary else f"{normalized}\n{secondary}"
        emotion = self.emotion_detector.detect(normalized, secondary)
        desired_support = self.strategy_planner.desired_support(
            emotion["emotions"], [], emotion["risk_flags"]
        )
        context = self.context_detector.detect(
            text=combined,
            emotions=emotion["emotions"],
            emotion_scores=emotion["emotion_scores"],
            risk_flags=emotion["risk_flags"],
            sentences=emotion["sentences"],
            desired_support=desired_support,
        )
        desired_support = self.strategy_planner.desired_support(
            emotion["emotions"], context["topics"], emotion["risk_flags"]
        )
        context = self.context_detector.detect(
            text=combined,
            emotions=emotion["emotions"],
            emotion_scores=emotion["emotion_scores"],
            risk_flags=emotion["risk_flags"],
            sentences=emotion["sentences"],
            desired_support=desired_support,
        )
        goal = self.strategy_planner.emotional_goal(
            emotion["emotions"], desired_support
        )
        strategy = self.strategy_planner.strategy(
            context["situation"],
            desired_support,
            emotion["emotions"],
            context["topics"],
        )
        nuanced = self.strategy_planner.nuanced_emotion(
            emotion["emotions"], context["topics"], context["situation"], goal
        )
        hf_scores = self._hf_scores(normalized)
        confidence = min(context["situation"].confidence, strategy.confidence)
        uncertainty = list(
            dict.fromkeys(
                context["situation"].uncertainty_reasons
                + strategy.alternative_interpretations
            )
        )
        summary = (
            f"현재 기록은 '{context['situation'].primary_event}' 상황으로 판단됩니다. "
            f"주요 감정은 {', '.join(emotion['emotions'][:3])}이고 "
            f"목표 정서는 {', '.join(goal.desired_state[:3]) or '부담 없는 탐색'}입니다. "
            f"추천은 {' -> '.join(strategy.emotional_arc)} 흐름을 우선합니다."
        )

        result = AnalysisResult(
            emotions=emotion["emotions"],
            emotion_scores=emotion["emotion_scores"],
            emotion_details=emotion["emotion_details"],
            topics=context["topics"],
            topic_details=context["topic_details"],
            context_keywords=context["context_keywords"],
            desired_support=desired_support,
            situation=context["situation"],
            recommendation_strategy=strategy,
            emotional_goal=goal,
            nuanced_emotion=nuanced,
            confidence=round(confidence, 3),
            uncertainty=uncertainty,
            summary=summary,
            risk_flags=emotion["risk_flags"],
            backend=self.backend,
            model_signals=hf_scores,
        )
        result_data = result.model_dump()
        # 하이브리드 게이팅: 로컬이 불확실하거나 모달리티 충돌이 있을 때만 LLM 호출(비용 절감).
        if self._should_use_llm(result_data, bool(emotion.get("modality_conflict"))):
            llm_input = normalized
            if secondary:
                llm_input = (
                    f"[사용자 작성]\n{normalized}\n\n[이미지/첨부 추론(참고)]\n{secondary}"
                )
            openai_signal = self.openai_enhancer.enhance(llm_input, result_data)
        else:
            openai_signal = None
        merged = merge_openai_analysis(result_data, openai_signal)
        merged = reconcile_analysis_consistency(merged)
        merged["structured_emotion"] = structured_emotion_view(normalized, merged)
        return merged

    @staticmethod
    def _should_use_llm(analysis: dict[str, Any], modality_conflict: bool) -> bool:
        """LLM 보강 호출 여부.

        OPENAI_EMOTION_USE_WHEN:
          - "never"  : 항상 미사용
          - "always" : 항상 사용(기본값, 기존 동작 유지)
          - "auto"   : 로컬 신뢰도가 낮거나/감정 후보가 많거나/혼합·충돌일 때만 사용
        """
        if not (config.OPENAI_ANALYSIS_ENABLED and config.OPENAI_API_KEY):
            return False
        mode = (config.OPENAI_EMOTION_USE_WHEN or "always").strip().lower()
        if mode == "never":
            return False
        if mode != "auto":
            return True
        if modality_conflict:
            return True
        if float(analysis.get("confidence", 1.0) or 0.0) < config.LOW_CONFIDENCE_THRESHOLD:
            return True
        if len(analysis.get("emotions", []) or []) >= 4:
            return True
        situation = analysis.get("situation", {}) or {}
        if str(situation.get("valence", "")) == "mixed":
            return True
        return False

    def _hf_scores(self, text: str) -> list[dict[str, Any]]:
        if self.pipeline is None:
            return []
        try:
            result = self.pipeline(text)
            if result and isinstance(result[0], list):
                result = result[0]
            return [
                {"label": item.get("label"), "score": round(float(item.get("score", 0)), 4)}
                for item in result[:8]
            ]
        except Exception:
            return []


def structured_emotion_view(text: str, analysis: dict[str, Any]) -> dict[str, Any]:
    emotions = [str(item) for item in analysis.get("emotions", []) if item]
    situation = analysis.get("situation", {})
    emotion_scores = analysis.get("emotion_scores", {})
    context_keywords = list(dict.fromkeys(analysis.get("context_keywords", [])))[:8]

    dominant = emotions[0] if emotions else "평온"
    positive_candidates = [
        emotion
        for emotion in emotions
        if emotion in {"기쁨", "즐거움", "뿌듯함", "감사", "안도감", "희망", "자신감", "설렘"}
    ]
    situation_valence = str(situation.get("valence", "neutral"))
    if dominant in {"평온", "평범함"} and positive_candidates and situation_valence in {"positive", "mixed"}:
        dominant = positive_candidates[0]
    if (
        any(item in emotions for item in ["뿌듯함", "성취감", "자신감"])
        and any(item in emotions for item in ["피로", "무기력"])
        and "뿌듯" in text
    ):
        dominant = "뿌듯함"

    sub_emotions = [item for item in emotions if item != dominant]
    if "피곤" in text and "피로" not in sub_emotions and dominant != "피로":
        sub_emotions.insert(0, "피로")
    if any(cue in text for cue in ["강아지", "반려견", "토리", "산책"]) and "따뜻함" not in sub_emotions:
        sub_emotions.append("따뜻함")
    if "운동" in text and "자기관리" not in context_keywords:
        context_keywords.append("자기관리")

    arousal = str(situation.get("arousal", "low"))
    fatigue_score = float(emotion_scores.get("fatigue", 0.0) or 0.0)
    pride_score = float(emotion_scores.get("pride", 0.0) or 0.0)
    if fatigue_score > 0 and pride_score > 0:
        energy_level = "medium_low"
    elif arousal == "high":
        energy_level = "high"
    elif fatigue_score > 0.25 or dominant in {"무기력", "피로"}:
        energy_level = "low"
    else:
        energy_level = "medium"

    valence = str(situation.get("valence", "neutral"))
    if dominant in {"뿌듯함", "기쁨", "자신감"} and fatigue_score > 0:
        valence = "positive_mixed"

    return {
        "dominant_emotion": dominant,
        "sub_emotions": sub_emotions[:5],
        "context_keywords": context_keywords[:8],
        "energy_level": energy_level,
        "valence": valence,
        "arousal": arousal,
    }


def reconcile_analysis_consistency(analysis: dict[str, Any]) -> dict[str, Any]:
    reconciled = dict(analysis)
    emotions = set(reconciled.get("emotions", []))
    situation = dict(reconciled.get("situation", {}))
    desired_support = list(reconciled.get("desired_support", []))
    has_positive = bool(emotions & POSITIVE_EMOTIONS)
    has_distress = bool(emotions & DISTRESS_EMOTIONS)
    valence = str(situation.get("valence", "neutral"))

    if has_positive and not has_distress and situation.get("intent") == "neutral_daily_based":
        situation["intent"] = "positive_emotion_based"
        if valence == "neutral":
            situation["valence"] = "positive"
    reconciled["situation"] = situation

    strategy = dict(reconciled.get("recommendation_strategy", {}))
    if situation.get("intent") == "positive_emotion_based":
        strategy["mode"] = "positive_emotion_based"
        if "축하" in desired_support:
            arc = EMOTIONAL_ARCS.get("축하", [])
        elif "음미" in desired_support:
            arc = EMOTIONAL_ARCS.get("음미", [])
        else:
            arc = EMOTIONAL_ARCS.get("증폭", [])
        if arc:
            strategy["emotional_arc"] = arc
    reconciled["recommendation_strategy"] = strategy
    return reconciled
