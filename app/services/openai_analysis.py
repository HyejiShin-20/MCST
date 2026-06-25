from __future__ import annotations

import json
import re
from typing import Any

from app import config
from app.services.text_utils import normalize_text


class OpenAIAnalysisEnhancer:
    def enhance(self, text: str, local_analysis: dict[str, Any]) -> dict[str, Any] | None:
        if not config.OPENAI_ANALYSIS_ENABLED or not config.OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(
                api_key=config.OPENAI_API_KEY or None,
                timeout=config.OPENAI_ANALYSIS_TIMEOUT_SECONDS,
            )
            response = client.responses.create(
                model=config.OPENAI_EMOTION_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You analyze Korean diary records for a culture "
                                    "recommendation backend. Return compact JSON only."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": self._prompt(text, local_analysis),
                            }
                        ],
                    },
                ],
            )
            parsed = _parse_json(_response_text(response))
            if not parsed:
                return None
            return {
                "provider": "openai",
                "model": config.OPENAI_EMOTION_MODEL,
                "result": parsed,
            }
        except Exception as primary_error:
            if not config.OPENAI_EMOTION_FALLBACK_MODEL:
                return {
                    "provider": "openai",
                    "model": config.OPENAI_EMOTION_MODEL,
                    "error": str(primary_error)[:300],
                }
            try:
                from openai import OpenAI  # type: ignore

                client = OpenAI(
                    api_key=config.OPENAI_API_KEY or None,
                    timeout=config.OPENAI_ANALYSIS_TIMEOUT_SECONDS,
                )
                response = client.responses.create(
                    model=config.OPENAI_EMOTION_FALLBACK_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": self._prompt(text, local_analysis),
                                }
                            ],
                        }
                    ],
                )
                parsed = _parse_json(_response_text(response))
                if not parsed:
                    return None
                return {
                    "provider": "openai",
                    "model": config.OPENAI_EMOTION_FALLBACK_MODEL,
                    "result": parsed,
                    "fallback_from": config.OPENAI_EMOTION_MODEL,
                }
            except Exception as fallback_error:
                return {
                    "provider": "openai",
                    "model": config.OPENAI_EMOTION_MODEL,
                    "error": str(fallback_error)[:300],
                    "fallback_from_error": str(primary_error)[:300],
                }

    @staticmethod
    def _prompt(text: str, local_analysis: dict[str, Any]) -> str:
        local_brief = {
            "emotions": local_analysis.get("emotions", []),
            "topics": local_analysis.get("topics", []),
            "situation": local_analysis.get("situation", {}),
            "desired_support": local_analysis.get("desired_support", []),
        }
        return (
            "Analyze the user's combined multimodal diary text. "
            "Focus on nuanced, mixed emotions, concrete situation, interests, "
            "and recommendation intent. Use Korean labels where useful.\n\n"
            "Return JSON with keys: emotions(list), emotion_scores(object), "
            "topics(list), situation(object with primary_event,intent,needs,"
            "risk_level,valence,arousal,intensity,confidence), desired_support(list), "
            "recommendation_strategy(object with mode,emotional_arc,prioritize,avoid), "
            "nuanced_emotion(string), uncertainty(list), summary(string).\n\n"
            f"USER_TEXT:\n{normalize_text(text)}\n\n"
            f"LOCAL_ANALYSIS:\n{json.dumps(local_brief, ensure_ascii=False)}"
        )


def merge_openai_analysis(
    local_analysis: dict[str, Any],
    openai_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    if not openai_signal:
        return local_analysis
    merged = dict(local_analysis)
    signals = list(merged.get("model_signals", []))
    signals.append({"type": "openai_analysis", **openai_signal})
    merged["model_signals"] = signals
    merged["llm_augmented"] = bool(openai_signal.get("result"))
    result = openai_signal.get("result") or {}
    if not isinstance(result, dict):
        return merged

    merged["openai_analysis"] = result
    # LLM 분석을 감정의 권위 소스로 둔다(union이 아니라 LLM 우선).
    # 로컬 감정은 이미 모달리티 충돌 해소를 거친 상태라, LLM 결과 뒤에 보조로만 덧붙인다.
    # LLM 감정이 없을 때만 로컬이 그대로 사용된다.
    merged["emotions"] = _merge_lists(
        result.get("emotions", []), merged.get("emotions", []), 5
    )
    merged["topics"] = _merge_lists(merged.get("topics", []), result.get("topics", []), 8)
    merged["desired_support"] = _merge_lists(
        merged.get("desired_support", []),
        result.get("desired_support", []),
        8,
    )
    if result.get("nuanced_emotion"):
        merged["nuanced_emotion"] = str(result["nuanced_emotion"])
    if result.get("summary"):
        merged["summary"] = str(result["summary"])
    if result.get("uncertainty"):
        merged["uncertainty"] = _merge_lists(
            merged.get("uncertainty", []),
            result.get("uncertainty", []),
            8,
        )
    if isinstance(result.get("situation"), dict):
        merged["situation"] = _merge_dict(
            merged.get("situation", {}),
            result.get("situation", {}),
            allowed_keys={
                "primary_event",
                "intent",
                "needs",
                "risk_level",
                "valence",
                "arousal",
                "intensity",
                "confidence",
            },
        )
    if isinstance(result.get("recommendation_strategy"), dict):
        merged["recommendation_strategy"] = _merge_dict(
            merged.get("recommendation_strategy", {}),
            result.get("recommendation_strategy", {}),
            allowed_keys={"mode", "emotional_arc", "prioritize", "avoid"},
        )
    return merged


def _merge_lists(primary: Any, secondary: Any, limit: int) -> list[str]:
    items: list[str] = []
    for value in list(primary or []) + list(secondary or []):
        text = normalize_text(str(value))
        if text and text not in items:
            items.append(text)
    return items[:limit]


def _merge_dict(primary: Any, secondary: Any, allowed_keys: set[str]) -> dict[str, Any]:
    merged = dict(primary or {}) if isinstance(primary, dict) else {}
    if not isinstance(secondary, dict):
        return merged
    for key in allowed_keys:
        value = secondary.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            merged[key] = _merge_lists(merged.get(key, []), value, 8)
        else:
            merged[key] = value
    return merged


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    try:
        data = response.model_dump()
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(response)


def _parse_json(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
