"""LLM 기반 콘텐츠 태거 (gpt-5.4-mini 등).

기존 키워드/프로토타입 태깅이 만든 노이즈를 고치기 위한 고품질 재태깅용.
설계 원칙:
- LLM 은 '정해진 어휘(tag_config)'에서 가장 핵심적인 태그만 간결하게 고른다.
  → 기존 태그 톤을 유지하고 엉뚱한 태그가 새지 않게 한다.
- 선택된 태그를 기존 finalize_tags() 로 그대로 후처리한다.
  → 출력 스키마(valence/arousal/confidence/flags 등)와 DB 저장 로직을 재사용.
- tagging_text 는 기존 build_tagging_text() 와 동일 → 임베딩 해시 불변(임베딩 재생성 불필요).
"""
from __future__ import annotations

import json
import re
from typing import Any

from tagger.embedding import embedding_model_name, get_embedding_hash
from tagger.preprocess import build_tagging_text, text_quality_score
from tagger.prototype_tagger import finalize_tags
from tagger.tag_config import (
    EMOTION_TAGS_NEGATIVE,
    EMOTION_TAGS_NEUTRAL,
    EMOTION_TAGS_POSITIVE,
    MOOD_TAGS,
    RECOMMENDATION_ROLES,
    THEME_TAGS,
)

EMOTION_VOCAB = list(dict.fromkeys(EMOTION_TAGS_NEGATIVE + EMOTION_TAGS_POSITIVE + EMOTION_TAGS_NEUTRAL))
THEME_VOCAB = list(dict.fromkeys(THEME_TAGS))
MOOD_VOCAB = list(dict.fromkeys(MOOD_TAGS))
ROLE_VOCAB = list(dict.fromkeys(RECOMMENDATION_ROLES))

# 핵심 위주로 간결하게 (버킷별 최대 개수)
MAX_EMOTION = 3
MAX_THEME = 3
MAX_MOOD = 2
MAX_ROLE = 2

TAG_METHOD = "llm_vocab_v1"

SYSTEM_PROMPT = (
    "당신은 한국어 문화 콘텐츠(영화/음악/도서)의 감정·주제 태거입니다.\n"
    "콘텐츠 설명을 읽고, 사용자가 그 감정/상황일 때 추천하기 좋은지를 기준으로 태그를 답니다.\n"
    "규칙:\n"
    "1) 반드시 아래 '허용 어휘'에 있는 단어만 그대로 사용합니다. 어휘에 없는 단어는 절대 만들지 않습니다.\n"
    "2) 가장 핵심적인 것만 간결하게 고릅니다. 애매하면 빼세요. (감정 1~3개, 주제 1~3개, 무드 1~2개, 추천역할 1~2개)\n"
    "3) 장르/연도 같은 분류 라벨이 아니라, 정서·분위기·쓰임새 중심으로 고릅니다.\n"
    "4) 출력은 오직 JSON 하나입니다. 설명/문장/코드블록 없이 JSON 객체만 출력합니다.\n\n"
    "허용 어휘:\n"
    f"- emotion_tags: {', '.join(EMOTION_VOCAB)}\n"
    f"- theme_tags: {', '.join(THEME_VOCAB)}\n"
    f"- mood_tags: {', '.join(MOOD_VOCAB)}\n"
    f"- recommendation_roles: {', '.join(ROLE_VOCAB)}\n\n"
    "출력 형식(JSON):\n"
    '{"emotion_tags": [], "theme_tags": [], "mood_tags": [], "recommendation_roles": []}'
)


def build_user_prompt(content: dict[str, Any]) -> str:
    _, tagging_text = build_tagging_text(content)
    return (
        "다음 콘텐츠에 어울리는 태그를 허용 어휘에서만 핵심 위주로 골라 JSON으로만 답하세요.\n\n"
        f"{tagging_text}"
    )


def build_request_messages(content: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI Responses API 의 input 메시지 배열을 만든다."""
    return [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "input_text", "text": build_user_prompt(content)}]},
    ]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_output_text(response_body: dict[str, Any]) -> str:
    """Responses API 응답 body 에서 모델 출력 텍스트를 안전하게 뽑는다."""
    text = response_body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    chunks: list[str] = []
    for item in response_body.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                value = part.get("text")
                if isinstance(value, str):
                    chunks.append(value)
    return "\n".join(chunks)


def parse_llm_tags(text: str) -> dict[str, list[str]]:
    """모델 출력 텍스트 → 허용 어휘로 필터링된 태그 딕셔너리."""
    match = _JSON_RE.search(text or "")
    if not match:
        raise ValueError("LLM_OUTPUT_NOT_JSON")
    data = json.loads(match.group(0))

    def clean(values: Any, vocab: list[str], limit: int) -> list[str]:
        allowed = set(vocab)
        result: list[str] = []
        if isinstance(values, list):
            for value in values:
                tag = str(value).strip()
                if tag in allowed and tag not in result:
                    result.append(tag)
        return result[:limit]

    return {
        "emotion_tags": clean(data.get("emotion_tags"), EMOTION_VOCAB, MAX_EMOTION),
        "theme_tags": clean(data.get("theme_tags"), THEME_VOCAB, MAX_THEME),
        "mood_tags": clean(data.get("mood_tags"), MOOD_VOCAB, MAX_MOOD),
        "recommendation_roles": clean(data.get("recommendation_roles"), ROLE_VOCAB, MAX_ROLE),
    }


def _scored(tags: list[str]) -> dict[str, float]:
    """선택 순서대로 내림차순 점수를 부여해 기존 finalize_tags 와 호환되게 만든다."""
    scores: dict[str, float] = {}
    for index, tag in enumerate(tags):
        scores[tag] = round(max(0.5, 0.92 - 0.08 * index), 4)
    return scores


def build_tag_result(
    content: dict[str, Any], llm_tags: dict[str, list[str]]
) -> tuple[dict[str, Any], str, str]:
    """LLM 태그 → 기존 스키마의 tag_result(+embedding_hash, model_name)."""
    processed_text, tagging_text = build_tagging_text(content)
    quality = text_quality_score(processed_text)
    if quality < 0.15:
        raise ValueError("TEXT_TOO_SHORT")

    scores = {
        "emotion_tags": _scored(llm_tags["emotion_tags"]),
        "theme_tags": _scored(llm_tags["theme_tags"]),
        "mood_tags": _scored(llm_tags["mood_tags"]),
        "recommendation_roles": _scored(llm_tags["recommendation_roles"]),
    }
    # keyword 버킷에도 동일 점수를 넣어 신뢰도 계산이 LLM 선택을 신뢰하도록 한다.
    merged_result = {
        "scores": {bucket: dict(values) for bucket, values in scores.items()},
        "keyword": {bucket: dict(values) for bucket, values in scores.items()},
        "prototype": {
            "emotion_tags": {},
            "theme_tags": {},
            "mood_tags": {},
            "recommendation_roles": {},
        },
    }
    final = finalize_tags(
        merged_result,
        content_type=str(content.get("content_type", "")),
        text_quality=quality,
    )

    raw = {
        "llm_tags": llm_tags,
        "tag_method": TAG_METHOD,
        "tagging_text_preview": tagging_text[:240],
    }
    tag_result = {
        "tag_method": TAG_METHOD,
        "processed_text": processed_text,
        "tagging_text": tagging_text,
        "raw": raw,
        "final": final,
    }
    model_name = embedding_model_name()
    embedding_hash = get_embedding_hash(tagging_text, model_name)
    return tag_result, embedding_hash, model_name
