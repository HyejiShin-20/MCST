from __future__ import annotations

from app.services.recommendation.rules import TYPE_LABELS


class ExplanationBuilder:
    def build(self, analysis: dict, item: dict) -> str:
        role = item.get("recommendation_role", "추천")
        label = TYPE_LABELS.get(item.get("content_type"), item.get("content_type", "콘텐츠"))
        title = item.get("title", "이 콘텐츠")
        structured = analysis.get("structured_emotion", {})
        situation = analysis.get("situation", {})
        affect_profile = item.get("content_affect_profile", {})
        context = structured.get("context_keywords", analysis.get("context_keywords", []))
        input_note = input_sentence(
            dominant=str(structured.get("dominant_emotion", "")),
            sub_emotions=structured.get("sub_emotions", []),
            context=context,
            situation=situation,
        )
        tag_note = content_tag_sentence(item)
        component_note = strongest_components(item.get("score_components", {}))
        role_text = role_sentence(
            role=role,
            label=label,
            title=title,
            tag_note=tag_note,
            component_note=component_note,
            affect_profile=affect_profile,
        )
        safety_note = ""
        safety_reasons = item.get("safety_reasons", [])
        if safety_reasons:
            safety_note = " 감정이 과하게 가라앉지 않도록 안전 조건도 함께 반영했습니다."
        return f"{input_note} {role_text}{safety_note}"


def strongest_components(components: dict) -> str:
    labels = {
        "semantic_similarity": "의미 유사도",
        "emotion_match": "감정 공감",
        "topic_match": "주제 일치",
        "situation_fit": "상황 적합도",
        "support_fit": "지원 방향",
        "affect_alignment": "정서 톤 적합도",
        "emotion_regulation": "감정 조절",
        "recovery_potential": "회복 가능성",
        "calm_fit": "안정 전환",
        "cognitive_load_fit": "인지 부담 적합도",
        "role_fit": "역할 적합도",
        "safety_score": "안전성",
        "personal_fit": "개인화",
    }
    top = sorted(
        ((key, value) for key, value in components.items() if key in labels),
        key=lambda value: value[1],
        reverse=True,
    )[:2]
    return ", ".join(labels[key] for key, _ in top) or "상황 적합도"


def input_sentence(
    *,
    dominant: str,
    sub_emotions: list[str],
    context: list[str],
    situation: dict,
) -> str:
    context_text = ", ".join(context[:3])
    emotion_parts = [item for item in [dominant, *sub_emotions[:2]] if item]
    emotion_text = ", ".join(dict.fromkeys(emotion_parts)) or "오늘의 감정"
    if dominant == "뿌듯함" and any(item in sub_emotions for item in ["피로", "무기력"]):
        return f"{context_text} 뒤에 남은 뿌듯함과 피로를 함께 기준으로 잡았습니다."
    if situation.get("intent") == "positive_emotion_based":
        return f"{context_text or '오늘의 기록'}에서 보인 {emotion_text}을 오래 해치지 않는 방향으로 골랐습니다."
    if situation.get("intent") == "neutral_daily_based":
        return f"{context_text or '잔잔한 일상'}의 흐름을 크게 흔들지 않는 쪽을 우선했습니다."
    return f"{context_text or '오늘의 기록'}에서 보인 {emotion_text} 감정을 기준으로 골랐습니다."


def content_tag_sentence(item: dict) -> str:
    tags = []
    tags.extend(item.get("display_tags", []))
    if not tags:
        tags.extend(item.get("emotion_tags", [])[:2])
        tags.extend(item.get("topic_tags", [])[:2])
        tags.extend(item.get("content_affect_profile", {}).get("dominant_tone", [])[:1])
    tags = list(dict.fromkeys([tag for tag in tags if tag]))[:4]
    if not tags:
        return "저부담"
    return ", ".join(tags)


def role_sentence(
    *,
    role: str,
    label: str,
    title: str,
    tag_note: str,
    component_note: str,
    affect_profile: dict,
) -> str:
    load = affect_profile.get("cognitive_load", 0.0)
    load_note = "부담이 낮은 편이라" if load <= 0.45 else "생각할 거리가 조금 있어"
    templates = {
        "공감": f"{label} 「{title}」는 {tag_note} 태그가 있어 지금 마음을 급히 바꾸기보다 따라가기에 좋습니다.",
        "회복": f"{label} 「{title}」는 {tag_note} 결이 있고 {load_note} 피곤한 뒤에 천천히 회복하기 좋습니다.",
        "전환": f"{label} 「{title}」는 {tag_note} 흐름이라 하루 끝의 기분을 살짝 환기하는 쪽에 가깝습니다.",
        "증폭": f"{label} 「{title}」는 {tag_note} 태그가 강해 남아 있는 좋은 에너지를 밝게 이어가기 좋습니다.",
        "축하": f"{label} 「{title}」는 {tag_note} 태그와 {component_note} 기준에서 오늘 해낸 일을 가볍게 축하하는 데 맞습니다.",
        "음미": f"{label} 「{title}」는 {tag_note} 결이 있어 좋은 여운을 크게 흔들지 않고 머물게 합니다.",
        "영감": f"{label} 「{title}」는 {tag_note} 쪽의 신호가 있어 다음 작업을 떠올리는 데 도움을 줄 수 있습니다.",
        "유지": f"{label} 「{title}」는 {tag_note} 분위기라 일상 만족감을 크게 흔들지 않고 곁에 두기 좋습니다.",
        "발견": f"{label} 「{title}」는 {tag_note} 태그를 가진 새 후보라 부담 없이 취향을 넓혀보기 좋습니다.",
        "배경": f"{label} 「{title}」는 {tag_note} 흐름이고 {load_note} 정리하거나 쉬는 시간의 배경으로 어울립니다.",
        "가벼운 성찰": f"{label} 「{title}」는 {tag_note} 결이 있어 하루를 짧게 돌아보는 데 맞습니다.",
        "탐색": f"{label} 「{title}」는 {tag_note} 단서가 있어 오늘의 관심사를 조금 넓히는 후보입니다.",
        "학습": f"{label} 「{title}」는 {tag_note} 축이 보여 궁금한 주제를 더 알아가는 입구로 괜찮습니다.",
    }
    return templates.get(
        role,
        f"{label} 「{title}」는 {tag_note} 태그와 {component_note} 기준에서 오늘 분위기에 무리 없이 맞습니다.",
    )
