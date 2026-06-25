from __future__ import annotations

import re
from collections import defaultdict

from tagger.tag_config import KEYWORD_RULES


TagScores = dict[str, dict[str, float]]
TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def keyword_matches(text: str, tokens: set[str], keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword:
        return False
    if " " in keyword:
        return keyword in text
    if len(keyword) == 1:
        return any(
            token == keyword
            or (
                token.startswith(keyword)
                and len(token) <= 3
                and not token.startswith(("특" + keyword, "개" + keyword))
            )
            for token in tokens
        )
    return any(token == keyword or token.startswith(keyword) for token in tokens)


def apply_keyword_rules(text: str) -> TagScores:
    lowered = text.lower()
    tokens = set(TOKEN_RE.findall(lowered))
    result: dict[str, defaultdict[str, float]] = {
        "emotion_tags": defaultdict(float),
        "theme_tags": defaultdict(float),
        "mood_tags": defaultdict(float),
        "recommendation_roles": defaultdict(float),
    }
    for rule in KEYWORD_RULES:
        hits = sum(1 for keyword in rule["keywords"] if keyword_matches(lowered, tokens, keyword))
        if not hits:
            continue
        boost = min(1.35, 1.0 + (hits - 1) * 0.12)
        for bucket in result:
            for tag, score in rule.get(bucket, {}).items():
                result[bucket][tag] = max(result[bucket][tag], min(1.0, score * boost))
    return {
        bucket: {tag: round(score, 4) for tag, score in scores.items()}
        for bucket, scores in result.items()
    }
