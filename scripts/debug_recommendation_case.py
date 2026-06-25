from __future__ import annotations

import json
import sys
import time
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def request_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten(recommendations: dict) -> list[dict]:
    return [
        item
        for content_type in ("music", "book")
        for item in recommendations.get(content_type, [])
    ]


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: debug_recommendation_case.py <text>")
    text = sys.argv[1]
    user_id = 1
    diary = request_json(
        "/api/diary",
        {
            "user_id": user_id,
            "input_type": "text",
            "text": text,
        },
    )
    recs = request_json(
        "/api/recommend",
        {
            "entry_id": diary["entry_id"],
            "content_types": ["music", "book"],
            "per_type": 4,
        },
    )
    report = {
        "entry_id": diary["entry_id"],
        "analysis": {
            "emotions": diary["analysis"].get("emotions"),
            "emotion_scores": diary["analysis"].get("emotion_scores"),
            "topics": diary["analysis"].get("topics"),
            "context_keywords": diary["analysis"].get("context_keywords"),
            "situation": diary["analysis"].get("situation"),
            "desired_support": diary["analysis"].get("desired_support"),
            "structured_emotion": diary["analysis"].get("structured_emotion"),
            "summary": diary["analysis"].get("summary"),
        },
        "recommendations": [
            {
                "title": item.get("title"),
                "type": item.get("content_type"),
                "score_percent": item.get("score_percent"),
                "tags": item.get("display_tags"),
                "emotions": item.get("emotion_tags"),
                "topics": item.get("topic_tags"),
                "role": item.get("recommendation_role"),
                "components": item.get("score_components"),
                "reason": item.get("reason"),
            }
            for item in flatten(recs.get("recommendations", {}))[:8]
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
