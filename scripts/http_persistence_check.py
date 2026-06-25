from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


def request_json(path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    username = f"smoke_{int(time.time())}"
    user = request_json(
        "/api/users",
        {"username": username, "display_name": "Smoke Test"},
    )["user"]
    user_id = user["user_id"]
    diary_text = (
        "오늘은 코딩 프로젝트를 진행하고 집에 와서는 운동을 했다. "
        "피곤하지만 뿌듯했고, 내일은 조금 더 차분하게 쉬고 싶다."
    )
    diary = request_json(
        "/api/diary",
        {"user_id": user_id, "input_type": "text", "text": diary_text},
    )
    recommendations = request_json(
        "/api/recommend",
        {
            "entry_id": diary["entry_id"],
            "content_types": ["music", "book"],
            "per_type": 2,
        },
    )
    candidates = (
        recommendations["recommendations"].get("music", [])
        + recommendations["recommendations"].get("book", [])
    )
    if not candidates:
        raise RuntimeError("recommendation result is empty")
    item = candidates[0]
    saved = request_json(
        "/api/repository/save",
        {
            "user_id": user_id,
            "content_id": item["content_id"],
            "content_type": item["content_type"],
            "title": item["title"],
            "creator": item.get("creator", ""),
            "genre": item.get("genre", ""),
            "source": item.get("source", ""),
            "entry_id": diary["entry_id"],
            "log_id": recommendations["log_id"],
            "recommendation_role": item.get("recommendation_role", ""),
            "reason": item.get("reason", ""),
            "score_percent": item.get("score_percent", 0),
            "display_tags": item.get("display_tags", []),
            "snapshot": item,
        },
    )
    repository = request_json(f"/api/repository?{urllib.parse.urlencode({'user_id': user_id})}")
    activity = request_json(f"/api/activity?{urllib.parse.urlencode({'user_id': user_id})}")
    taste = request_json(
        f"/api/taste-analysis?{urllib.parse.urlencode({'user_id': user_id, 'use_openai': 'false'})}"
    )
    print(
        json.dumps(
            {
                "user_id": user_id,
                "entry_id": diary["entry_id"],
                "log_id": recommendations["log_id"],
                "saved_title": saved["saved_item"]["title"],
                "saved_type": saved["saved_item"]["content_type"],
                "repository_count": repository["counts"]["total"],
                "activity_summary": activity["summary"],
                "taste_headline": taste["taste_summary"]["headline"],
                "content_ratio": taste["content_ratio"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
