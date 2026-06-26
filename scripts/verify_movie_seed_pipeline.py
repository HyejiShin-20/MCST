from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OPENAI_ANALYSIS_ENABLED"] = "false"
os.environ["RETRIEVAL_BACKEND"] = "memory"

from app.controllers import BackendController
from app.database import Database


CASES = [
    {
        "label": "positive_daily",
        "text": "밀크티 마시고 할 일 하는 하루 최고. 기분 좋고 여유롭다.",
    },
    {
        "label": "tired_recovery",
        "text": "집에 가고 싶다. 너무 피곤하고 힘들어서 그냥 빨리 자고 싶다.",
    },
    {
        "label": "curious_movie",
        "text": "요즘 영화가 너무 보고 싶다. 가볍게 웃을 수 있는 한국 코미디나 드라마를 찾고 싶다.",
    },
]


def compact_recommendations(recommendations: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    compact: dict[str, list[dict[str, Any]]] = {}
    for content_type, items in recommendations.get("recommendations", {}).items():
        compact[content_type] = [
            {
                "title": item.get("title"),
                "creator": item.get("creator"),
                "genre": item.get("genre"),
                "score_percent": item.get("score_percent"),
                "display_tags": item.get("display_tags"),
                "reason": item.get("reason"),
            }
            for item in items[:3]
        ]
    return compact


def main() -> None:
    database = Database()
    database.initialize()
    controller = BackendController(database=database)
    report: dict[str, Any] = {
        "health": controller.health(),
        "db_counts": {},
        "cases": [],
    }

    with database.connect() as connection:
        report["db_counts"] = {
            "content_by_type": [
                list(row)
                for row in connection.execute(
                """
                SELECT content_type, COUNT(*)
                FROM content_items
                GROUP BY content_type
                ORDER BY content_type
                """
                ).fetchall()
            ],
            "movie_recommendable": connection.execute(
                """
                SELECT COUNT(*)
                FROM content_items
                WHERE content_type = 'movie'
                  AND tag_status = 'tagged'
                  AND is_recommendable = 1
                  AND tag_confidence >= 0.55
                """
            ).fetchone()[0],
        }

    for offset, case in enumerate(CASES, start=1):
        user_id = 91000 + offset
        diary = controller.create_diary(
            {
                "user_id": user_id,
                "input_type": "text",
                "text": case["text"],
            }
        )
        recommendations = controller.recommend(
            {
                "entry_id": diary["entry_id"],
                "content_types": ["movie", "music", "book"],
                "per_type": 3,
            }
        )
        report["cases"].append(
            {
                "label": case["label"],
                "input": case["text"],
                "analysis": {
                    "emotions": diary["analysis"].get("emotions"),
                    "context_keywords": diary["analysis"].get("context_keywords"),
                    "recommendation_strategy": diary["analysis"].get("recommendation_strategy"),
                },
                "recommendation_counts": {
                    key: len(value)
                    for key, value in recommendations.get("recommendations", {}).items()
                },
                "top_recommendations": compact_recommendations(recommendations),
            }
        )
        controller.delete_user_data(user_id)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
