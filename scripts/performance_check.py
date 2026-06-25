from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.controllers import BackendController
from app.database import Database


CASES = [
    {
        "id": "negative_relationship",
        "label": "부정/관계 상처",
        "text": "친구가 내 말을 무시해서 화났는데, 집에 와서는 내가 별로 중요한 사람이 아닌 것 같아서 울었다.",
    },
    {
        "id": "positive_achievement",
        "label": "긍정/성취",
        "text": "오늘 발표가 생각보다 잘 끝났다. 긴장했는데 끝나고 나니까 너무 뿌듯했다.",
    },
    {
        "id": "neutral_daily",
        "label": "중립/잔잔한 일상",
        "text": "오늘은 별일 없었다. 집에 와서 밥 먹고 씻고 그냥 누워 있었다.",
    },
    {
        "id": "interest_space",
        "label": "관심사/우주",
        "text": "오늘 우주 관련 영상을 봤는데 갑자기 우주비행사가 되고 싶어졌다.",
    },
    {
        "id": "mixed_joy_fatigue",
        "label": "복합/기쁨과 피로",
        "text": "오늘 좋은 일이 있어서 기분은 좋은데 너무 피곤해서 아무것도 크게 하고 싶지는 않다.",
    },
]


def database_summary(database: Database) -> dict[str, Any]:
    with sqlite3.connect(database.db_path) as connection:
        connection.row_factory = sqlite3.Row
        total_contents = connection.execute(
            "SELECT COUNT(*) AS count FROM content_items"
        ).fetchone()["count"]
        total_tags = connection.execute(
            "SELECT COUNT(*) AS count FROM content_tags"
        ).fetchone()["count"]
        status_counts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT tag_status, COUNT(*) AS count
                FROM content_items
                GROUP BY tag_status
                ORDER BY tag_status
                """
            )
        ]
        type_counts = [
            dict(row)
            for row in connection.execute(
                """
                SELECT content_type, COUNT(*) AS count
                FROM content_items
                GROUP BY content_type
                ORDER BY content_type
                """
            )
        ]
        recommendable_count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM content_items
            WHERE tag_status = 'tagged'
              AND is_recommendable = 1
              AND tag_confidence >= ?
            """,
            (config.MIN_RECOMMEND_CONFIDENCE,),
        ).fetchone()["count"]
        confidences = [
            float(row["tag_confidence"])
            for row in connection.execute(
                "SELECT tag_confidence FROM content_items ORDER BY content_id"
            )
        ]
    return {
        "total_contents": total_contents,
        "total_tag_rows": total_tags,
        "tag_status_counts": status_counts,
        "content_type_counts": type_counts,
        "recommendable_count": recommendable_count,
        "min_recommend_confidence": config.MIN_RECOMMEND_CONFIDENCE,
        "confidence": {
            "min": round(min(confidences), 4) if confidences else 0,
            "max": round(max(confidences), 4) if confidences else 0,
            "avg": round(mean(confidences), 4) if confidences else 0,
        },
    }


def compact_recommendations(recommendations: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for content_type, items in recommendations.items():
        compact[content_type] = [
            {
                "title": item["title"],
                "creator": item["creator"],
                "role": item["recommendation_role"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in items[:2]
        ]
    return compact


def run_cases(controller: BackendController) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    user_id = 99001
    for case in CASES:
        diary = controller.create_diary(
            {
                "user_id": user_id,
                "input_type": "text",
                "text": case["text"],
            }
        )
        rec = controller.recommend(
            {
                "entry_id": diary["entry_id"],
                "content_types": ["movie", "music", "book"],
                "per_type": 3,
            }
        )
        results.append(
            {
                "id": case["id"],
                "label": case["label"],
                "input": case["text"],
                "analysis": {
                    "emotions": diary["analysis"]["emotions"],
                    "topics": diary["analysis"]["topics"],
                    "intent": diary["analysis"]["situation"]["intent"],
                    "primary_event": diary["analysis"]["situation"]["primary_event"],
                    "roles": rec["recommendation_roles"],
                },
                "recommendations": compact_recommendations(rec["recommendations"]),
            }
        )
        controller.delete_user_data(user_id)
    return results


def main() -> None:
    database = Database()
    controller = BackendController(database=database)
    report = {
        "database": database_summary(database),
        "cases": run_cases(controller),
    }
    output_path = config.EXPORT_DIR / "performance_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_path": str(output_path), **report}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
