from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.controllers import BackendController
from app.database import Database


def evaluate_case(controller: BackendController, case: dict[str, Any]) -> dict[str, Any]:
    user_id = 9000 + abs(hash(case["id"])) % 1000
    diary = controller.create_diary(
        {"user_id": user_id, "input_type": "text", "text": case["text"]}
    )
    recommendations = controller.recommend(
        {
            "entry_id": diary["entry_id"],
            "content_types": ["movie", "music", "book"],
            "per_type": case.get("per_type", 4),
        }
    )
    checks = {
        "role_sequence": recommendations["recommendation_roles"]
        == case["expected_roles"],
        "counts": all(
            len(items) == case.get("per_type", 4)
            for items in recommendations["recommendations"].values()
        ),
        "components": all(
            all(
                component in item["score_components"]
                for component in case["required_components"]
            )
            for items in recommendations["recommendations"].values()
            for item in items
        ),
        "topic_presence": any(
            set(item["topic_tags"]) & set(case["expected_any_topic"])
            for items in recommendations["recommendations"].values()
            for item in items
        ),
        "safety": recommendations["algorithm_profile"]["blocked_by_safety"]
        <= case.get("blocked_by_safety_max", 999),
    }
    controller.delete_user_data(user_id)
    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "actual": {
            "roles": recommendations["recommendation_roles"],
            "top_titles": {
                key: [item["title"] for item in value[:2]]
                for key, value in recommendations["recommendations"].items()
            },
            "top_roles": {
                key: [item["recommendation_role"] for item in value]
                for key, value in recommendations["recommendations"].items()
            },
            "blocked_by_safety": recommendations["algorithm_profile"]["blocked_by_safety"],
        },
    }


def main() -> None:
    cases = json.loads(
        Path("data/evaluation/recommendation_cases.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        controller = BackendController(
            database=Database(
                Path(temp_dir) / "recommendation_eval.sqlite3",
                Path("data/samples/content_items.json"),
            )
        )
        results = [evaluate_case(controller, case) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    report = {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / max(1, len(results)), 3),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

