from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.controllers import BackendController
from app.database import Database


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "smoke.sqlite3"
        sample_path = Path("data/samples/content_items.json")
        controller = BackendController(database=Database(db_path, sample_path))

        diary = controller.create_diary(
            {
                "user_id": 7,
                "input_type": "text",
                "text": "오늘 발표를 망쳐서 너무 창피하고 불안했다. 다시 자신감을 찾고 싶다.",
            }
        )
        recommendations = controller.recommend(
            {
                "entry_id": diary["entry_id"],
                "content_types": ["movie", "music", "book"],
                "per_type": 3,
            }
        )
        profile = controller.get_profile(7)
        reset = controller.reset_memory(7)
        deleted = controller.delete_user_data(7)

        result = {
            "health": controller.health(),
            "entry_id": diary["entry_id"],
            "analysis": diary["analysis"],
            "recommendation_counts": {
                key: len(value)
                for key, value in recommendations["recommendations"].items()
            },
            "profile_exists": profile["profile"] is not None,
            "reset": reset["memory_reset"],
            "deleted": deleted["deleted"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
