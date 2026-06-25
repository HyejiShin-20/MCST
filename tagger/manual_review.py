from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Database


def split_tags(value: str) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual tag override")
    parser.add_argument("--content-id", required=True)
    parser.add_argument("--emotion-tags", default="")
    parser.add_argument("--theme-tags", default="")
    parser.add_argument("--mood-tags", default="")
    parser.add_argument("--roles", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--reviewed-by", default="manual")
    args = parser.parse_args()

    database = Database()
    database.initialize()
    database.apply_manual_tag_override(
        content_id=args.content_id,
        emotion_tags=split_tags(args.emotion_tags),
        theme_tags=split_tags(args.theme_tags),
        mood_tags=split_tags(args.mood_tags),
        recommendation_roles=split_tags(args.roles),
        note=args.note,
        reviewed_by=args.reviewed_by,
    )
    print(json.dumps({"content_id": args.content_id, "manual_override": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
