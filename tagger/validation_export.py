from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.database import Database


def export_rows(args: argparse.Namespace) -> dict[str, Any]:
    database = Database()
    database.initialize()
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    content_type = None if args.content_type == "all" else args.content_type
    rows = database.list_contents(
        [content_type] if content_type else None,
        recommendable_only=False,
    )
    selected = _select_rows(rows, args)
    output_name = _output_name(args)
    csv_path = config.EXPORT_DIR / f"{output_name}.csv"
    _write_csv(csv_path, selected[: args.limit])
    distribution = _distribution(rows)
    distribution_path = config.EXPORT_DIR / "tag_distribution.json"
    distribution_path.write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "csv_path": str(csv_path),
        "distribution_path": str(distribution_path),
        "row_count": len(selected[: args.limit]),
        "total_contents": len(rows),
        "distribution": distribution,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows
    if args.low_confidence:
        selected = [
            row
            for row in selected
            if float(row.get("tag_confidence") or 0.0) < config.HIGH_CONFIDENCE_THRESHOLD
        ]
    if args.tag:
        selected = [
            row
            for row in selected
            if args.tag in row.get("emotion_tags", [])
            or args.tag in row.get("topic_tags", [])
            or args.tag in row.get("mood_tags", [])
        ]
    if args.role:
        selected = [
            row
            for row in selected
            if args.role in row.get("recommendation_roles", [])
        ]
    return sorted(
        selected,
        key=lambda row: float(row.get("tag_confidence") or 0.0),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "content_id",
        "content_type",
        "title",
        "creator",
        "tag_status",
        "tag_version",
        "tag_confidence",
        "is_recommendable",
        "emotion_tags",
        "theme_tags",
        "mood_tags",
        "recommendation_roles",
        "tagging_text_preview",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "content_id": row.get("content_id", ""),
                    "content_type": row.get("content_type", ""),
                    "title": row.get("title", ""),
                    "creator": row.get("creator", ""),
                    "tag_status": row.get("tag_status", ""),
                    "tag_version": row.get("tag_version", ""),
                    "tag_confidence": row.get("tag_confidence", ""),
                    "is_recommendable": row.get("is_recommendable", ""),
                    "emotion_tags": "|".join(row.get("emotion_tags", [])),
                    "theme_tags": "|".join(row.get("topic_tags", [])),
                    "mood_tags": "|".join(row.get("mood_tags", [])),
                    "recommendation_roles": "|".join(row.get("recommendation_roles", [])),
                    "tagging_text_preview": (row.get("tagging_text") or row.get("summary", ""))[:180],
                }
            )


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    emotion = Counter()
    theme = Counter()
    mood = Counter()
    roles = Counter()
    status = Counter()
    content_types = Counter()
    for row in rows:
        emotion.update(row.get("emotion_tags", []))
        theme.update(row.get("topic_tags", []))
        mood.update(row.get("mood_tags", []))
        roles.update(row.get("recommendation_roles", []))
        status.update([row.get("tag_status", "unknown")])
        content_types.update([row.get("content_type", "unknown")])
    return {
        "emotion_tags": emotion.most_common(),
        "theme_tags": theme.most_common(),
        "mood_tags": mood.most_common(),
        "recommendation_roles": roles.most_common(),
        "tag_status": status.most_common(),
        "content_types": content_types.most_common(),
    }


def _output_name(args: argparse.Namespace) -> str:
    if args.low_confidence:
        return "low_confidence_tags"
    if args.tag:
        return f"top_by_tag_{args.tag}"
    if args.role:
        return f"top_by_role_{args.role}"
    return "tag_review_export"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export content tagging review files")
    parser.add_argument("--low-confidence", action="store_true")
    parser.add_argument("--distribution", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--role", default="")
    parser.add_argument("--content-type", choices=["movie", "music", "book", "all"], default="all")
    parser.add_argument("--limit", type=int, default=200)
    return parser


def main() -> None:
    export_rows(build_parser().parse_args())


if __name__ == "__main__":
    main()
