"""로컬 DB(확장 + LLM 재태깅 완료)를 '시드 JSONL'로 내보낸다.

목적: LLM 태깅 결과(무드/역할/valence/confidence/flags 포함)를 시드 파일에 구워넣어
배포(Render) 때마다 동일한 고품질 태그로 재현되게 한다. (A안)

각 줄(JSON)에는 기존 시드 호환 필드 + 'tags'(전체 최종 태그)가 들어간다.
database.seed_content_if_empty() 가 'tags' 가 있으면 그대로 content_tags 에 시드한다.

사용:
  python tagger/export_seed_catalog.py
  python tagger/export_seed_catalog.py --out data/processed/culture_import_music_book_movie.jsonl
  python tagger/export_seed_catalog.py --only-recommendable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Database

DEFAULT_OUT = "data/processed/culture_import_music_book_movie.jsonl"


def _row_to_seed_item(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("tag_metadata") or {}
    emotion_tags = row.get("emotion_tags") or []
    theme_tags = row.get("topic_tags") or []
    tags = {
        "tag_method": row.get("tag_method") or "llm_vocab_v1",
        "tag_version": row.get("tag_version") or "",
        "emotion_tags": emotion_tags,
        "theme_tags": theme_tags,
        "mood_tags": row.get("mood_tags") or [],
        "recommendation_roles": row.get("recommendation_roles") or [],
        "valence": meta.get("valence", 0.0),
        "arousal": meta.get("arousal", 0.35),
        "intensity": meta.get("intensity", 0.4),
        "energy": meta.get("energy", 0.45),
        "cognitive_load": meta.get("cognitive_load", 0.45),
        "tag_confidence": row.get("tag_confidence", 0.7),
        "is_recommendable": bool(row.get("is_recommendable", 1)),
        "dark_flag": bool(meta.get("dark_flag", False)),
        "too_heavy_flag": bool(meta.get("too_heavy_flag", False)),
        "background_friendly": bool(meta.get("background_friendly", False)),
    }
    return {
        "content_id": row["content_id"],
        "content_type": row["content_type"],
        "title": row.get("title", ""),
        "creator": row.get("creator", ""),
        "genre": row.get("genre", ""),
        "summary": row.get("summary", ""),
        "source": row.get("source", ""),
        # 기존 시드 호환 필드
        "emotion_tags": emotion_tags,
        "topic_tags": theme_tags,
        # LLM 전체 태그
        "tags": tags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DB content as an enriched seed JSONL")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--only-recommendable", action="store_true", help="추천 가능 항목만 내보내기")
    args = parser.parse_args()

    database = Database()
    database.initialize()
    rows = database.list_contents()
    if args.only_recommendable:
        rows = [row for row in rows if int(row.get("is_recommendable", 0)) == 1]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    by_type: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            item = _row_to_seed_item(row)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            by_type[item["content_type"]] = by_type.get(item["content_type"], 0) + 1

    print(f"[export] {written}개 시드 항목 → {out_path}")
    print(f"  타입별: {json.dumps(by_type, ensure_ascii=False)}")
    print("  다음: git add 후 커밋/푸시 → Render 재배포 시 이 태그로 시드됩니다.")


if __name__ == "__main__":
    main()
