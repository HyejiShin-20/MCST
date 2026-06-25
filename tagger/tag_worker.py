from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.database import Database
from tagger.embedding import embedding_model_name, get_embedding_hash
from tagger.keyword_rules import apply_keyword_rules
from tagger.preprocess import build_tagging_text, text_quality_score
from tagger.prototype_tagger import finalize_tags, merge_tags, tag_by_prototype


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def tag_content(content: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    processed_text, tagging_text = build_tagging_text(content)
    quality = text_quality_score(processed_text)
    if quality < 0.15:
        raise ValueError("TEXT_TOO_SHORT")
    keyword_result = apply_keyword_rules(tagging_text)
    keyword_result = merge_seed_tags(keyword_result, content)
    prototype_result = tag_by_prototype(tagging_text)
    merged = merge_tags(keyword_result, prototype_result)
    final = finalize_tags(
        merged,
        content_type=str(content.get("content_type", "")),
        text_quality=quality,
    )
    removed_tags = sanitize_final_tags(final, tagging_text, content)
    if _list_field(content, "emotion_tags", "emotion_tags_json") or _list_field(content, "topic_tags", "topic_tags_json"):
        final["tag_confidence"] = max(final["tag_confidence"], 0.62)
        final["is_recommendable"] = final["is_recommendable"] or quality >= 0.35
    raw = {
        "keyword": keyword_result,
        "prototype": prototype_result,
        "merged_scores": merged["scores"],
        "sanitized_removed_tags": removed_tags,
        "text_quality_score": quality,
        "tagging_text_preview": tagging_text[:240],
    }
    tag_result = {
        "tag_method": "keyword_prototype",
        "processed_text": processed_text,
        "tagging_text": tagging_text,
        "raw": raw,
        "final": final,
    }
    model_name = embedding_model_name()
    embedding_hash = get_embedding_hash(tagging_text, model_name)
    return tag_result, embedding_hash, model_name


def sanitize_final_tags(
    final: dict[str, Any], tagging_text: str, content: dict[str, Any]
) -> dict[str, list[str]]:
    removed: dict[str, list[str]] = {
        "emotion_tags": [],
        "theme_tags": [],
        "mood_tags": [],
        "recommendation_roles": [],
    }
    text = tagging_text.lower()
    tokens = set(TOKEN_RE.findall(text))
    genre = str(content.get("genre") or "").lower()

    if not _has_space_signal(text, tokens, genre):
        _remove_tag(final, removed, "theme_tags", "우주")
        _remove_tag(final, removed, "theme_tags", "과학")
        for role in ["학습", "탐색", "몰입"]:
            if role in final.get("recommendation_roles", []) and not _has_any(text, ["학습", "공부", "지식", "탐구", "과학", "우주", "기술"]):
                _remove_tag(final, removed, "recommendation_roles", role)

    evidence_rules = {
        "분노": ["분노", "화가", "화났", "짜증", "억울", "배신", "싸움", "갈등"],
        "불안": ["불안", "걱정", "긴장", "초조", "두렵", "압박"],
        "설렘": ["설레", "설렘", "두근", "고백", "연인", "사랑", "기대"],
        "이별": ["이별", "헤어", "그립", "눈물", "빈자리", "상실"],
        "무기력": ["무기력", "지쳤", "피곤", "피로", "번아웃", "아무것도", "하기 싫"],
    }
    for tag, cues in evidence_rules.items():
        if tag in final.get("emotion_tags", []) and not _has_any(text, cues):
            _remove_tag(final, removed, "emotion_tags", tag)
        if tag in final.get("theme_tags", []) and not _has_any(text, cues):
            _remove_tag(final, removed, "theme_tags", tag)

    if not final.get("emotion_tags"):
        final["emotion_tags"] = ["평범함"]
    if not final.get("theme_tags"):
        final["theme_tags"] = ["일상"]
    if not final.get("mood_tags"):
        final["mood_tags"] = ["저부담"]
    if not final.get("recommendation_roles"):
        final["recommendation_roles"] = ["발견"]
    final["dark_flag"] = any(
        tag in final.get("emotion_tags", []) + final.get("mood_tags", [])
        for tag in ["우울", "어두움", "강렬함", "긴장감"]
    )
    final["background_friendly"] = any(
        tag in final.get("mood_tags", []) + final.get("recommendation_roles", [])
        for tag in ["배경친화", "저부담", "배경", "유지"]
    )
    return {bucket: tags for bucket, tags in removed.items() if tags}


def _remove_tag(
    final: dict[str, Any], removed: dict[str, list[str]], bucket: str, tag: str
) -> None:
    values = list(final.get(bucket, []))
    if tag not in values:
        return
    final[bucket] = [value for value in values if value != tag]
    removed[bucket].append(tag)


def _has_space_signal(text: str, tokens: set[str], genre: str) -> bool:
    if "과학" in genre or "science" in genre:
        return True
    if _has_any(text, ["우주", "행성", "로켓", "천문", "은하", "별빛", "별자리", "우주비행사", "과학"]):
        return True
    return any(token in {"별", "별을", "별이", "별과"} for token in tokens)


def _has_any(text: str, cues: list[str]) -> bool:
    return any(cue in text for cue in cues)


def merge_seed_tags(
    keyword_result: dict[str, dict[str, float]], content: dict[str, Any]
) -> dict[str, dict[str, float]]:
    merged = {
        bucket: dict(scores)
        for bucket, scores in keyword_result.items()
    }
    sample_source = str(content.get("source") or "").startswith("MVP curated sample")
    emotion_tags = _list_field(content, "emotion_tags", "emotion_tags_json") if sample_source else []
    theme_tags = _list_field(content, "topic_tags", "topic_tags_json")
    if not sample_source:
        theme_tags = [tag for tag in theme_tags if is_structural_seed_tag(tag, content)]
    for tag in emotion_tags:
        merged.setdefault("emotion_tags", {})[tag] = max(
            merged.setdefault("emotion_tags", {}).get(tag, 0.0),
            0.76,
        )
    for tag in theme_tags:
        merged.setdefault("theme_tags", {})[tag] = max(
            merged.setdefault("theme_tags", {}).get(tag, 0.0),
            0.76,
        )
    return merged


def is_structural_seed_tag(tag: str, content: dict[str, Any]) -> bool:
    tag = str(tag).strip()
    genre = str(content.get("genre") or "").strip()
    source = str(content.get("source") or "")
    if not tag:
        return False
    if tag == genre or tag in set(part.strip() for part in genre.split("/") if part.strip()):
        return True
    if "/" in tag:
        return True
    if tag.endswith("년대") and tag[:4].isdigit():
        return True
    structural = {
        "베스트셀러",
        "장르종합",
        "발라드",
        "댄스",
        "랩/힙합",
        "R&B/Soul",
        "록/메탈",
        "인디",
        "트로트",
        "경제/자기계발",
        "소설/판타지",
    }
    if tag in structural:
        return True
    return "교보문고" in source and tag in {"베스트셀러"}


def _list_field(content: dict[str, Any], list_key: str, json_key: str) -> list[str]:
    value = content.get(list_key)
    if isinstance(value, list):
        return [str(item) for item in value]
    raw = content.get(json_key)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return []
    return []


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    database = Database()
    database.initialize()
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    targets = database.list_tagging_targets(
        content_type=args.content_type,
        limit=args.limit or args.batch_size,
        retag=args.retag,
        only_low_confidence=args.only_low_confidence,
    )
    success = 0
    failed = 0
    skipped = 0
    confidences: list[float] = []
    errors: Counter[str] = Counter()
    dry_run_items: list[dict[str, Any]] = []

    for content in targets[: args.batch_size]:
        content_id = str(content["content_id"])
        try:
            if not args.dry_run:
                database.mark_content_processing(content_id)
            tag_result, embedding_hash, model_name = tag_content(content)
            confidences.append(float(tag_result["final"]["tag_confidence"]))
            if args.dry_run:
                dry_run_items.append(
                    {
                        "content_id": content_id,
                        "title": content["title"],
                        "final": tag_result["final"],
                    }
                )
            else:
                database.save_content_tagging_result(
                    content_id=content_id,
                    tag_result=tag_result,
                    embedding_hash=embedding_hash,
                    embedding_model_name=model_name,
                )
            success += 1
        except Exception as exc:
            failed += 1
            error_type = str(exc) if str(exc).isupper() else "UNKNOWN_ERROR"
            errors[error_type] += 1
            if not args.dry_run:
                database.mark_content_tagging_failed(content_id, error_type, str(exc))

    target_count = len(targets[: args.batch_size])
    if len(targets) > target_count:
        skipped += len(targets) - target_count
    finished = datetime.now(timezone.utc)
    report = {
        "tag_version": config.CURRENT_TAG_VERSION,
        "tag_method": "keyword_prototype",
        "content_type": args.content_type,
        "batch_size": args.batch_size,
        "target_count": target_count,
        "success_count": success,
        "failed_count": failed,
        "skipped_count": skipped,
        "average_confidence": round(sum(confidences) / max(1, len(confidences)), 4),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round(time.perf_counter() - start_time, 3),
        "run_options": vars(args),
        "error_summary": dict(errors),
        "dry_run_items": dry_run_items[:20],
    }
    if not args.dry_run:
        report["run_id"] = database.record_tag_run(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Content auto-tagging worker")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--content-type", choices=["movie", "music", "book", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retag", action="store_true")
    parser.add_argument("--only-low-confidence", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_worker(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
