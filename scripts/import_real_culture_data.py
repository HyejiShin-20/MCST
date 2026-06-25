from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import Database
from app.services.text_utils import normalize_text
from tagger.preprocess import preprocess_lyrics


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MELON_CHART_PATH = RAW_DIR / "music" / "melon_monthly_top50.csv"
MELON_LYRICS_PATH = RAW_DIR / "music" / "melon_song_lyrics.csv"
KYOBO_PATH = (
    RAW_DIR
    / "교보문고_연도별_베스트셀러(2000~2025)"
    / "kyobo_annual_bestsellers_2000_2025.csv"
)

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")

STOPWORDS = {
    "그리고",
    "그래서",
    "하지만",
    "나는",
    "너는",
    "우리는",
    "그대",
    "우리",
    "너를",
    "나를",
    "다시",
    "아직",
    "정말",
    "오늘",
    "내일",
    "지금",
    "그냥",
    "모든",
    "없는",
    "있는",
    "하게",
    "해서",
    "보다",
    "처럼",
}

EMOTION_RULES: dict[str, list[str]] = {
    "슬픔": ["이별", "눈물", "슬프", "그립", "외로", "아프", "떠나", "울", "어둠", "상처"],
    "그리움": ["그리", "기억", "추억", "다시", "돌아", "기다", "보고 싶"],
    "분노": ["분노", "미워", "화가", "억울", "배신", "소리쳐", "참을 수"],
    "불안": ["불안", "두려", "걱정", "떨", "흔들", "막막", "길을 잃"],
    "무기력": ["지쳐", "피곤", "버거", "힘들", "멈춰", "무너", "끝난"],
    "기쁨": ["행복", "웃", "좋아", "신나", "축하", "반짝", "빛나"],
    "설렘": ["설레", "두근", "처음", "고백", "기대", "만나"],
    "감사": ["고마", "감사", "덕분", "소중"],
    "평온": ["평온", "잔잔", "조용", "쉬어", "바람", "하늘", "별", "안녕"],
    "희망": ["희망", "꿈", "내일", "빛", "살아", "괜찮", "일어나", "앞으로"],
    "호기심": ["궁금", "찾아", "모험", "새로운", "세계", "비밀"],
}

TOPIC_RULES: dict[str, list[str]] = {
    "사랑": ["사랑", "마음", "연인", "고백", "너와", "함께"],
    "이별": ["이별", "떠나", "헤어", "그리움", "눈물"],
    "위로": ["위로", "괜찮", "안아", "쉬어", "치유"],
    "성장": ["성장", "도전", "꿈", "성공", "미래", "시작"],
    "가족": ["가족", "엄마", "아빠", "부모", "아이"],
    "사회": ["사회", "민중", "자유", "세상", "역사", "정의"],
    "일상": ["하루", "집", "길", "밤", "아침", "시간"],
    "여행": ["여행", "바다", "기차", "길", "도시"],
    "판타지": ["마법", "모험", "비밀", "세계", "왕국"],
    "경제": ["돈", "부자", "투자", "경제", "기업", "재테크"],
    "인문": ["철학", "역사", "인간", "사회", "문명", "삶"],
    "과학": ["과학", "우주", "기술", "생명", "실험"],
    "학습": ["공부", "학습", "시험", "지식", "교훈"],
}


def clean(value: Any) -> str:
    return normalize_text(str(value or ""))


def norm_key(*parts: Any) -> str:
    joined = " ".join(clean(part).lower() for part in parts)
    return re.sub(r"[^0-9a-z가-힣]+", "", joined)


def stable_id(prefix: str, key: str) -> str:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}_{digest}"


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except Exception:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def tokenize(text: str) -> list[str]:
    tokens = [token for token in TOKEN_RE.findall(clean(text)) if token not in STOPWORDS]
    return [token for token in tokens if not token.isdigit()]


def top_keywords(text: str, limit: int = 8) -> list[str]:
    counter = Counter(tokenize(text))
    return [token for token, _ in counter.most_common(limit)]


def score_rules(text: str, rules: dict[str, list[str]], limit: int) -> list[str]:
    lowered = clean(text).lower()
    scored: list[tuple[str, int]] = []
    for tag, keywords in rules.items():
        hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if hits:
            scored.append((tag, hits))
    return [tag for tag, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:limit]]


def phrase_for_tags(tags: list[str]) -> str:
    if not tags:
        return "감정 단서가 약한 일상적 정서"
    mapping = {
        "슬픔": "상실감과 쓸쓸함",
        "그리움": "지나간 관계나 시간을 떠올리는 그리움",
        "분노": "강한 긴장과 억울함",
        "불안": "흔들림과 걱정",
        "무기력": "지친 마음과 낮은 에너지",
        "기쁨": "밝고 활기 있는 기분",
        "설렘": "기대와 두근거림",
        "감사": "따뜻한 고마움",
        "평온": "잔잔하고 안정적인 분위기",
        "희망": "다시 나아가려는 희망",
        "호기심": "새로운 것을 향한 호기심",
    }
    return ", ".join(mapping.get(tag, tag) for tag in tags[:3])


def generation_label(years: set[int]) -> str:
    if not years:
        return "연도 미상"
    median = sorted(years)[len(years) // 2]
    decade = median // 10 * 10
    return f"{decade}년대"


def read_bugs_years(raw_dir: Path) -> dict[str, set[int]]:
    try:
        from openpyxl import load_workbook  # type: ignore
    except Exception:
        return {}

    candidates = sorted(
        (path for path in raw_dir.rglob("*.xlsx") if "통합" in path.name),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(raw_dir.rglob("*.xlsx"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        return {}

    years_by_key: dict[str, set[int]] = defaultdict(set)
    workbook = load_workbook(candidates[0], read_only=True, data_only=True)
    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        headers = [clean(value) for value in next(rows, [])]
        year_idx = headers.index("연도") if "연도" in headers else 0
        title_idx = headers.index("곡") if "곡" in headers else 1
        artist_idx = headers.index("아티스트") if "아티스트" in headers else 2
        for row in rows:
            title = clean(row[title_idx] if len(row) > title_idx else "")
            artist = clean(row[artist_idx] if len(row) > artist_idx else "")
            year = parse_int(row[year_idx] if len(row) > year_idx else "")
            if title and artist and year:
                years_by_key[norm_key(title, artist)].add(year)
    finally:
        workbook.close()
    return years_by_key


def aggregate_melon_chart(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = clean(row.get("title"))
            artist = clean(row.get("artist"))
            if not title or not artist:
                continue
            key = norm_key(title, artist)
            record = records.setdefault(
                key,
                {
                    "title": title,
                    "artist": artist,
                    "albums": Counter(),
                    "genres": Counter(),
                    "years": set(),
                    "song_ids": set(),
                    "appearances": 0,
                    "best_rank": 9999,
                    "max_like_count": 0,
                },
            )
            record["appearances"] += 1
            record["albums"][clean(row.get("album"))] += 1
            record["genres"][clean(row.get("genre_name"))] += 1
            year = parse_int(row.get("year"))
            if year:
                record["years"].add(year)
            if row.get("song_id"):
                record["song_ids"].add(clean(row.get("song_id")))
            record["best_rank"] = min(record["best_rank"], parse_int(row.get("rank"), 9999))
            record["max_like_count"] = max(record["max_like_count"], parse_int(row.get("like_count")))
    return records


def chart_score(record: dict[str, Any]) -> float:
    best_rank = max(1, int(record.get("best_rank") or 9999))
    return (
        float(record.get("appearances") or 0) * 3.0
        + len(record.get("years") or []) * 2.0
        + min(12.0, float(record.get("max_like_count") or 0) / 8000.0)
        + max(0.0, 55.0 - best_rank) / 8.0
    )


def read_lyrics_for_keys(path: Path, target_keys: set[str]) -> dict[str, dict[str, Any]]:
    lyrics_by_key: dict[str, dict[str, Any]] = {}
    if not target_keys:
        return lyrics_by_key
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = clean(row.get("title"))
            artist = clean(row.get("artist"))
            if not title or not artist:
                continue
            key = norm_key(title, artist)
            if key not in target_keys or key in lyrics_by_key:
                continue
            raw_lyrics = str(row.get("lyrics") or "")
            processed = preprocess_lyrics(raw_lyrics)
            if not processed:
                continue
            keywords = top_keywords(processed, limit=10)
            emotion_tags = score_rules(processed, EMOTION_RULES, limit=4)
            topic_tags = score_rules(processed, TOPIC_RULES, limit=5)
            lyrics_by_key[key] = {
                "lyrics_hash": hashlib.blake2b(raw_lyrics.encode("utf-8"), digest_size=8).hexdigest(),
                "lyric_keywords": keywords,
                "emotion_tags": emotion_tags or ["평온"],
                "topic_tags": topic_tags or ["일상"],
                "lyric_summary": (
                    f"가사 요약: {phrase_for_tags(emotion_tags)} 중심의 정서가 두드러진다. "
                    f"핵심 키워드: {', '.join(keywords[:7]) or '키워드 부족'}."
                ),
            }
    return lyrics_by_key


def build_music_items(
    chart_path: Path,
    lyrics_path: Path,
    bugs_dir: Path,
    limit: int,
    candidate_multiplier: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = aggregate_melon_chart(chart_path)
    bugs_years = read_bugs_years(bugs_dir)
    for key, years in bugs_years.items():
        if key in records:
            records[key]["years"].update(years)
            records[key]["source_has_bugs"] = True
    ranked = sorted(records.items(), key=lambda item: chart_score(item[1]), reverse=True)
    target_keys = {key for key, _ in ranked[: max(limit * candidate_multiplier, limit)]}
    lyrics_by_key = read_lyrics_for_keys(lyrics_path, target_keys)

    items: list[dict[str, Any]] = []
    for key, record in ranked:
        lyric = lyrics_by_key.get(key)
        if not lyric:
            continue
        years = set(record.get("years") or [])
        genre = record["genres"].most_common(1)[0][0] if record["genres"] else "가요"
        album = record["albums"].most_common(1)[0][0] if record["albums"] else ""
        generation = generation_label(years)
        source_parts = ["melon_monthly_top50", "melon_song_lyrics"]
        if record.get("source_has_bugs"):
            source_parts.append("bugs_yearly_songs")
        metadata_note = (
            f"{generation} 차트 기반 곡이며, 최고 순위 {record['best_rank']}위, "
            f"차트 등장 {record['appearances']}회로 확인된다."
        )
        summary = (
            f"{record['artist']}의 음악 콘텐츠. {metadata_note} "
            f"{lyric['lyric_summary']} 장르/분류: {genre}."
        )
        topic_tags = list(dict.fromkeys(lyric["topic_tags"] + [genre, generation]))
        emotion_tags = list(dict.fromkeys(lyric["emotion_tags"]))
        tagging_text = "\n".join(
            [
                f"제목: {record['title']}",
                f"아티스트: {record['artist']}",
                "유형: music",
                f"장르: {genre}",
                f"앨범: {album}",
                f"요약: {summary}",
                f"감정 태그: {', '.join(emotion_tags)}",
                f"주제 태그: {', '.join(topic_tags)}",
                f"가사 키워드: {', '.join(lyric['lyric_keywords'])}",
                f"원천: {', '.join(source_parts)}",
            ]
        )
        items.append(
            {
                "content_id": stable_id("music_real", key),
                "content_type": "music",
                "title": record["title"],
                "creator": record["artist"],
                "genre": genre,
                "summary": summary,
                "source": "+".join(source_parts),
                "emotion_tags": emotion_tags,
                "topic_tags": topic_tags,
                "embedding_text": tagging_text,
                "raw_description": summary,
                "processed_description": summary,
                "tagging_text": tagging_text,
                "import_metadata": {
                    "album": album,
                    "years": sorted(years),
                    "generation": generation,
                    "best_rank": record["best_rank"],
                    "appearances": record["appearances"],
                    "max_like_count": record["max_like_count"],
                    "song_ids": sorted(record["song_ids"]),
                    "lyrics_hash": lyric["lyrics_hash"],
                },
            }
        )
        if len(items) >= limit:
            break

    stats = {
        "melon_chart_unique": len(records),
        "bugs_unique": len(bugs_years),
        "lyrics_matched": len(lyrics_by_key),
        "music_items": len(items),
    }
    return items, stats


def infer_book_genre(text: str) -> str:
    checks = [
        ("경제/자기계발", ["부자", "돈", "투자", "경제", "성공", "습관", "자기계발"]),
        ("소설/판타지", ["소설", "마법", "모험", "사랑", "이야기", "주인공", "세계"]),
        ("에세이/위로", ["에세이", "위로", "마음", "삶", "행복", "관계"]),
        ("인문/사회", ["철학", "역사", "사회", "인간", "문명", "정치"]),
        ("과학/지식", ["과학", "우주", "생명", "기술", "지식"]),
        ("학습/교육", ["공부", "학습", "시험", "영어", "교육"]),
        ("건강/생활", ["건강", "요리", "운동", "생활", "몸"]),
    ]
    lowered = text.lower()
    for genre, keywords in checks:
        if any(keyword in lowered for keyword in keywords):
            return genre
    return "베스트셀러"


def aggregate_kyobo(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = clean(row.get("title"))
            author = clean(row.get("author"))
            if not title or not author:
                continue
            key = norm_key(title, author)
            record = records.setdefault(
                key,
                {
                    "title": title,
                    "author": author,
                    "publisher": clean(row.get("publisher")),
                    "years": set(),
                    "best_rank": 9999,
                    "summary": "",
                    "review_keywords": Counter(),
                    "rating": 0.0,
                    "review_count": 0.0,
                    "detail_url": clean(row.get("detail_url")),
                },
            )
            year = parse_int(row.get("year"))
            if year:
                record["years"].add(year)
            record["best_rank"] = min(record["best_rank"], parse_int(row.get("rank"), 9999))
            summary = clean(row.get("summary"))
            if len(summary) > len(record["summary"]):
                record["summary"] = summary
            keyword = clean(row.get("review_keyword"))
            if keyword:
                record["review_keywords"][keyword] += 1
            record["rating"] = max(record["rating"], parse_float(row.get("rating")))
            record["review_count"] = max(record["review_count"], parse_float(row.get("review_count")))
            if not record["detail_url"]:
                record["detail_url"] = clean(row.get("detail_url"))
    return records


def book_score(record: dict[str, Any]) -> float:
    summary_bonus = min(10.0, len(record.get("summary") or "") / 120.0)
    best_rank = max(1, int(record.get("best_rank") or 9999))
    return (
        len(record.get("years") or []) * 4.0
        + max(0.0, 110.0 - best_rank) / 8.0
        + min(8.0, float(record.get("review_count") or 0) / 120.0)
        + float(record.get("rating") or 0) / 2.0
        + summary_bonus
    )


def build_book_items(path: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = aggregate_kyobo(path)
    ranked = sorted(records.items(), key=lambda item: book_score(item[1]), reverse=True)
    items: list[dict[str, Any]] = []
    for key, record in ranked:
        base_summary = record["summary"]
        review_keywords = [keyword for keyword, _ in record["review_keywords"].most_common(3)]
        descriptor_source = base_summary or (
            f"{record['title']}은 교보문고 연도별 베스트셀러에 오른 도서다. "
            f"리뷰 키워드: {', '.join(review_keywords) or '없음'}."
        )
        genre = infer_book_genre(" ".join([record["title"], descriptor_source, " ".join(review_keywords)]))
        emotion_tags = score_rules(descriptor_source, EMOTION_RULES, limit=4) or ["호기심"]
        topic_tags = score_rules(descriptor_source + " " + genre, TOPIC_RULES, limit=5) or [genre]
        keywords = top_keywords(descriptor_source, limit=8)
        years = set(record.get("years") or [])
        generation = generation_label(years)
        summary = (
            f"{record['author']}의 도서. {generation} 교보문고 연도별 베스트셀러에 포함되었고 "
            f"최고 순위 {record['best_rank']}위로 확인된다. "
            f"{descriptor_source} 핵심 키워드: {', '.join(keywords) or ', '.join(review_keywords) or '키워드 부족'}."
        )
        topic_tags = list(dict.fromkeys(topic_tags + [genre, generation]))
        tagging_text = "\n".join(
            [
                f"제목: {record['title']}",
                f"저자: {record['author']}",
                "유형: book",
                f"장르: {genre}",
                f"출판사: {record['publisher']}",
                f"요약: {summary}",
                f"감정 태그: {', '.join(emotion_tags)}",
                f"주제 태그: {', '.join(topic_tags)}",
                f"리뷰 키워드: {', '.join(review_keywords)}",
                "원천: kyobo_annual_bestseller",
            ]
        )
        items.append(
            {
                "content_id": stable_id("book_real", key),
                "content_type": "book",
                "title": record["title"],
                "creator": record["author"],
                "genre": genre,
                "summary": summary,
                "source": "kyobo_annual_bestseller",
                "emotion_tags": emotion_tags,
                "topic_tags": topic_tags,
                "embedding_text": tagging_text,
                "raw_description": summary,
                "processed_description": summary,
                "tagging_text": tagging_text,
                "import_metadata": {
                    "publisher": record["publisher"],
                    "years": sorted(years),
                    "generation": generation,
                    "best_rank": record["best_rank"],
                    "rating": record["rating"],
                    "review_count": record["review_count"],
                    "review_keywords": review_keywords,
                    "detail_url": record["detail_url"],
                },
            }
        )
        if len(items) >= limit:
            break
    stats = {
        "kyobo_unique": len(records),
        "book_items": len(items),
        "book_items_with_summary": sum(1 for item in items if len(item["summary"]) >= 80),
    }
    return items, stats


def write_processed(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            safe_item = dict(item)
            safe_item.pop("import_metadata", None)
            handle.write(json.dumps(safe_item, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import real music/book culture data into the MVP DB.")
    parser.add_argument("--music-limit", type=int, default=500)
    parser.add_argument("--book-limit", type=int, default=500)
    parser.add_argument("--skip-music", action="store_true")
    parser.add_argument("--skip-books", action="store_true")
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--processed-output", type=Path, default=PROCESSED_DIR / "culture_import_music_book.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    items: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    if not args.skip_music:
        music_items, music_stats = build_music_items(
            MELON_CHART_PATH,
            MELON_LYRICS_PATH,
            RAW_DIR / "bugs_연도별가요",
            max(0, args.music_limit),
        )
        items.extend(music_items)
        stats.update(music_stats)

    if not args.skip_books:
        book_items, book_stats = build_book_items(KYOBO_PATH, max(0, args.book_limit))
        items.extend(book_items)
        stats.update(book_stats)

    write_processed(items, args.processed_output)
    db_result = {"skipped": bool(args.skip_db or args.dry_run)}
    if not args.skip_db and not args.dry_run:
        database = Database()
        database.initialize()
        db_result = database.upsert_content_items(items, reset_tagging=True)

    report = {
        "music_limit": args.music_limit,
        "book_limit": args.book_limit,
        "processed_output": str(args.processed_output),
        "total_items": len(items),
        "content_type_counts": dict(Counter(item["content_type"] for item in items)),
        "stats": stats,
        "db": db_result,
        "raw_lyrics_persisted": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
