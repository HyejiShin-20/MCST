from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config
from app.services.text_utils import stable_hash


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(
        self,
        db_path: Path | str = config.DB_PATH,
        sample_content_path: Path | str = config.SAMPLE_CONTENT_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.sample_content_path = Path(sample_content_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        password_hash TEXT NOT NULL DEFAULT '',
                        is_guest INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS diary_entries (
                        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        input_type TEXT NOT NULL,
                        raw_text TEXT NOT NULL,
                        processed_text TEXT NOT NULL,
                        source_media_id INTEGER,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS entry_analyses (
                        entry_id INTEGER PRIMARY KEY,
                        analysis_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(entry_id) REFERENCES diary_entries(entry_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS media_transcriptions (
                        media_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        entry_id INTEGER,
                        input_type TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        source_filename TEXT NOT NULL DEFAULT '',
                        source_mime_type TEXT NOT NULL DEFAULT '',
                        source_size_bytes INTEGER NOT NULL DEFAULT 0,
                        file_sha256 TEXT NOT NULL DEFAULT '',
                        candidates_json TEXT NOT NULL,
                        selected_text TEXT NOT NULL DEFAULT '',
                        confidence REAL NOT NULL DEFAULT 0,
                        needs_user_review INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL DEFAULT 'candidate_ready',
                        error_type TEXT NOT NULL DEFAULT '',
                        error_message TEXT NOT NULL DEFAULT '',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(entry_id) REFERENCES diary_entries(entry_id) ON DELETE SET NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_media_transcriptions_user_id
                    ON media_transcriptions(user_id);

                    CREATE TABLE IF NOT EXISTS long_term_profiles (
                        user_id INTEGER PRIMARY KEY,
                        profile_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS content_items (
                        content_id TEXT PRIMARY KEY,
                        content_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        creator TEXT NOT NULL,
                        genre TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        source TEXT NOT NULL,
                        emotion_tags_json TEXT NOT NULL,
                        topic_tags_json TEXT NOT NULL,
                        embedding_text TEXT NOT NULL,
                        raw_description TEXT NOT NULL DEFAULT '',
                        processed_description TEXT NOT NULL DEFAULT '',
                        tagging_text TEXT NOT NULL DEFAULT '',
                        tag_status TEXT NOT NULL DEFAULT 'pending',
                        tag_version TEXT NOT NULL DEFAULT '',
                        tag_confidence REAL NOT NULL DEFAULT 0,
                        is_recommendable INTEGER NOT NULL DEFAULT 1,
                        error_type TEXT NOT NULL DEFAULT '',
                        error_message TEXT NOT NULL DEFAULT '',
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        last_attempt_at TEXT,
                        embedding_hash TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS content_tags (
                        content_id TEXT PRIMARY KEY,
                        tag_version TEXT NOT NULL,
                        tag_method TEXT NOT NULL,
                        emotion_tags_json TEXT NOT NULL,
                        theme_tags_json TEXT NOT NULL,
                        mood_tags_json TEXT NOT NULL,
                        recommendation_roles_json TEXT NOT NULL,
                        valence REAL NOT NULL,
                        arousal REAL NOT NULL,
                        intensity REAL NOT NULL,
                        energy REAL NOT NULL,
                        cognitive_load REAL NOT NULL,
                        tag_confidence REAL NOT NULL,
                        raw_tag_result_json TEXT NOT NULL,
                        final_tag_result_json TEXT NOT NULL,
                        dark_flag INTEGER NOT NULL DEFAULT 0,
                        too_heavy_flag INTEGER NOT NULL DEFAULT 0,
                        background_friendly INTEGER NOT NULL DEFAULT 0,
                        manual_override INTEGER NOT NULL DEFAULT 0,
                        manual_emotion_tags_json TEXT NOT NULL DEFAULT '[]',
                        manual_theme_tags_json TEXT NOT NULL DEFAULT '[]',
                        manual_mood_tags_json TEXT NOT NULL DEFAULT '[]',
                        manual_recommendation_roles_json TEXT NOT NULL DEFAULT '[]',
                        manual_note TEXT NOT NULL DEFAULT '',
                        reviewed_by TEXT NOT NULL DEFAULT '',
                        reviewed_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(content_id) REFERENCES content_items(content_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS content_embeddings (
                        embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_id TEXT NOT NULL,
                        embedding_type TEXT NOT NULL,
                        embedding_model_name TEXT NOT NULL,
                        embedding_hash TEXT NOT NULL,
                        vector_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(content_id, embedding_type, embedding_model_name, embedding_hash),
                        FOREIGN KEY(content_id) REFERENCES content_items(content_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS tag_runs (
                        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tag_version TEXT NOT NULL,
                        tag_method TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        batch_size INTEGER NOT NULL,
                        target_count INTEGER NOT NULL,
                        success_count INTEGER NOT NULL,
                        failed_count INTEGER NOT NULL,
                        skipped_count INTEGER NOT NULL,
                        average_confidence REAL NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        run_options_json TEXT NOT NULL,
                        error_summary_json TEXT NOT NULL,
                        log_path TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS recommendation_logs (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        entry_id INTEGER NOT NULL,
                        response_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(entry_id) REFERENCES diary_entries(entry_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS saved_items (
                        saved_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        content_id TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        creator TEXT NOT NULL DEFAULT '',
                        genre TEXT NOT NULL DEFAULT '',
                        source TEXT NOT NULL DEFAULT '',
                        entry_id INTEGER,
                        log_id INTEGER,
                        recommendation_role TEXT NOT NULL DEFAULT '',
                        reason TEXT NOT NULL DEFAULT '',
                        score_percent INTEGER NOT NULL DEFAULT 0,
                        display_tags_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'planned',
                        note TEXT NOT NULL DEFAULT '',
                        snapshot_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(user_id, content_id),
                        FOREIGN KEY(entry_id) REFERENCES diary_entries(entry_id) ON DELETE SET NULL,
                        FOREIGN KEY(log_id) REFERENCES recommendation_logs(log_id) ON DELETE SET NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_saved_items_user_id
                    ON saved_items(user_id);

                    CREATE TABLE IF NOT EXISTS feedback_events (
                        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        entry_id INTEGER,
                        log_id INTEGER,
                        content_id TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        rating REAL,
                        note TEXT NOT NULL DEFAULT '',
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(entry_id) REFERENCES diary_entries(entry_id) ON DELETE CASCADE,
                        FOREIGN KEY(log_id) REFERENCES recommendation_logs(log_id) ON DELETE SET NULL,
                        FOREIGN KEY(content_id) REFERENCES content_items(content_id)
                    );
                    """
                )
                self._ensure_diary_entry_columns(connection)
                self._ensure_content_item_columns(connection)
                self._ensure_user_columns(connection)
                self._ensure_guest_user(connection)
        self.seed_content_if_empty()
        self.backfill_content_tagging_defaults()

    @staticmethod
    def _ensure_user_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        if "password_hash" not in existing:
            connection.execute(
                "ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''"
            )
        if "is_guest" not in existing:
            connection.execute(
                "ALTER TABLE users ADD COLUMN is_guest INTEGER NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _ensure_guest_user(connection: sqlite3.Connection) -> None:
        # 체험(게스트) 전용 계정. 비밀번호 없이 '둘러보기' 버튼으로만 로그인된다.
        # 데모 기본 진입을 없애기 위해 더 이상 'demo' 계정은 시드하지 않는다.
        now = now_iso()
        connection.execute(
            """
            INSERT OR IGNORE INTO users (
                user_id, username, display_name, password_hash, is_guest,
                created_at, last_seen_at
            ) VALUES (?, ?, ?, '', 1, ?, ?)
            """,
            (config.DEFAULT_USER_ID, "guest", "체험 계정", now, now),
        )

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def login_or_create_user(self, username: str, display_name: str = "") -> dict[str, Any]:
        username = username.strip()
        display_name = display_name.strip() or username
        now = now_iso()
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO users (username, display_name, created_at, last_seen_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(username) DO UPDATE SET
                        display_name = excluded.display_name,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (username, display_name, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
        return dict(row)

    def create_user(
        self, username: str, display_name: str, password_hash: str
    ) -> dict[str, Any] | None:
        """새 사용자를 생성한다. 아이디가 이미 있으면 None을 반환한다."""
        username = username.strip()
        display_name = display_name.strip() or username
        now = now_iso()
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO users (
                        username, display_name, password_hash, is_guest,
                        created_at, last_seen_at
                    ) VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (username, display_name, password_hash, now, now),
                )
                if cursor.rowcount == 0:
                    return None
                row = connection.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
        return dict(row)

    def ensure_guest_user(self) -> dict[str, Any]:
        """체험(게스트) 계정을 보장하고 반환한다."""
        now = now_iso()
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO users (username, display_name, password_hash, is_guest, created_at, last_seen_at)
                    VALUES ('guest', '체험 계정', '', 1, ?, ?)
                    ON CONFLICT(username) DO UPDATE SET last_seen_at = excluded.last_seen_at
                    """,
                    (now, now),
                )
                row = connection.execute(
                    "SELECT * FROM users WHERE username = 'guest'",
                ).fetchone()
        return dict(row)

    def delete_user(self, user_id: int) -> bool:
        """사용자 데이터와 계정을 모두 삭제한다."""
        self.delete_user_data(user_id)
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM users WHERE user_id = ?", (user_id,)
                )
                return cursor.rowcount > 0

    def touch_user(self, user_id: int) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    "UPDATE users SET last_seen_at = ? WHERE user_id = ?",
                    (now_iso(), user_id),
                )

    @staticmethod
    def _ensure_diary_entry_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(diary_entries)").fetchall()
        }
        if "source_media_id" not in existing:
            connection.execute(
                "ALTER TABLE diary_entries ADD COLUMN source_media_id INTEGER"
            )

    @staticmethod
    def _ensure_content_item_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(content_items)").fetchall()
        }
        columns = {
            "raw_description": "TEXT NOT NULL DEFAULT ''",
            "processed_description": "TEXT NOT NULL DEFAULT ''",
            "tagging_text": "TEXT NOT NULL DEFAULT ''",
            "tag_status": "TEXT NOT NULL DEFAULT 'pending'",
            "tag_version": "TEXT NOT NULL DEFAULT ''",
            "tag_confidence": "REAL NOT NULL DEFAULT 0",
            "is_recommendable": "INTEGER NOT NULL DEFAULT 1",
            "error_type": "TEXT NOT NULL DEFAULT ''",
            "error_message": "TEXT NOT NULL DEFAULT ''",
            "retry_count": "INTEGER NOT NULL DEFAULT 0",
            "last_attempt_at": "TEXT",
            "embedding_hash": "TEXT NOT NULL DEFAULT ''",
        }
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(
                    f"ALTER TABLE content_items ADD COLUMN {name} {definition}"
                )

    def seed_content_if_empty(self) -> None:
        with closing(self.connect()) as connection:
            with connection:
                raw_seed = self.sample_content_path.read_text(encoding="utf-8")
                if self.sample_content_path.suffix == ".jsonl":
                    # 실데이터 카탈로그(JSONL: 한 줄당 1개 객체) 지원
                    items = [
                        json.loads(line)
                        for line in raw_seed.splitlines()
                        if line.strip()
                    ]
                else:
                    items = json.loads(raw_seed)
                for item in items:
                    embedding_text = self.embedding_text(item)
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO content_items (
                            content_id, content_type, title, creator, genre, summary, source,
                            emotion_tags_json, topic_tags_json, embedding_text,
                            raw_description, processed_description, tagging_text,
                            tag_status, tag_version, tag_confidence, is_recommendable,
                            embedding_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["content_id"],
                            item["content_type"],
                            item["title"],
                            item["creator"],
                            item["genre"],
                            item["summary"],
                            item["source"],
                            json.dumps(item["emotion_tags"], ensure_ascii=False),
                            json.dumps(item["topic_tags"], ensure_ascii=False),
                            embedding_text,
                            item["summary"],
                            item["summary"],
                            embedding_text,
                            "tagged",
                            config.CURRENT_TAG_VERSION,
                            0.82,
                            1,
                            str(stable_hash(embedding_text)),
                        ),
                    )
                    if cursor.rowcount:
                        self._upsert_seed_content_tags(connection, item, embedding_text)

    @staticmethod
    def _upsert_seed_content_tags(
        connection: sqlite3.Connection, item: dict[str, Any], tagging_text: str
    ) -> None:
        now = now_iso()
        final_result = {
            "emotion_tags": {tag: 0.8 for tag in item["emotion_tags"]},
            "theme_tags": {tag: 0.8 for tag in item["topic_tags"]},
            "mood_tags": {},
            "recommendation_roles": {},
            "tagging_text_preview": tagging_text[:240],
        }
        connection.execute(
            """
            INSERT OR REPLACE INTO content_tags (
                content_id, tag_version, tag_method,
                emotion_tags_json, theme_tags_json, mood_tags_json, recommendation_roles_json,
                valence, arousal, intensity, energy, cognitive_load, tag_confidence,
                raw_tag_result_json, final_tag_result_json,
                dark_flag, too_heavy_flag, background_friendly,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["content_id"],
                config.CURRENT_TAG_VERSION,
                "seed_sample",
                json.dumps(item["emotion_tags"], ensure_ascii=False),
                json.dumps(item["topic_tags"], ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                json.dumps([], ensure_ascii=False),
                0.0,
                0.35,
                0.4,
                0.45,
                0.45,
                0.82,
                json.dumps(final_result, ensure_ascii=False),
                json.dumps(final_result, ensure_ascii=False),
                int(bool(set(item["emotion_tags"]) & {"우울", "절망", "비극", "고립"})),
                int(bool(set(item["emotion_tags"]) & {"비극", "분노"})),
                int(bool(set(item["emotion_tags"]) & {"평온", "위로", "따뜻함"})),
                now,
                now,
            ),
        )

    @staticmethod
    def embedding_text(item: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"제목: {item['title']}",
                f"유형: {item['content_type']}",
                f"창작자: {item['creator']}",
                f"장르: {item['genre']}",
                f"요약: {item['summary']}",
                f"감정 태그: {', '.join(item['emotion_tags'])}",
                f"주제 태그: {', '.join(item['topic_tags'])}",
            ]
        )

    def backfill_content_tagging_defaults(self) -> None:
        with closing(self.connect()) as connection:
            with connection:
                rows = connection.execute("SELECT * FROM content_items").fetchall()
                for row in rows:
                    data = dict(row)
                    emotion_tags = json.loads(data["emotion_tags_json"])
                    topic_tags = json.loads(data["topic_tags_json"])
                    tagging_text = data.get("tagging_text") or data["embedding_text"]
                    current_status = data.get("tag_status") or "tagged"
                    pending_statuses = {
                        "pending",
                        "processing",
                        "failed",
                        "tag_failed",
                        "embedding_failed",
                        "validation_needed",
                    }
                    is_pending = current_status in pending_statuses and not data.get("tag_version")
                    tag_version = "" if is_pending else (data.get("tag_version") or config.CURRENT_TAG_VERSION)
                    tag_status = current_status if is_pending else current_status
                    tag_confidence = float(
                        data.get("tag_confidence")
                        or (0.0 if is_pending else 0.82)
                    )
                    connection.execute(
                        """
                        UPDATE content_items
                        SET raw_description = COALESCE(NULLIF(raw_description, ''), summary),
                            processed_description = COALESCE(NULLIF(processed_description, ''), summary),
                            tagging_text = COALESCE(NULLIF(tagging_text, ''), embedding_text),
                            tag_status = ?,
                            tag_version = ?,
                            tag_confidence = ?,
                            is_recommendable = COALESCE(is_recommendable, 1),
                            embedding_hash = COALESCE(NULLIF(embedding_hash, ''), ?)
                        WHERE content_id = ?
                        """,
                        (
                            tag_status,
                            tag_version,
                            tag_confidence,
                            str(stable_hash(tagging_text)),
                            data["content_id"],
                        ),
                    )
                    existing = connection.execute(
                        "SELECT 1 FROM content_tags WHERE content_id = ?",
                        (data["content_id"],),
                    ).fetchone()
                    if not existing and not is_pending:
                        self._upsert_seed_content_tags(
                            connection,
                            {
                                "content_id": data["content_id"],
                                "emotion_tags": emotion_tags,
                                "topic_tags": topic_tags,
                            },
                            tagging_text,
                        )

    def upsert_content_items(
        self,
        items: list[dict[str, Any]],
        *,
        reset_tagging: bool = True,
    ) -> dict[str, int]:
        if not items:
            return {"upserted": 0}
        now = now_iso()
        upserted = 0
        with closing(self.connect()) as connection:
            with connection:
                for item in items:
                    emotion_tags = item.get("emotion_tags") or []
                    topic_tags = item.get("topic_tags") or []
                    embedding_text = item.get("embedding_text") or self.embedding_text(
                        {
                            "title": item["title"],
                            "content_type": item["content_type"],
                            "creator": item.get("creator", ""),
                            "genre": item.get("genre", ""),
                            "summary": item.get("summary", ""),
                            "emotion_tags": emotion_tags,
                            "topic_tags": topic_tags,
                        }
                    )
                    raw_description = item.get("raw_description") or item.get("summary", "")
                    processed_description = item.get("processed_description") or raw_description
                    tagging_text = item.get("tagging_text") or embedding_text
                    tag_status = "pending" if reset_tagging else item.get("tag_status", "pending")
                    connection.execute(
                        """
                        INSERT INTO content_items (
                            content_id, content_type, title, creator, genre, summary, source,
                            emotion_tags_json, topic_tags_json, embedding_text,
                            raw_description, processed_description, tagging_text,
                            tag_status, tag_version, tag_confidence, is_recommendable,
                            error_type, error_message, retry_count, last_attempt_at,
                            embedding_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', 0, ?, ?)
                        ON CONFLICT(content_id) DO UPDATE SET
                            content_type = excluded.content_type,
                            title = excluded.title,
                            creator = excluded.creator,
                            genre = excluded.genre,
                            summary = excluded.summary,
                            source = excluded.source,
                            emotion_tags_json = excluded.emotion_tags_json,
                            topic_tags_json = excluded.topic_tags_json,
                            embedding_text = excluded.embedding_text,
                            raw_description = excluded.raw_description,
                            processed_description = excluded.processed_description,
                            tagging_text = excluded.tagging_text,
                            tag_status = excluded.tag_status,
                            tag_version = excluded.tag_version,
                            tag_confidence = excluded.tag_confidence,
                            is_recommendable = excluded.is_recommendable,
                            error_type = '',
                            error_message = '',
                            retry_count = 0,
                            last_attempt_at = excluded.last_attempt_at,
                            embedding_hash = excluded.embedding_hash
                        """,
                        (
                            item["content_id"],
                            item["content_type"],
                            item["title"],
                            item.get("creator", ""),
                            item.get("genre", ""),
                            item.get("summary", ""),
                            item.get("source", ""),
                            json.dumps(emotion_tags, ensure_ascii=False),
                            json.dumps(topic_tags, ensure_ascii=False),
                            embedding_text,
                            raw_description,
                            processed_description,
                            tagging_text,
                            tag_status,
                            "" if reset_tagging else item.get("tag_version", ""),
                            0.0 if reset_tagging else float(item.get("tag_confidence", 0.0)),
                            int(item.get("is_recommendable", 1)),
                            now,
                            "" if reset_tagging else item.get("embedding_hash", ""),
                        ),
                    )
                    upserted += 1
        return {"upserted": upserted}

    def add_diary_entry(
        self,
        user_id: int,
        input_type: str,
        raw_text: str,
        processed_text: str | None = None,
        source_media_id: int | None = None,
    ) -> int:
        processed = processed_text if processed_text is not None else raw_text
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO diary_entries (
                        user_id, input_type, raw_text, processed_text,
                        source_media_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, input_type, raw_text, processed, source_media_id, now_iso()),
                )
                return int(cursor.lastrowid)

    def save_media_transcription(
        self,
        *,
        user_id: int,
        input_type: str,
        provider: str,
        model_name: str,
        source_filename: str,
        source_mime_type: str,
        source_size_bytes: int,
        file_sha256: str,
        candidates: list[dict[str, Any]],
        selected_text: str,
        confidence: float,
        needs_user_review: bool,
        status: str,
        error_type: str = "",
        error_message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = now_iso()
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO media_transcriptions (
                        user_id, input_type, provider, model_name,
                        source_filename, source_mime_type, source_size_bytes,
                        file_sha256, candidates_json, selected_text, confidence,
                        needs_user_review, status, error_type, error_message,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        input_type,
                        provider,
                        model_name,
                        source_filename,
                        source_mime_type,
                        source_size_bytes,
                        file_sha256,
                        json.dumps(candidates, ensure_ascii=False),
                        selected_text,
                        confidence,
                        int(needs_user_review),
                        status,
                        error_type,
                        error_message[:600],
                        json.dumps(metadata or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                return int(cursor.lastrowid)

    def get_media_transcription(self, media_id: int) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM media_transcriptions WHERE media_id = ?",
                (media_id,),
            ).fetchone()
        return self._media_transcription_row(row) if row else None

    def confirm_media_transcription(
        self,
        media_id: int,
        selected_text: str,
        entry_id: int | None = None,
    ) -> bool:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE media_transcriptions
                    SET selected_text = ?,
                        entry_id = COALESCE(?, entry_id),
                        status = 'confirmed',
                        needs_user_review = 0,
                        updated_at = ?
                    WHERE media_id = ?
                    """,
                    (selected_text, entry_id, now_iso(), media_id),
                )
                return cursor.rowcount > 0

    def delete_media_transcription(self, media_id: int) -> bool:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM media_transcriptions WHERE media_id = ?",
                    (media_id,),
                )
                return cursor.rowcount > 0

    def save_analysis(self, entry_id: int, analysis: dict[str, Any]) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO entry_analyses (entry_id, analysis_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (entry_id, json.dumps(analysis, ensure_ascii=False), now_iso()),
                )

    def get_entry(self, entry_id: int) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM diary_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_analysis(self, entry_id: int) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT analysis_json FROM entry_analyses WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        return json.loads(row["analysis_json"]) if row else None

    def list_entries(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT e.*, a.analysis_json
                FROM diary_entries e
                LEFT JOIN entry_analyses a ON e.entry_id = a.entry_id
                WHERE e.user_id = ?
                ORDER BY e.entry_id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._entry_row(row) for row in rows]

    def list_recent_analyses(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT a.analysis_json
                FROM diary_entries e
                JOIN entry_analyses a ON e.entry_id = a.entry_id
                WHERE e.user_id = ?
                ORDER BY e.entry_id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [json.loads(row["analysis_json"]) for row in rows]

    def upsert_profile(self, user_id: int, profile: dict[str, Any]) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO long_term_profiles (user_id, profile_json, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, json.dumps(profile, ensure_ascii=False), now_iso()),
                )

    def get_profile(self, user_id: int) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT profile_json FROM long_term_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        return json.loads(row["profile_json"]) if row else None

    def list_contents(
        self,
        content_types: list[str] | None = None,
        recommendable_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                c.*,
                t.emotion_tags_json AS tagged_emotion_tags_json,
                t.theme_tags_json AS tagged_theme_tags_json,
                t.mood_tags_json AS mood_tags_json,
                t.recommendation_roles_json AS recommendation_roles_json,
                t.valence, t.arousal, t.intensity, t.energy, t.cognitive_load,
                t.raw_tag_result_json, t.final_tag_result_json,
                t.dark_flag, t.too_heavy_flag, t.background_friendly,
                t.manual_override,
                t.manual_emotion_tags_json, t.manual_theme_tags_json,
                t.manual_mood_tags_json, t.manual_recommendation_roles_json
            FROM content_items c
            LEFT JOIN content_tags t ON c.content_id = t.content_id
        """
        conditions: list[str] = []
        params: list[Any] = []
        if content_types:
            placeholders = ",".join("?" for _ in content_types)
            conditions.append(f"c.content_type IN ({placeholders})")
            params.extend(content_types)
        if recommendable_only:
            conditions.append("c.tag_status = 'tagged'")
            conditions.append("c.is_recommendable = 1")
            conditions.append("c.tag_confidence >= ?")
            params.append(config.MIN_RECOMMEND_CONFIDENCE)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.content_type, c.title"
        with closing(self.connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._content_row(row) for row in rows]

    def get_content(self, content_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    c.*,
                    t.emotion_tags_json AS tagged_emotion_tags_json,
                    t.theme_tags_json AS tagged_theme_tags_json,
                    t.mood_tags_json AS mood_tags_json,
                    t.recommendation_roles_json AS recommendation_roles_json,
                    t.valence, t.arousal, t.intensity, t.energy, t.cognitive_load,
                    t.raw_tag_result_json, t.final_tag_result_json,
                    t.dark_flag, t.too_heavy_flag, t.background_friendly,
                    t.manual_override,
                    t.manual_emotion_tags_json, t.manual_theme_tags_json,
                    t.manual_mood_tags_json, t.manual_recommendation_roles_json
                FROM content_items c
                LEFT JOIN content_tags t ON c.content_id = t.content_id
                WHERE c.content_id = ?
                """,
                (content_id,),
            ).fetchone()
        return self._content_row(row) if row else None

    def save_recommendation_log(
        self, user_id: int, entry_id: int, response: dict[str, Any]
    ) -> int:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO recommendation_logs (user_id, entry_id, response_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, entry_id, json.dumps(response, ensure_ascii=False), now_iso()),
                )
                return int(cursor.lastrowid)

    def list_recommendation_logs(
        self, user_id: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    r.*,
                    e.input_type,
                    e.raw_text,
                    e.processed_text,
                    e.source_media_id
                FROM recommendation_logs r
                JOIN diary_entries e ON e.entry_id = r.entry_id
                WHERE r.user_id = ?
                ORDER BY r.log_id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._recommendation_log_row(row) for row in rows]

    def list_media_transcriptions(
        self, user_id: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM media_transcriptions
                WHERE user_id = ?
                ORDER BY media_id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._media_transcription_row(row) for row in rows]

    def get_saved_item_by_content(
        self, user_id: int, content_id: str
    ) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM saved_items
                WHERE user_id = ? AND content_id = ?
                """,
                (user_id, content_id),
            ).fetchone()
        return self._saved_item_row(row) if row else None

    def save_collection_item(
        self,
        *,
        user_id: int,
        content_id: str,
        content_type: str,
        title: str,
        creator: str = "",
        genre: str = "",
        source: str = "",
        entry_id: int | None = None,
        log_id: int | None = None,
        recommendation_role: str = "",
        reason: str = "",
        score_percent: int = 0,
        display_tags: list[str] | None = None,
        status: str = "planned",
        note: str = "",
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = now_iso()
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO saved_items (
                        user_id, content_id, content_type, title, creator,
                        genre, source, entry_id, log_id, recommendation_role,
                        reason, score_percent, display_tags_json, status, note,
                        snapshot_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, content_id) DO UPDATE SET
                        content_type = excluded.content_type,
                        title = excluded.title,
                        creator = excluded.creator,
                        genre = excluded.genre,
                        source = excluded.source,
                        entry_id = COALESCE(excluded.entry_id, saved_items.entry_id),
                        log_id = COALESCE(excluded.log_id, saved_items.log_id),
                        recommendation_role = excluded.recommendation_role,
                        reason = excluded.reason,
                        score_percent = excluded.score_percent,
                        display_tags_json = excluded.display_tags_json,
                        status = excluded.status,
                        note = excluded.note,
                        snapshot_json = excluded.snapshot_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        content_id,
                        content_type,
                        title,
                        creator,
                        genre,
                        source,
                        entry_id,
                        log_id,
                        recommendation_role,
                        reason,
                        int(score_percent or 0),
                        json.dumps(display_tags or [], ensure_ascii=False),
                        status,
                        note,
                        json.dumps(snapshot or {}, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT *
                    FROM saved_items
                    WHERE user_id = ? AND content_id = ?
                    """,
                    (user_id, content_id),
                ).fetchone()
        return self._saved_item_row(row)

    def list_saved_items(
        self,
        user_id: int,
        *,
        content_type: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions = ["user_id = ?"]
        params: list[Any] = [user_id]
        if content_type and content_type != "all":
            conditions.append("content_type = ?")
            params.append(content_type)
        if status and status != "all":
            conditions.append("status = ?")
            params.append(status)
        params.append(limit)
        with closing(self.connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM saved_items
                WHERE {' AND '.join(conditions)}
                ORDER BY updated_at DESC, saved_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._saved_item_row(row) for row in rows]

    def update_saved_item(
        self,
        *,
        saved_id: int,
        user_id: int,
        status: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if note is not None:
            updates.append("note = ?")
            params.append(note)
        if not updates:
            with closing(self.connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM saved_items WHERE saved_id = ? AND user_id = ?",
                    (saved_id, user_id),
                ).fetchone()
            return self._saved_item_row(row) if row else None

        updates.append("updated_at = ?")
        params.append(now_iso())
        params.extend([saved_id, user_id])
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    f"""
                    UPDATE saved_items
                    SET {', '.join(updates)}
                    WHERE saved_id = ? AND user_id = ?
                    """,
                    params,
                )
                row = connection.execute(
                    "SELECT * FROM saved_items WHERE saved_id = ? AND user_id = ?",
                    (saved_id, user_id),
                ).fetchone()
        return self._saved_item_row(row) if row else None

    def delete_saved_item(self, saved_id: int, user_id: int) -> bool:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM saved_items WHERE saved_id = ? AND user_id = ?",
                    (saved_id, user_id),
                )
                return cursor.rowcount > 0

    def list_tagging_targets(
        self,
        *,
        content_type: str | None = None,
        limit: int = 100,
        retag: bool = False,
        only_low_confidence: bool = False,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if content_type and content_type != "all":
            conditions.append("c.content_type = ?")
            params.append(content_type)
        if only_low_confidence:
            conditions.append("c.tag_confidence < ?")
            params.append(config.HIGH_CONFIDENCE_THRESHOLD)
        elif retag:
            conditions.append("1 = 1")
        else:
            conditions.append("c.tag_status IN ('pending', 'failed', 'tag_failed', 'embedding_failed')")
            conditions.append("c.retry_count < 3")
        query = "SELECT c.* FROM content_items c"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY c.content_type, c.content_id LIMIT ?"
        params.append(limit)
        with closing(self.connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def mark_content_processing(self, content_id: str) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE content_items
                    SET tag_status = 'processing',
                        error_type = '',
                        error_message = '',
                        last_attempt_at = ?
                    WHERE content_id = ?
                    """,
                    (now_iso(), content_id),
                )

    def save_content_tagging_result(
        self,
        *,
        content_id: str,
        tag_result: dict[str, Any],
        embedding_hash: str,
        embedding_model_name: str,
    ) -> None:
        now = now_iso()
        final = tag_result["final"]
        raw = tag_result["raw"]
        status = "tagged"
        if final["tag_confidence"] < config.LOW_CONFIDENCE_THRESHOLD:
            status = "validation_needed"
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE content_items
                    SET processed_description = ?,
                        tagging_text = ?,
                        embedding_text = ?,
                        tag_status = ?,
                        tag_version = ?,
                        tag_confidence = ?,
                        is_recommendable = ?,
                        error_type = '',
                        error_message = '',
                        embedding_hash = ?,
                        last_attempt_at = ?
                    WHERE content_id = ?
                    """,
                    (
                        tag_result["processed_text"],
                        tag_result["tagging_text"],
                        tag_result["tagging_text"],
                        status,
                        config.CURRENT_TAG_VERSION,
                        final["tag_confidence"],
                        int(final["is_recommendable"]),
                        embedding_hash,
                        now,
                        content_id,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO content_tags (
                        content_id, tag_version, tag_method,
                        emotion_tags_json, theme_tags_json, mood_tags_json,
                        recommendation_roles_json, valence, arousal, intensity,
                        energy, cognitive_load, tag_confidence,
                        raw_tag_result_json, final_tag_result_json,
                        dark_flag, too_heavy_flag, background_friendly,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_id,
                        config.CURRENT_TAG_VERSION,
                        tag_result["tag_method"],
                        json.dumps(final["emotion_tags"], ensure_ascii=False),
                        json.dumps(final["theme_tags"], ensure_ascii=False),
                        json.dumps(final["mood_tags"], ensure_ascii=False),
                        json.dumps(final["recommendation_roles"], ensure_ascii=False),
                        final["valence"],
                        final["arousal"],
                        final["intensity"],
                        final["energy"],
                        final["cognitive_load"],
                        final["tag_confidence"],
                        json.dumps(raw, ensure_ascii=False),
                        json.dumps(final, ensure_ascii=False),
                        int(final["dark_flag"]),
                        int(final["too_heavy_flag"]),
                        int(final["background_friendly"]),
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO content_embeddings (
                        content_id, embedding_type, embedding_model_name,
                        embedding_hash, vector_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_id,
                        "content",
                        embedding_model_name,
                        embedding_hash,
                        f"{content_id}:{embedding_hash}",
                        now,
                        now,
                    ),
                )

    def mark_content_tagging_failed(
        self, content_id: str, error_type: str, error_message: str
    ) -> None:
        status = "embedding_failed" if error_type == "EMBEDDING_FAILED" else "tag_failed"
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE content_items
                    SET tag_status = ?,
                        error_type = ?,
                        error_message = ?,
                        retry_count = retry_count + 1,
                        last_attempt_at = ?
                    WHERE content_id = ?
                    """,
                    (
                        status,
                        error_type,
                        error_message[:600],
                        now_iso(),
                        content_id,
                    ),
                )

    def record_tag_run(self, run: dict[str, Any]) -> int:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO tag_runs (
                        tag_version, tag_method, content_type, batch_size,
                        target_count, success_count, failed_count, skipped_count,
                        average_confidence, started_at, finished_at,
                        duration_seconds, run_options_json, error_summary_json, log_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run["tag_version"],
                        run["tag_method"],
                        run["content_type"],
                        run["batch_size"],
                        run["target_count"],
                        run["success_count"],
                        run["failed_count"],
                        run["skipped_count"],
                        run["average_confidence"],
                        run["started_at"],
                        run["finished_at"],
                        run["duration_seconds"],
                        json.dumps(run.get("run_options", {}), ensure_ascii=False),
                        json.dumps(run.get("error_summary", {}), ensure_ascii=False),
                        run.get("log_path", ""),
                    ),
                )
                return int(cursor.lastrowid)

    def apply_manual_tag_override(
        self,
        *,
        content_id: str,
        emotion_tags: list[str] | None = None,
        theme_tags: list[str] | None = None,
        mood_tags: list[str] | None = None,
        recommendation_roles: list[str] | None = None,
        note: str = "",
        reviewed_by: str = "manual",
    ) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE content_tags
                    SET manual_override = 1,
                        manual_emotion_tags_json = COALESCE(?, manual_emotion_tags_json),
                        manual_theme_tags_json = COALESCE(?, manual_theme_tags_json),
                        manual_mood_tags_json = COALESCE(?, manual_mood_tags_json),
                        manual_recommendation_roles_json = COALESCE(?, manual_recommendation_roles_json),
                        manual_note = ?,
                        reviewed_by = ?,
                        reviewed_at = ?,
                        updated_at = ?
                    WHERE content_id = ?
                    """,
                    (
                        json.dumps(emotion_tags, ensure_ascii=False) if emotion_tags is not None else None,
                        json.dumps(theme_tags, ensure_ascii=False) if theme_tags is not None else None,
                        json.dumps(mood_tags, ensure_ascii=False) if mood_tags is not None else None,
                        json.dumps(recommendation_roles, ensure_ascii=False)
                        if recommendation_roles is not None
                        else None,
                        note,
                        reviewed_by,
                        now_iso(),
                        now_iso(),
                        content_id,
                    ),
                )

    def save_feedback(
        self,
        *,
        user_id: int,
        entry_id: int | None,
        log_id: int | None,
        content_id: str,
        content_type: str,
        signal: str,
        rating: float | None,
        note: str,
        metadata: dict[str, Any],
    ) -> int:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO feedback_events (
                        user_id, entry_id, log_id, content_id, content_type,
                        signal, rating, note, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        entry_id,
                        log_id,
                        content_id,
                        content_type,
                        signal,
                        rating,
                        note,
                        json.dumps(metadata, ensure_ascii=False),
                        now_iso(),
                    ),
                )
                return int(cursor.lastrowid)

    def list_feedback_events(self, user_id: int, limit: int = 100) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM feedback_events
                WHERE user_id = ?
                ORDER BY feedback_id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._feedback_row(row) for row in rows]

    def delete_entry(self, entry_id: int) -> bool:
        with closing(self.connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM diary_entries WHERE entry_id = ?", (entry_id,)
                )
                return cursor.rowcount > 0

    def reset_memory(self, user_id: int) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM long_term_profiles WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM feedback_events WHERE user_id = ?", (user_id,)
                )

    def delete_user_data(self, user_id: int) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM saved_items WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM media_transcriptions WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM feedback_events WHERE user_id = ?", (user_id,)
                )
                connection.execute(
                    "DELETE FROM recommendation_logs WHERE user_id = ?", (user_id,)
                )
                connection.execute("DELETE FROM diary_entries WHERE user_id = ?", (user_id,))
                connection.execute(
                    "DELETE FROM long_term_profiles WHERE user_id = ?", (user_id,)
                )

    @staticmethod
    def _entry_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        analysis_json = data.pop("analysis_json", None)
        data["analysis"] = json.loads(analysis_json) if analysis_json else None
        return data

    @staticmethod
    def _content_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        base_emotion_tags = json.loads(data.pop("emotion_tags_json"))
        base_topic_tags = json.loads(data.pop("topic_tags_json"))
        tagged_emotion = json.loads(data.pop("tagged_emotion_tags_json") or "[]")
        tagged_theme = json.loads(data.pop("tagged_theme_tags_json") or "[]")
        mood_tags = json.loads(data.pop("mood_tags_json") or "[]")
        roles = json.loads(data.pop("recommendation_roles_json") or "[]")
        manual_override = bool(data.get("manual_override") or 0)
        manual_emotion = json.loads(data.pop("manual_emotion_tags_json") or "[]")
        manual_theme = json.loads(data.pop("manual_theme_tags_json") or "[]")
        manual_mood = json.loads(data.pop("manual_mood_tags_json") or "[]")
        manual_roles = json.loads(data.pop("manual_recommendation_roles_json") or "[]")
        data["emotion_tags"] = manual_emotion if manual_override and manual_emotion else (tagged_emotion or base_emotion_tags)
        data["topic_tags"] = manual_theme if manual_override and manual_theme else (tagged_theme or base_topic_tags)
        data["mood_tags"] = manual_mood if manual_override and manual_mood else mood_tags
        data["recommendation_roles"] = manual_roles if manual_override and manual_roles else roles
        data["tag_metadata"] = {
            "valence": data.pop("valence", None),
            "arousal": data.pop("arousal", None),
            "intensity": data.pop("intensity", None),
            "energy": data.pop("energy", None),
            "cognitive_load": data.pop("cognitive_load", None),
            "dark_flag": bool(data.pop("dark_flag", 0) or 0),
            "too_heavy_flag": bool(data.pop("too_heavy_flag", 0) or 0),
            "background_friendly": bool(data.pop("background_friendly", 0) or 0),
            "raw_tag_result": json.loads(data.pop("raw_tag_result_json") or "{}"),
            "final_tag_result": json.loads(data.pop("final_tag_result_json") or "{}"),
        }
        return data

    @staticmethod
    def _media_transcription_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["candidates"] = json.loads(data.pop("candidates_json") or "[]")
        data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        data["needs_user_review"] = bool(data["needs_user_review"])
        return data

    @staticmethod
    def _recommendation_log_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        response = json.loads(data.pop("response_json") or "{}")
        recommendations = response.get("recommendations", {})
        flat_items: list[dict[str, Any]] = []
        for items in recommendations.values():
            for item in items:
                flat_items.append(
                    {
                        "content_id": item.get("content_id", ""),
                        "content_type": item.get("content_type", ""),
                        "title": item.get("title", ""),
                        "creator": item.get("creator", ""),
                        "score_percent": item.get("score_percent", 0),
                        "display_tags": item.get("display_tags", []),
                        "reason": item.get("reason", ""),
                    }
                )
        data["response"] = response
        data["top_recommendations"] = flat_items[:8]
        return data

    @staticmethod
    def _saved_item_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["display_tags"] = json.loads(data.pop("display_tags_json") or "[]")
        data["snapshot"] = json.loads(data.pop("snapshot_json") or "{}")
        return data

    @staticmethod
    def _feedback_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json"))
        return data
