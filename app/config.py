from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def _load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"
DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "app.sqlite3"))


def _configure_google_application_credentials() -> None:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    raw_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    raw_b64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64", "").strip()
    if not raw_json and not raw_b64:
        return

    try:
        content = (
            base64.b64decode(raw_b64).decode("utf-8")
            if raw_b64
            else raw_json
        )
    except Exception:
        return
    if not content:
        return

    secret_dir = Path(os.getenv("GOOGLE_CREDENTIALS_TMP_DIR", tempfile.gettempdir()))
    secret_dir.mkdir(parents=True, exist_ok=True)
    credential_path = secret_dir / "google-application-credentials.json"
    credential_path.write_text(content, encoding="utf-8")
    try:
        credential_path.chmod(0o600)
    except OSError:
        pass
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)


_configure_google_application_credentials()


def _resolve_sample_content_path() -> Path:
    # 상대 경로 환경변수도 항상 프로젝트 루트(BASE_DIR) 기준으로 풀어
    # 배포 환경의 작업 디렉토리(CWD)와 무관하게 동작하도록 한다.
    raw = os.getenv("SAMPLE_CONTENT_PATH")
    if not raw:
        return SAMPLE_DIR / "content_items.json"
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (BASE_DIR / candidate)


SAMPLE_CONTENT_PATH = _resolve_sample_content_path()

HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
HF_EMOTION_MODEL = os.getenv("HF_EMOTION_MODEL", "")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "true").lower() not in {
    "0",
    "false",
    "no",
}
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# OpenAI 임베딩 호출 1회당 최대 입력 개수. 대량 카탈로그를 안전하게 배치 처리한다.
OPENAI_EMBEDDING_BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "128"))

DEFAULT_USER_ID = int(os.getenv("DEFAULT_USER_ID", "1"))
DEFAULT_PER_TYPE = int(os.getenv("DEFAULT_PER_TYPE", "4"))
MAX_PER_TYPE = int(os.getenv("MAX_PER_TYPE", "4"))
DEFAULT_CONTENT_TYPES = [
    item.strip()
    for item in os.getenv("DEFAULT_CONTENT_TYPES", "music,book,movie").split(",")
    if item.strip() in {"movie", "music", "book"}
]
if not DEFAULT_CONTENT_TYPES:
    DEFAULT_CONTENT_TYPES = ["music", "book", "movie"]
HIDE_SAMPLE_CONTENT_IN_DEMO = os.getenv(
    "HIDE_SAMPLE_CONTENT_IN_DEMO",
    "true",
).lower() not in {"0", "false", "no"}

CURRENT_TAG_VERSION = os.getenv(
    "CURRENT_TAG_VERSION",
    "v1_keyword_prototype_balanced_emotion",
)
MIN_RECOMMEND_CONFIDENCE = float(os.getenv("MIN_RECOMMEND_CONFIDENCE", "0.55"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.45"))
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.75"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", DATA_DIR / "exports"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "culture_contents")
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "chroma").lower()
CHROMA_TOP_K = int(os.getenv("CHROMA_TOP_K", "80"))

MEDIA_TMP_DIR = Path(os.getenv("MEDIA_TMP_DIR", DATA_DIR / "tmp_media"))
MEDIA_MAX_UPLOAD_MB = int(os.getenv("MEDIA_MAX_UPLOAD_MB", "25"))
MEDIA_DELETE_UPLOADS = os.getenv("MEDIA_DELETE_UPLOADS", "true").lower() not in {
    "0",
    "false",
    "no",
}
TRANSCRIPTION_REVIEW_REQUIRED = os.getenv(
    "TRANSCRIPTION_REVIEW_REQUIRED",
    "true",
).lower() not in {"0", "false", "no"}

OCR_PROVIDER = os.getenv("OCR_PROVIDER", "google")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_VISION_FEATURE = os.getenv("GOOGLE_VISION_FEATURE", "DOCUMENT_TEXT_DETECTION")
GOOGLE_VISION_LANGUAGE_HINTS = [
    item.strip()
    for item in os.getenv("GOOGLE_VISION_LANGUAGE_HINTS", "ko,en").split(",")
    if item.strip()
]

STT_PROVIDER = os.getenv("STT_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL",
    "gpt-4o-mini-transcribe",
)
OPENAI_TRANSCRIPTION_LANGUAGE = os.getenv("OPENAI_TRANSCRIPTION_LANGUAGE", "ko")
OPENAI_EMOTION_MODEL = os.getenv("OPENAI_EMOTION_MODEL", "gpt-5.4-mini")
OPENAI_EMOTION_FALLBACK_MODEL = os.getenv("OPENAI_EMOTION_FALLBACK_MODEL", "gpt-5.4-nano")
OPENAI_EMOTION_USE_WHEN = os.getenv("OPENAI_EMOTION_USE_WHEN", "always")
OPENAI_ANALYSIS_ENABLED = os.getenv("OPENAI_ANALYSIS_ENABLED", "true").lower() not in {
    "0",
    "false",
    "no",
}
OPENAI_ANALYSIS_TIMEOUT_SECONDS = float(
    os.getenv("OPENAI_ANALYSIS_TIMEOUT_SECONDS", "25")
)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini")
OPENAI_VISION_DETAIL = os.getenv("OPENAI_VISION_DETAIL", "high")
OPENAI_IMAGE_CAPTION_ENABLED = os.getenv(
    "OPENAI_IMAGE_CAPTION_ENABLED",
    "true",
).lower() not in {"0", "false", "no"}
OPENAI_IMAGE_CAPTION_TIMEOUT_SECONDS = float(
    os.getenv("OPENAI_IMAGE_CAPTION_TIMEOUT_SECONDS", "25")
)

LINK_TIMEOUT_SECONDS = float(os.getenv("LINK_TIMEOUT_SECONDS", "10"))
LINK_MAX_BYTES = int(os.getenv("LINK_MAX_BYTES", str(2 * 1024 * 1024)))
LINK_MAX_TEXT_CHARS = int(os.getenv("LINK_MAX_TEXT_CHARS", "6000"))
