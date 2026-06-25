from __future__ import annotations

import base64
import json
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main


class FakeMediaTranscriber:
    def transcribe_bytes(
        self,
        *,
        input_type: str,
        content: bytes,
        filename: str = "",
        mime_type: str = "",
        provider: str | None = None,
    ) -> dict[str, Any]:
        text = "테스트 파일에서 추출한 문장입니다. 피곤하지만 뿌듯한 하루입니다."
        return {
            "input_type": input_type,
            "provider": "fake-http",
            "model_name": "fake-http-media",
            "source_filename": filename,
            "source_mime_type": mime_type,
            "source_size_bytes": len(content),
            "file_sha256": "fake-http-sha",
            "candidates": [
                {
                    "rank": 1,
                    "text": text,
                    "confidence": 0.9,
                    "provider": "fake-http",
                    "model_name": "fake-http-media",
                }
            ],
            "selected_text": text,
            "confidence": 0.9,
            "needs_user_review": True,
            "status": "candidate_ready",
            "metadata": {"test_double": True, "temp_file_deleted": True},
        }


def run_link_server() -> tuple[ThreadingHTTPServer, tempfile.TemporaryDirectory[str], str]:
    temp_dir = tempfile.TemporaryDirectory()
    Path(temp_dir.name, "index.html").write_text(
        "<html><body><h1>산책과 운동</h1>"
        "<p>오늘은 코딩하고 산책했다. 피곤하지만 뿌듯했다.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    handler = partial(SimpleHTTPRequestHandler, directory=temp_dir.name)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, temp_dir, f"http://127.0.0.1:{server.server_address[1]}/index.html"


def main_check() -> dict[str, Any]:
    main.controller.media_transcriber = FakeMediaTranscriber()
    encoded = base64.b64encode(b"fake image")
    image = main.controller.transcribe_media_bytes(
        user_id=1,
        input_type="image",
        content=base64.b64decode(encoded),
        filename="note.png",
        mime_type="image/png",
    )
    audio = main.controller.transcribe_media_bytes(
        user_id=1,
        input_type="audio",
        content=base64.b64decode(encoded),
        filename="note.webm",
        mime_type="audio/webm",
    )
    image_media = image["media_transcription"]
    diary = main.controller.create_diary_from_transcription(
        {
            "user_id": 1,
            "media_id": image_media["media_id"],
            "selected_text": image_media["selected_text"],
            "context_text": "집에 가서 빨리 쉬고 싶다.",
            "per_type": 2,
        }
    )
    server, temp_dir, url = run_link_server()
    try:
        link = main.controller.transcribe_link(
            user_id=1,
            url=url,
        )
    finally:
        server.shutdown()
        temp_dir.cleanup()

    return {
        "image": {
            "status": 200,
            "input_type": image["media_transcription"].get("input_type"),
            "selected_text": image["media_transcription"].get("selected_text"),
        },
        "audio": {
            "status": 200,
            "input_type": audio["media_transcription"].get("input_type"),
            "selected_text": audio["media_transcription"].get("selected_text"),
        },
        "diary_from_image": {
            "status": 200,
            "entry_id": diary["diary"]["entry_id"],
            "selected_text": diary["selected_text"],
            "raw_text": diary["raw_text"],
            "analysis_valence": diary["diary"]["analysis"]["structured_emotion"]["valence"],
        },
        "link": {
            "status": 200,
            "input_type": link["media_transcription"].get("input_type"),
            "selected_text": link["media_transcription"].get("selected_text"),
        },
    }


if __name__ == "__main__":
    report = main_check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if any(item["status"] != 200 for item in report.values()):
        raise SystemExit(1)
