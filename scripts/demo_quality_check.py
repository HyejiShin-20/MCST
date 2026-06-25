from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config


CASES = [
    {
        "id": "positive_friend_meeting",
        "label": "긍정/즐거움",
        "text": (
            "오늘 오랜만에 친구들을 만나서 많이 웃었다. 별일 아닌 농담도 즐거웠고 "
            "집에 돌아오는 길까지 기분이 환했다. 이 좋은 기분을 조금 더 이어가고 싶다."
        ),
    },
    {
        "id": "neutral_rain_evening",
        "label": "중립/잔잔한 일상",
        "text": (
            "퇴근하고 집에서 조용히 저녁을 먹었다. 창밖에는 비가 조금 왔고, "
            "특별히 힘든 일도 좋은 일도 없었지만 차분하게 하루를 정리하고 싶다."
        ),
    },
    {
        "id": "negative_exhausted",
        "label": "부정/지침과 위로 필요",
        "text": (
            "오늘은 일이 계속 꼬여서 너무 지쳤다. 괜히 나만 뒤처지는 것 같고 마음이 무겁다. "
            "크게 자극적이지 않으면서도 위로가 되는 음악이나 책을 보고 싶다."
        ),
    },
    {
        "id": "anger_conflict",
        "label": "복합/분노와 관계 갈등",
        "text": (
            "친구가 내 말을 끝까지 듣지도 않고 비난해서 화가 많이 났다. "
            "아직 분하지만 관계를 완전히 끊고 싶은 건 아니라 마음을 가라앉히고 싶다."
        ),
    },
]


def request_json(
    base_url: str,
    path: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {body}") from exc


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            out, err = process.communicate(timeout=5)
            raise RuntimeError(f"server exited early\nSTDOUT={out}\nSTDERR={err}")
        try:
            return request_json(base_url, "/api/health", timeout=5)
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready: {last_error}")


def run_case(base_url: str, user_id: int, case: dict[str, str]) -> dict[str, Any]:
    diary = request_json(
        base_url,
        "/api/diary",
        "POST",
        {
            "user_id": user_id,
            "input_type": "text",
            "text": case["text"],
        },
    )
    recommendations = request_json(
        base_url,
        "/api/recommend",
        "POST",
        {
            "entry_id": diary["entry_id"],
            "content_types": ["music", "book"],
            "per_type": 3,
        },
    )
    grouped = recommendations["recommendations"]
    compact = {}
    for content_type, items in grouped.items():
        compact[content_type] = [
            {
                "title": item["title"],
                "creator": item["creator"],
                "source": item["source"],
                "score_percent": item["score_percent"],
                "role": item.get("recommendation_role"),
                "reason": item["reason"],
            }
            for item in items[:3]
        ]
    counts = {key: len(value) for key, value in grouped.items()}
    sample_leaks = [
        item
        for items in grouped.values()
        for item in items
        if str(item.get("source", "")).startswith("MVP curated sample")
    ]
    return {
        "id": case["id"],
        "label": case["label"],
        "input": case["text"],
        "entry_id": diary["entry_id"],
        "analysis": {
            "summary": diary["analysis"].get("summary"),
            "emotions": diary["analysis"].get("emotions", []),
            "topics": diary["analysis"].get("topics", []),
            "intent": diary["analysis"].get("situation", {}).get("intent"),
            "nuanced_emotion": diary["analysis"].get("nuanced_emotion"),
            "llm_augmented": diary["analysis"].get("llm_augmented", False),
        },
        "retrieval_backend": recommendations.get("retrieval_backend"),
        "embedding_backend": recommendations.get("embedding_backend"),
        "recommendation_counts": counts,
        "recommendations": compact,
        "passed": counts.get("music", 0) >= 3
        and counts.get("book", 0) >= 3
        and not sample_leaks,
        "sample_leaks": len(sample_leaks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an end-to-end music/book demo quality check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--user-id", type=int, default=902624)
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        health = wait_for_server(base_url, process, timeout=120)
        results = [run_case(base_url, args.user_id, case) for case in CASES]
        report = {
            "base_url": base_url,
            "health": health,
            "passed": sum(1 for result in results if result["passed"]),
            "total": len(results),
            "results": results,
        }
        try:
            request_json(base_url, f"/api/users/{args.user_id}/data", "DELETE", timeout=20)
        except Exception:
            pass
    finally:
        if not args.keep_server:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    output_path = config.EXPORT_DIR / "demo_quality_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **report}, ensure_ascii=False, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
