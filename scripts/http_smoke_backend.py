from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from typing import Any


BASE_URL = "http://127.0.0.1:8010"


def request_json(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8010",
            "--log-level",
            "warning",
        ],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        for _ in range(180):
            if process.poll() is not None:
                out, err = process.communicate(timeout=5)
                raise RuntimeError(f"server exited early\nSTDOUT={out}\nSTDERR={err}")
            try:
                health = request_json("/api/health")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise TimeoutError("server did not answer /api/health")

        diary = request_json(
            "/api/diary",
            "POST",
            {
                "user_id": 31,
                "input_type": "text",
                "text": "오늘 발표를 망쳐서 너무 창피하고 불안했다. 다시 자신감을 찾고 싶고 나를 일으켜 줄 음악과 책이 필요하다.",
            },
        )
        recommendations = request_json(
            "/api/recommend",
            "POST",
            {
                "entry_id": diary["entry_id"],
                "content_types": ["music", "book"],
                "per_type": 4,
            },
        )
        first_item = (
            recommendations["recommendations"]["music"]
            or recommendations["recommendations"]["book"]
        )[0]
        feedback = request_json(
            "/api/feedback",
            "POST",
            {
                "user_id": 31,
                "entry_id": diary["entry_id"],
                "log_id": recommendations["log_id"],
                "content_id": first_item["content_id"],
                "signal": "helpful",
                "rating": 5,
                "note": "데모 피드백",
            },
        )
        feedback_events = request_json("/api/feedback?user_id=31")
        profile = request_json("/api/profile/31")
        reset = request_json("/api/users/31/memory/reset", "POST")
        deleted = request_json("/api/users/31/data", "DELETE")

        result = {
            "health": health,
            "entry_id": diary["entry_id"],
            "emotions": diary["analysis"]["emotions"],
            "topics": diary["analysis"]["topics"],
            "situation": diary["analysis"]["situation"],
            "strategy": diary["analysis"]["recommendation_strategy"],
            "emotional_goal": diary["analysis"]["emotional_goal"],
            "nuanced_emotion": diary["analysis"]["nuanced_emotion"],
            "recommendation_counts": {
                key: len(value)
                for key, value in recommendations["recommendations"].items()
            },
            "recommendation_roles": recommendations["recommendation_roles"],
            "content_embedding_cache": recommendations["algorithm_profile"][
                "content_embedding_cache"
            ],
            "top_titles": {
                key: value[0]["title"] if value else ""
                for key, value in recommendations["recommendations"].items()
            },
            "top_roles": {
                key: value[0]["recommendation_role"] if value else ""
                for key, value in recommendations["recommendations"].items()
            },
            "top_components": first_item["score_components"],
            "sample_reason": first_item["reason"],
            "feedback_id": feedback["feedback_id"],
            "feedback_events": len(feedback_events["feedback_events"]),
            "feedback_count": profile["profile"]["feedback_count"],
            "profile_exists": profile["profile"] is not None,
            "memory_reset": reset["memory_reset"],
            "deleted": deleted["deleted"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
