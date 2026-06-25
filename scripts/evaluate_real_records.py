from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.controllers import ApiError, BackendController


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
        text = (
            "오늘은 피곤했지만 마음을 정리하고 싶다. "
            "조용한 작품과 회복에 도움이 되는 추천을 받고 싶다."
        )
        return {
            "input_type": input_type,
            "provider": provider or "fake",
            "model_name": "fake-media-transcriber",
            "source_filename": filename,
            "source_mime_type": mime_type,
            "source_size_bytes": len(content),
            "file_sha256": "fake-sha256",
            "candidates": [
                {
                    "rank": 1,
                    "text": text,
                    "confidence": 0.92,
                    "provider": provider or "fake",
                    "model_name": "fake-media-transcriber",
                }
            ],
            "selected_text": text,
            "confidence": 0.92,
            "needs_user_review": True,
            "status": "candidate_ready",
            "metadata": {"temp_file_deleted": True, "test_double": True},
        }


class FakeLinkTranscriber:
    def transcribe_url(self, url: str) -> dict[str, Any]:
        text = "링크 본문에는 산책과 운동을 마친 뒤 피곤하지만 뿌듯했던 하루가 담겨 있다."
        return {
            "input_type": "link",
            "provider": "fake-link",
            "model_name": "fake-link-transcriber",
            "source_filename": url,
            "source_mime_type": "text/html",
            "source_size_bytes": len(text.encode("utf-8")),
            "file_sha256": "fake-link-sha256",
            "candidates": [
                {
                    "rank": 1,
                    "text": text,
                    "confidence": 0.86,
                    "provider": "fake-link",
                    "model_name": "fake-link-transcriber",
                }
            ],
            "selected_text": text,
            "confidence": 0.86,
            "needs_user_review": True,
            "status": "candidate_ready",
            "metadata": {"url": url, "test_double": True},
        }


CASES: list[dict[str, Any]] = [
    {
        "id": "relationship_hurt",
        "label": "관계 상처/분노+슬픔",
        "text": "어제 친한 친구에게 고민을 말했는데 대충 넘기는 느낌이라 서운하고 화가 났다. 그래도 관계를 끊고 싶은 건 아니고 마음을 좀 정리하고 싶다.",
        "expected_intents": ["negative_emotion_based", "mixed_emotion_based"],
        "expected_emotions_any": ["분노", "슬픔", "외로움"],
        "expected_roles_any": ["공감", "안정", "회복", "전환"],
    },
    {
        "id": "quiet_neutral",
        "label": "중립/조용한 하루",
        "text": "오늘은 특별한 일은 없었다. 퇴근하고 집에서 밥을 먹고 빨래를 돌린 다음 조용히 누워 있었다.",
        "expected_intents": ["neutral_daily_based"],
        "expected_emotions_any": ["평범함", "평온"],
        "expected_roles_any": ["유지", "배경", "발견", "가벼운 성찰"],
    },
    {
        "id": "achievement_tired",
        "label": "성취+피로 복합",
        "text": "프로젝트 데모가 생각보다 잘 끝나서 뿌듯했다. 그런데 긴장이 풀리니까 너무 피곤해서 오늘은 아무것도 더 하고 싶지 않다.",
        "expected_intents": ["mixed_emotion_based"],
        "expected_emotions_any": ["뿌듯함", "무기력", "안도감", "불안"],
        "expected_roles_any": ["축하", "음미", "안정", "회복"],
    },
    {
        "id": "curiosity_learning",
        "label": "관심사/학습 욕구",
        "text": "요즘 우주 다큐멘터리를 보다가 천문학이 너무 궁금해졌다. 어렵더라도 관련된 책이나 영화를 더 찾아보고 싶다.",
        "expected_intents": ["interest_based"],
        "expected_emotions_any": ["호기심", "설렘"],
        "expected_roles_any": ["탐색", "학습", "몰입", "영감"],
    },
    {
        "id": "bright_social",
        "label": "긍정/즐거운 만남",
        "text": "오랜만에 동아리 사람들이랑 웃으면서 밥을 먹었다. 별거 아닌 농담도 너무 즐거웠고 기분이 환해졌다.",
        "expected_intents": ["positive_emotion_based"],
        "expected_emotions_any": ["기쁨", "즐거움"],
        "expected_roles_any": ["증폭", "음미", "영감", "발견"],
    },
]


def compact_recommendations(recommendations: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for content_type, items in recommendations.items():
        compact[content_type] = [
            {
                "title": item["title"],
                "creator": item["creator"],
                "role": item["recommendation_role"],
                "reason": item["reason"],
            }
            for item in items[:2]
        ]
    return compact


def run_case(controller: BackendController, user_id: int, case: dict[str, Any]) -> dict[str, Any]:
    diary = controller.create_diary(
        {"user_id": user_id, "input_type": "text", "text": case["text"]}
    )
    recommendations = controller.recommend(
        {
            "entry_id": diary["entry_id"],
            "content_types": ["music", "book"],
            "per_type": 3,
        }
    )
    analysis = diary["analysis"]
    checks = {
        "intent": analysis["situation"]["intent"] in case["expected_intents"],
        "emotion": bool(set(analysis["emotions"]) & set(case["expected_emotions_any"])),
        "role": bool(
            set(recommendations["recommendation_roles"])
            & set(case["expected_roles_any"])
        ),
        "counts": (
            len(recommendations["recommendations"].get("movie", [])) == 0
            and len(recommendations["recommendations"].get("music", [])) == 3
            and len(recommendations["recommendations"].get("book", [])) == 3
        ),
    }
    return {
        "id": case["id"],
        "label": case["label"],
        "input": case["text"],
        "passed": all(checks.values()),
        "checks": checks,
        "analysis": {
            "emotions": analysis["emotions"],
            "topics": analysis["topics"],
            "intent": analysis["situation"]["intent"],
            "primary_event": analysis["situation"]["primary_event"],
            "roles": recommendations["recommendation_roles"],
        },
        "top_recommendations": compact_recommendations(
            recommendations["recommendations"]
        ),
    }


def probe_multimodal(user_id: int) -> dict[str, Any]:
    controller = BackendController(
        media_transcriber=FakeMediaTranscriber(),
        link_transcriber=FakeLinkTranscriber(),
    )
    results: dict[str, Any] = {}
    try:
        for input_type in ["image", "audio"]:
            media_response = controller.transcribe_media_bytes(
                user_id=user_id,
                input_type=input_type,
                content=f"fake {input_type}".encode("utf-8"),
                filename=f"sample.{ 'png' if input_type == 'image' else 'wav' }",
                mime_type=f"{input_type}/{'png' if input_type == 'image' else 'wav'}",
            )
            media = media_response["media_transcription"]
            confirmed = controller.create_diary_from_transcription(
                {
                    "user_id": user_id,
                    "media_id": media["media_id"],
                    "selected_text": media["candidates"][0]["text"],
                    "per_type": 3,
                }
            )
            results[input_type] = {
                "implemented": True,
                "media_id": media["media_id"],
                "needs_user_review": media["needs_user_review"],
                "entry_id": confirmed["diary"]["entry_id"],
                "recommendation_counts": {
                    key: len(value)
                    for key, value in confirmed["recommendations"][
                        "recommendations"
                    ].items()
                },
            }
        link_response = controller.transcribe_link(
            user_id=user_id,
            url="https://example.com/daily",
        )
        link_media = link_response["media_transcription"]
        confirmed = controller.create_diary_from_transcription(
            {
                "user_id": user_id,
                "media_id": link_media["media_id"],
                "selected_text": link_media["selected_text"],
                "per_type": 3,
            }
        )
        results["link"] = {
            "implemented": True,
            "media_id": link_media["media_id"],
            "needs_user_review": link_media["needs_user_review"],
            "entry_id": confirmed["diary"]["entry_id"],
            "recommendation_counts": {
                key: len(value)
                for key, value in confirmed["recommendations"][
                    "recommendations"
                ].items()
            },
        }
    except ApiError as exc:
        results["error"] = {
            "implemented": False,
            "status_code": exc.status_code,
            "message": exc.message,
        }
    finally:
        controller.delete_user_data(user_id)
    return results


def main() -> None:
    user_id = 880024
    controller = BackendController()
    try:
        results = [run_case(controller, user_id, case) for case in CASES]
        report = {
            "passed": sum(1 for result in results if result["passed"]),
            "total": len(results),
            "results": results,
            "multimodal_transcription_probe": probe_multimodal(user_id + 1),
        }
    finally:
        controller.delete_user_data(user_id)

    output_path = config.EXPORT_DIR / "real_record_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_path": str(output_path), **report}, ensure_ascii=False, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
