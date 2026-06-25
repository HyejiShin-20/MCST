from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.controllers import BackendController
from app.database import Database


class FakeMediaTranscriber:
    def transcribe_bytes(
        self,
        *,
        input_type: str,
        content: bytes,
        filename: str = "",
        mime_type: str = "",
        provider: str | None = None,
    ) -> dict[str, object]:
        text = "Today I felt tired but curious, and I want a quiet recommendation."
        return {
            "input_type": input_type,
            "provider": provider or "fake",
            "model_name": "fake-transcriber",
            "source_filename": filename,
            "source_mime_type": mime_type,
            "source_size_bytes": len(content),
            "file_sha256": "fake-sha256",
            "candidates": [
                {
                    "rank": 1,
                    "text": text,
                    "confidence": 0.91,
                    "provider": provider or "fake",
                    "model_name": "fake-transcriber",
                }
            ],
            "selected_text": text,
            "confidence": 0.91,
            "needs_user_review": True,
            "status": "candidate_ready",
            "metadata": {"temp_file_deleted": True},
        }


class FakeLinkTranscriber:
    def transcribe_url(self, url: str) -> dict[str, object]:
        text = "링크 속 글은 오늘의 산책과 운동, 그리고 뿌듯한 마무리를 다룬다."
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
                    "confidence": 0.77,
                    "provider": "fake-link",
                    "model_name": "fake-link-transcriber",
                }
            ],
            "selected_text": text,
            "confidence": 0.77,
            "needs_user_review": True,
            "status": "candidate_ready",
            "metadata": {"url": url, "test_double": True},
        }


class BackendServiceTest(unittest.TestCase):
    def test_diary_recommend_memory_and_delete_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = BackendController(
                database=Database(
                    Path(temp_dir) / "test.sqlite3",
                    Path("data/samples/content_items.json"),
                )
            )
            diary = controller.create_diary(
                {
                    "user_id": 1,
                    "input_type": "text",
                    "text": "우주비행사가 되고 싶어졌다. 별과 미래를 생각하니 설렌다.",
                }
            )
            self.assertIn("우주", diary["analysis"]["topics"])
            self.assertEqual(
                diary["analysis"]["situation"]["primary_event"],
                "진로/미래 탐색",
            )
            self.assertIn("emotion_details", diary["analysis"])
            self.assertIn("recommendation_strategy", diary["analysis"])
            self.assertIn("emotional_goal", diary["analysis"])
            self.assertIn("nuanced_emotion", diary["analysis"])
            self.assertIn("confidence", diary["analysis"])

            recommendations = controller.recommend(
                {"entry_id": diary["entry_id"], "per_type": 4}
            )
            self.assertEqual(len(recommendations["recommendations"]["movie"]), 0)
            self.assertGreaterEqual(len(recommendations["recommendations"]["book"]), 1)
            self.assertGreaterEqual(
                len(recommendations["recommendations"]["music"])
                + len(recommendations["recommendations"]["book"]),
                1,
            )
            self.assertIn("algorithm_profile", recommendations)
            self.assertIn("recommendation_strategy", recommendations)
            self.assertIn("recommendation_roles", recommendations)
            self.assertIn("recommendation_slots", recommendations)
            self.assertIn(
                "situation_fit",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "support_fit",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "role_fit",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "emotion_regulation",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "recovery_potential",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "input_priority_fit",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "context_mismatch_penalty",
                recommendations["recommendations"]["book"][0]["score_components"],
            )
            self.assertIn(
                "content_affect_profile",
                recommendations["recommendations"]["book"][0],
            )
            self.assertIn(
                "dominant_tone",
                recommendations["recommendations"]["book"][0]["content_affect_profile"],
            )
            self.assertEqual(
                recommendations["algorithm_profile"]["pipeline"],
                [
                    "strategy_builder",
                    "candidate_retriever",
                    "chroma_candidate_store",
                    "content_affect_profiler",
                    "content_embedding_index",
                    "safety_policy",
                    "feature_extractor",
                    "slot_scorer",
                    "portfolio_builder",
                    "explanation_builder",
                ],
            )
            self.assertGreater(
                recommendations["algorithm_profile"]["content_embedding_cache"]["request_misses"],
                0,
            )
            self.assertIn(
                "recommendation_role",
                recommendations["recommendations"]["book"][0],
            )
            self.assertIn(
                "expected_effect",
                recommendations["recommendations"]["book"][0],
            )
            self.assertTrue(
                any(
                    "우주" in item["topic_tags"]
                    for item in recommendations["recommendations"]["book"]
                )
            )

            top_book = recommendations["recommendations"]["book"][0]
            feedback = controller.add_feedback(
                {
                    "user_id": 1,
                    "entry_id": diary["entry_id"],
                    "log_id": recommendations["log_id"],
                    "content_id": top_book["content_id"],
                    "signal": "saved",
                    "rating": 5,
                    "note": "다음에도 이런 정서 톤을 보고 싶다.",
                }
            )
            self.assertEqual(feedback["signal"], "saved")
            self.assertEqual(feedback["profile"]["feedback_count"], 1)
            self.assertIn(
                top_book["content_type"],
                feedback["profile"]["content_feedback_profile"]["preferred_content_types"],
            )
            self.assertEqual(
                len(controller.list_feedback(1)["feedback_events"]),
                1,
            )

            personalized = controller.recommend(
                {"entry_id": diary["entry_id"], "per_type": 4}
            )
            self.assertGreater(
                personalized["algorithm_profile"]["content_embedding_cache"]["request_hits"],
                0,
            )
            self.assertIn(
                "preference_penalty",
                personalized["recommendations"]["book"][0]["score_components"],
            )

            profile = controller.get_profile(1)
            self.assertIsNotNone(profile["profile"])

            controller.reset_memory(1)
            self.assertIsNone(controller.get_profile(1)["profile"])
            self.assertEqual(controller.list_feedback(1)["feedback_events"], [])

            controller.delete_user_data(1)
            self.assertEqual(controller.list_entries(1)["entries"], [])

    def test_media_transcription_review_confirm_and_delete_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = BackendController(
                database=Database(
                    Path(temp_dir) / "test.sqlite3",
                    Path("data/samples/content_items.json"),
                ),
                media_transcriber=FakeMediaTranscriber(),
                link_transcriber=FakeLinkTranscriber(),
            )

            media_response = controller.transcribe_media_bytes(
                user_id=7,
                input_type="image",
                content=b"fake image bytes",
                filename="daily-note.png",
                mime_type="image/png",
            )
            media = media_response["media_transcription"]
            self.assertIsNotNone(media)
            self.assertEqual(media["input_type"], "image")
            self.assertTrue(media["needs_user_review"])
            self.assertEqual(media["status"], "candidate_ready")
            self.assertEqual(len(media["candidates"]), 1)

            reviewed_text = (
                "I had a quiet day. I was tired, but I also felt curious and hopeful."
            )
            context_text = "집에 가고 싶다. 너무 피곤해서 빨리 쉬고 싶다."
            response = controller.create_diary_from_transcription(
                {
                    "user_id": 7,
                    "media_id": media["media_id"],
                    "selected_text": reviewed_text,
                    "context_text": context_text,
                    "per_type": 3,
                }
            )
            self.assertEqual(response["selected_text"], reviewed_text)
            self.assertEqual(response["diary"]["user_id"], 7)
            self.assertEqual(
                response["media_transcription"]["status"],
                "confirmed",
            )
            self.assertFalse(response["media_transcription"]["needs_user_review"])
            self.assertEqual(
                response["media_transcription"]["entry_id"],
                response["diary"]["entry_id"],
            )
            self.assertEqual(
                response["recommendations"]["analysis_summary"],
                response["diary"]["analysis"]["summary"],
            )
            self.assertEqual(
                len(response["recommendations"]["recommendations"]["movie"]),
                0,
            )
            self.assertEqual(
                len(response["recommendations"]["recommendations"]["music"]),
                3,
            )
            self.assertEqual(
                len(response["recommendations"]["recommendations"]["book"]),
                3,
            )

            entry = controller.database.get_entry(response["diary"]["entry_id"])
            self.assertEqual(entry["input_type"], "image")
            self.assertEqual(entry["source_media_id"], media["media_id"])
            self.assertIn(context_text, entry["raw_text"])
            self.assertIn(reviewed_text, entry["raw_text"])
            self.assertIn(context_text, entry["processed_text"])
            self.assertIn("피로", response["diary"]["analysis"]["emotions"])

            delete_response = controller.delete_media_transcription(media["media_id"])
            self.assertTrue(delete_response["deleted"])
            self.assertIsNone(
                controller.database.get_media_transcription(media["media_id"])
            )

            link_response = controller.transcribe_link(
                user_id=7,
                url="https://example.com/daily-note",
            )
            link_media = link_response["media_transcription"]
            self.assertEqual(link_media["input_type"], "link")
            self.assertEqual(link_media["provider"], "fake-link")
            self.assertEqual(link_media["status"], "candidate_ready")
            self.assertIn("뿌듯한 마무리", link_media["selected_text"])

            link_diary = controller.create_diary_from_transcription(
                {
                    "user_id": 7,
                    "media_id": link_media["media_id"],
                    "selected_text": link_media["selected_text"],
                    "per_type": 2,
                }
            )
            self.assertEqual(link_diary["diary"]["user_id"], 7)
            self.assertEqual(
                controller.database.get_entry(link_diary["diary"]["entry_id"])["input_type"],
                "link",
            )

            image_again = controller.transcribe_media_bytes(
                user_id=7,
                input_type="image",
                content=b"second fake image bytes",
                filename="drawing.png",
                mime_type="image/png",
            )["media_transcription"]
            audio_again = controller.transcribe_media_bytes(
                user_id=7,
                input_type="audio",
                content=b"fake audio bytes",
                filename="voice.webm",
                mime_type="audio/webm",
            )["media_transcription"]
            multimodal = controller.create_diary_from_transcription(
                {
                    "user_id": 7,
                    "media_id": image_again["media_id"],
                    "media_items": [
                        {
                            "media_id": image_again["media_id"],
                            "selected_text": "A playful drawing from my diary.",
                        },
                        {
                            "media_id": audio_again["media_id"],
                            "selected_text": "I am tired but proud after finishing work.",
                        },
                    ],
                    "selected_text": (
                        "[그림]\nA playful drawing from my diary.\n\n"
                        "[음성]\nI am tired but proud after finishing work."
                    ),
                    "context_text": "오늘은 지쳤지만 뿌듯했다.",
                    "per_type": 2,
                }
            )
            multimodal_entry = controller.database.get_entry(
                multimodal["diary"]["entry_id"]
            )
            self.assertEqual(multimodal_entry["input_type"], "multimodal")
            self.assertEqual(
                set(multimodal["media_ids"]),
                {image_again["media_id"], audio_again["media_id"]},
            )
            self.assertIn("A playful drawing", multimodal_entry["raw_text"])
            self.assertIn("I am tired but proud", multimodal_entry["processed_text"])
            self.assertEqual(
                controller.database.get_media_transcription(image_again["media_id"])["entry_id"],
                multimodal["diary"]["entry_id"],
            )
            self.assertEqual(
                controller.database.get_media_transcription(audio_again["media_id"])["entry_id"],
                multimodal["diary"]["entry_id"],
            )

    def test_user_repository_activity_and_saved_taste_analysis_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = BackendController(
                database=Database(
                    Path(temp_dir) / "test.sqlite3",
                    Path("data/samples/content_items.json"),
                )
            )
            user = controller.login_or_create_user(
                {"username": "tester", "display_name": "Tester"}
            )["user"]
            self.assertGreater(user["user_id"], 0)

            diary = controller.create_diary(
                {
                    "user_id": user["user_id"],
                    "input_type": "text",
                    "text": "오늘은 코딩 프로젝트를 하고 운동도 했다. 피곤하지만 뿌듯한 하루였다.",
                }
            )
            recommendations = controller.recommend(
                {"entry_id": diary["entry_id"], "content_types": ["music", "book"], "per_type": 2}
            )
            first_item = (
                recommendations["recommendations"]["music"]
                or recommendations["recommendations"]["book"]
            )[0]
            saved = controller.save_repository_item(
                {
                    "user_id": user["user_id"],
                    "content_id": first_item["content_id"],
                    "content_type": first_item["content_type"],
                    "title": first_item["title"],
                    "creator": first_item["creator"],
                    "genre": first_item["genre"],
                    "source": first_item["source"],
                    "entry_id": diary["entry_id"],
                    "log_id": recommendations["log_id"],
                    "recommendation_role": first_item.get("recommendation_role", ""),
                    "reason": first_item["reason"],
                    "score_percent": first_item["score_percent"],
                    "display_tags": first_item["display_tags"],
                    "snapshot": first_item,
                }
            )["saved_item"]
            self.assertEqual(saved["content_id"], first_item["content_id"])
            self.assertEqual(saved["status"], "planned")

            repository = controller.list_repository(user["user_id"])
            self.assertEqual(repository["counts"]["total"], 1)
            self.assertEqual(repository["saved_items"][0]["title"], first_item["title"])

            activity = controller.get_activity(user["user_id"])
            self.assertEqual(activity["summary"]["entry_count"], 1)
            self.assertEqual(activity["summary"]["recommendation_count"], 1)
            self.assertEqual(activity["summary"]["saved_count"], 1)

            taste = controller.analyze_saved_taste(user["user_id"], use_openai=False)
            self.assertEqual(taste["basis"], "saved_items_only")
            self.assertEqual(taste["saved_count"], 1)
            self.assertIn("taste_summary", taste)
            self.assertIn("emotion_patterns", taste)
            self.assertIn("content_ratio", taste)
            self.assertIn("recommendation_direction", taste)

            deleted = controller.delete_repository_item(saved["saved_id"], user["user_id"])
            self.assertTrue(deleted["deleted"])
            self.assertEqual(controller.list_repository(user["user_id"])["counts"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
