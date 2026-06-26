from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any
from urllib.parse import parse_qs

from app.controllers import ApiError, BackendController


class MiniASGI:
    """Tiny ASGI fallback so the MVP can run even before FastAPI is installed."""

    def __init__(self, controller: BackendController) -> None:
        self.controller = controller

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        method = scope["method"].upper()
        path = scope["path"]
        query = parse_qs(scope.get("query_string", b"").decode("utf-8"))

        if method == "OPTIONS":
            await self._send_json(send, 204, {})
            return

        try:
            body = await self._read_json(receive)
            status, payload = self._route(method, path, query, body)
        except ApiError as exc:
            status, payload = exc.status_code, {"detail": exc.message}
        except Exception as exc:
            status, payload = 500, {"detail": f"internal error: {exc}"}
        await self._send_json(send, status, payload)

    async def _lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _read_json(self, receive: Any) -> dict[str, Any]:
        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        raw = b"".join(chunks)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _route(
        self, method: str, path: str, query: dict[str, list[str]], body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        if method == "GET" and path == "/api/health":
            return 200, self.controller.health()
        if method == "GET" and path == "/api/content":
            return 200, self.controller.list_contents(self._first(query, "content_type"))
        content_match = re.fullmatch(r"/api/content/(.+)", path)
        if method == "GET" and content_match:
            return 200, self.controller.get_content(content_match.group(1))
        if method == "POST" and path == "/api/auth/register":
            return 200, self.controller.register_user(body)
        if method == "POST" and path == "/api/auth/login":
            return 200, self.controller.login_user(body)
        if method == "POST" and path == "/api/auth/guest":
            return 200, self.controller.guest_login()
        if method == "GET" and path == "/api/auth/check-username":
            return 200, self.controller.check_username(self._first(query, "username") or "")
        if method in {"POST"} and path in {"/api/users", "/api/users/login"}:
            return 200, self.controller.login_or_create_user(body)
        user_match = re.fullmatch(r"/api/users/(\d+)", path)
        if method == "GET" and user_match:
            return 200, self.controller.get_user(int(user_match.group(1)))
        if method == "POST" and path == "/api/diary":
            return 200, self.controller.create_diary(body)
        if method == "POST" and path == "/api/media/transcribe-base64":
            try:
                content = base64.b64decode(
                    str(body.get("data_base64") or ""),
                    validate=True,
                )
            except (binascii.Error, ValueError) as exc:
                raise ApiError(400, "data_base64 is invalid") from exc
            return 200, self.controller.transcribe_media_bytes(
                user_id=int(body.get("user_id") or 1),
                input_type=str(body.get("input_type") or ""),
                content=content,
                filename=str(body.get("filename") or ""),
                mime_type=str(body.get("mime_type") or ""),
                provider=body.get("provider"),
            )
        if method == "POST" and path == "/api/link/transcribe":
            return 200, self.controller.transcribe_link(
                user_id=int(body.get("user_id") or 1),
                url=str(body.get("url") or ""),
            )
        if method == "POST" and path == "/api/diary/from-transcription":
            return 200, self.controller.create_diary_from_transcription(body)
        if method == "POST" and path == "/api/recommend":
            return 200, self.controller.recommend(body)
        if method == "POST" and path == "/api/feedback":
            return 200, self.controller.add_feedback(body)
        if method == "GET" and path == "/api/feedback":
            user_id = int(self._first(query, "user_id") or 1)
            limit = int(self._first(query, "limit") or 100)
            return 200, self.controller.list_feedback(user_id, limit)
        if method == "GET" and path == "/api/entries":
            user_id = int(self._first(query, "user_id") or 1)
            limit = int(self._first(query, "limit") or 50)
            return 200, self.controller.list_entries(user_id, limit)
        if method == "GET" and path == "/api/activity":
            user_id = int(self._first(query, "user_id") or 1)
            limit = int(self._first(query, "limit") or 50)
            return 200, self.controller.get_activity(user_id, limit)
        if method == "GET" and path == "/api/repository":
            user_id = int(self._first(query, "user_id") or 1)
            content_type = self._first(query, "content_type")
            status = self._first(query, "status")
            limit = int(self._first(query, "limit") or 200)
            return 200, self.controller.list_repository(user_id, content_type, status, limit)
        if method == "POST" and path == "/api/repository/save":
            return 200, self.controller.save_repository_item(body)
        repository_match = re.fullmatch(r"/api/repository/(\d+)", path)
        if method == "PATCH" and repository_match:
            return 200, self.controller.update_repository_item(
                int(repository_match.group(1)), body
            )
        if method == "DELETE" and repository_match:
            user_id = int(self._first(query, "user_id") or body.get("user_id") or 1)
            return 200, self.controller.delete_repository_item(
                int(repository_match.group(1)), user_id
            )
        if method == "GET" and path == "/api/taste-analysis":
            user_id = int(self._first(query, "user_id") or 1)
            use_openai = (self._first(query, "use_openai") or "true").lower() not in {
                "0",
                "false",
                "no",
            }
            return 200, self.controller.analyze_saved_taste(user_id, use_openai)

        entry_match = re.fullmatch(r"/api/diary/(\d+)", path)
        if method == "DELETE" and entry_match:
            return 200, self.controller.delete_entry(int(entry_match.group(1)))

        media_match = re.fullmatch(r"/api/media/transcriptions/(\d+)", path)
        if method == "GET" and media_match:
            return 200, self.controller.get_media_transcription(int(media_match.group(1)))
        if method == "DELETE" and media_match:
            return 200, self.controller.delete_media_transcription(
                int(media_match.group(1))
            )

        profile_match = re.fullmatch(r"/api/profile/(\d+)", path)
        if method == "GET" and profile_match:
            return 200, self.controller.get_profile(int(profile_match.group(1)))

        reset_match = re.fullmatch(r"/api/users/(\d+)/memory/reset", path)
        if method == "POST" and reset_match:
            return 200, self.controller.reset_memory(int(reset_match.group(1)))

        delete_user_match = re.fullmatch(r"/api/users/(\d+)/data", path)
        if method == "DELETE" and delete_user_match:
            return 200, self.controller.delete_user_data(int(delete_user_match.group(1)))

        raise ApiError(404, "endpoint not found")

    @staticmethod
    def _first(query: dict[str, list[str]], key: str) -> str | None:
        values = query.get(key)
        return values[0] if values else None

    async def _send_json(self, send: Any, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"access-control-allow-origin", b"*"),
            (b"access-control-allow-methods", b"GET,POST,PATCH,DELETE,OPTIONS"),
            (b"access-control-allow-headers", b"content-type"),
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
