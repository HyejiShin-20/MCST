from __future__ import annotations


class CandidateRetriever:
    def retrieve(self, contents: list[dict], content_types: list[str] | None = None) -> list[dict]:
        if not content_types:
            return contents[:]
        allowed = set(content_types)
        return [content for content in contents if content.get("content_type") in allowed]

