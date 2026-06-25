from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def script_sources(html: str) -> list[str]:
    return re.findall(r'<script\s+src="([^"]+)"', html)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    checks = {
        "frontend/index.html": {
            "ids": ["apiStatus", "transcriptionReview", "transcriptionText", "recommendationsList"],
            "scripts": ["js/session.js", "js/main.js"],
        },
        "frontend/archive.html": {
            "ids": ["archiveTimeline"],
            "scripts": ["js/session.js", "js/archive.js"],
        },
        "frontend/repository.html": {
            "ids": ["repositoryGrid"],
            "scripts": ["js/session.js", "js/repository.js"],
        },
        "frontend/discovery.html": {
            "ids": [],
            "scripts": ["js/session.js", "js/discovery.js"],
        },
    }
    for html_path, expected in checks.items():
        html = read(html_path)
        sources = script_sources(html)
        for script in expected["scripts"]:
            require(script in sources, f"{html_path} missing {script}")
            require((ROOT / "frontend" / script).exists(), f"{script} file does not exist")
        require(
            sources.index("js/session.js") < sources.index(expected["scripts"][-1]),
            f"{html_path} must load session.js before page script",
        )
        require('data-user-panel' in html, f"{html_path} missing user panel")
        for element_id in expected["ids"]:
            require(f'id="{element_id}"' in html, f"{html_path} missing #{element_id}")

    js_expectations = {
        "frontend/js/main.js": ["/api/repository/save", "saveLastRun", "restoreLastRun"],
        "frontend/js/archive.js": ["/api/activity", "archiveTimeline"],
        "frontend/js/repository.js": ["/api/repository", "repositoryGrid"],
        "frontend/js/discovery.js": ["/api/taste-analysis", "saved_count", "generated_report"],
        "frontend/js/session.js": ["/api/users", "emotionCultureUser"],
    }
    for js_path, needles in js_expectations.items():
        source = read(js_path)
        for needle in needles:
            require(needle in source, f"{js_path} missing {needle}")

    print("frontend static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
