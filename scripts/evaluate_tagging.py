from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tagger.tag_worker import tag_content


def overlaps(actual: list[str], expected: list[str]) -> bool:
    return bool(set(actual) & set(expected))


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    tag_result, _, _ = tag_content(case)
    final = tag_result["final"]
    checks = {
        "emotion": overlaps(final["emotion_tags"], case["expected_emotion_any"]),
        "theme": overlaps(final["theme_tags"], case["expected_theme_any"]),
        "role": overlaps(final["recommendation_roles"], case["expected_role_any"]),
        "confidence": final["tag_confidence"] >= 0.45,
    }
    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "actual": final,
    }


def main() -> None:
    cases = json.loads(Path("data/evaluation/tagging_cases.json").read_text(encoding="utf-8"))
    results = [evaluate_case(case) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    report = {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / max(1, len(results)), 3),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
