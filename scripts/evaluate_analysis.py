from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.analyzer import EmotionAnalyzer


def overlaps(actual: list[str], expected: list[str]) -> bool:
    return bool(set(actual) & set(expected))


def evaluate_case(analyzer: EmotionAnalyzer, case: dict[str, Any]) -> dict[str, Any]:
    result = analyzer.analyze(case["text"])
    checks = {
        "emotion": overlaps(result["emotions"], case["expected_emotions_any"]),
        "topic": overlaps(result["topics"], case["expected_topics_any"]),
        "primary_event": result["situation"]["primary_event"] == case["expected_primary_event"],
        "support": overlaps(result["desired_support"], case["expected_support_any"]),
    }
    if "expected_intent" in case:
        checks["intent"] = result["situation"]["intent"] == case["expected_intent"]
    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "actual": {
            "emotions": result["emotions"],
            "topics": result["topics"],
            "primary_event": result["situation"]["primary_event"],
            "intent": result["situation"]["intent"],
            "desired_support": result["desired_support"],
            "confidence": result["confidence"],
            "uncertainty": result["uncertainty"],
        },
    }


def main() -> None:
    cases = json.loads(Path("data/evaluation/analysis_cases.json").read_text(encoding="utf-8"))
    analyzer = EmotionAnalyzer()
    results = [evaluate_case(analyzer, case) for case in cases]
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
