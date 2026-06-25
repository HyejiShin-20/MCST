from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text).lower()
    tokens = TOKEN_RE.findall(normalized)
    enriched: list[str] = []
    for token in tokens:
        enriched.append(token)
        if len(token) >= 4:
            enriched.extend(token[i : i + 2] for i in range(len(token) - 1))
            enriched.extend(token[i : i + 3] for i in range(len(token) - 2))
    return enriched


def stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    value = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def weighted_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_counter = Counter(item for item in left if item)
    right_counter = Counter(item for item in right if item)
    if not left_counter or not right_counter:
        return 0.0
    overlap = sum(min(left_counter[key], right_counter[key]) for key in left_counter)
    denominator = math.sqrt(sum(left_counter.values()) * sum(right_counter.values()))
    return float(overlap / denominator) if denominator else 0.0


def top_items(values: Iterable[str], limit: int = 5) -> list[str]:
    counter = Counter(item for item in values if item)
    return [item for item, _ in counter.most_common(limit)]

