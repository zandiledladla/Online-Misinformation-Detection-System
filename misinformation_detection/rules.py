"""Interpretable linguistic-risk rules used for comparison with the ML model."""

from __future__ import annotations

import re
from dataclasses import dataclass


RULES = {
    "conspiracy_language": (
        r"\bconspiracy\b",
        r"\bthey are hiding\b",
        r"\bhidden truth\b",
        r"\bsecretly\b",
        r"\bcover.?up\b",
        r"\bthey don't want you to know\b",
        r"\bhidden agenda\b",
    ),
    "deceptive_claims": (
        r"\bfake news\b",
        r"\bhoax\b",
        r"\bfalse information\b",
        r"\bfabricated\b",
        r"\bcompletely false\b",
        r"\bmade up\b",
        r"\bfalse claims?\b",
    ),
    "sensational_language": (
        r"\bshocking\b",
        r"\bno one is telling\b",
        r"\byou won't believe\b",
        r"\bexposed\b",
        r"\bunbelievable\b",
        r"\boutrageous\b",
    ),
    "absolute_claims": (
        r"\balways\b",
        r"\bnever\b",
        r"\beveryone\b",
        r"\bnobody\b",
        r"\b100%\b",
        r"\bwithout exception\b",
    ),
    "strong_certainty": (
        r"\bdefinitely\b",
        r"\bcertainly\b",
        r"\bthere is no doubt\b",
        r"\bguaranteed\b",
        r"\bundeniably\b",
        r"\bbeyond doubt\b",
    ),
}

WEIGHTS = {
    "conspiracy_language": 2,
    "deceptive_claims": 2,
    "sensational_language": 1,
    "absolute_claims": 1,
    "strong_certainty": 1,
}
MAX_SCORE = sum(WEIGHTS.values())


@dataclass(frozen=True)
class RuleResult:
    label: str
    score: int
    risk_percentage: float
    indicators: tuple[str, ...]


def analyse_rules(text: str) -> RuleResult:
    """Return linguistic indicators and a heuristic risk score for text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    normalized = text.lower()
    indicators = tuple(
        category
        for category, patterns in RULES.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    )
    score = sum(WEIGHTS[category] for category in indicators)
    return RuleResult(
        label="Misleading" if score >= 2 else "Truthful",
        score=score,
        risk_percentage=(score / MAX_SCORE) * 100,
        indicators=indicators,
    )
