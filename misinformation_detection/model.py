"""Dataset loading, model training and comparative prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .rules import RuleResult, analyse_rules


LIAR_COLUMNS = (
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state", "party", "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
)
LABEL_MAPPING = {
    "true": "Truthful",
    "mostly-true": "Truthful",
    "half-true": "Misleading",
    "barely-true": "Misleading",
    "false": "Misleading",
    "pants-fire": "Misleading",
}


@dataclass(frozen=True)
class AnalysisResult:
    final_label: str
    ml_label: str
    ml_confidence: float
    misleading_probability: float
    rule_result: RuleResult


def load_split(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    frame = frame.dropna(subset=["statement", "label"]).copy()
    frame["binary_label"] = frame["label"].map(LABEL_MAPPING)
    return frame.dropna(subset=["binary_label"])


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30_000)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42)),
        ]
    )


def train_model(train_path: str | Path) -> Pipeline:
    training = load_split(train_path)
    model = build_model()
    model.fit(training["statement"], training["binary_label"])
    return model


def analyse_text(text: str, model: Pipeline) -> AnalysisResult:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    ml_label = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    misleading_probability = float(probabilities[classes.index("Misleading")]) * 100
    confidence = float(max(probabilities)) * 100

    # The model supplies the final label. Rules are an interpretable comparison.
    return AnalysisResult(
        final_label=ml_label,
        ml_label=ml_label,
        ml_confidence=confidence,
        misleading_probability=misleading_probability,
        rule_result=analyse_rules(text),
    )
