"""Evaluate the ML and rule-based approaches on a LIAR dataset split."""

from __future__ import annotations

import argparse

from sklearn.metrics import accuracy_score, classification_report

from misinformation_detection.model import load_split, train_model
from misinformation_detection.rules import analyse_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.tsv")
    parser.add_argument("--evaluation-data", default="data/valid.tsv")
    args = parser.parse_args()

    model = train_model(args.train_data)
    evaluation = load_split(args.evaluation_data)
    expected = evaluation["binary_label"]
    approaches = {
        "ML TF-IDF and Logistic Regression": model.predict(evaluation["statement"]),
        "Rule-based linguistic analysis": [analyse_rules(text).label for text in evaluation["statement"]],
    }

    for name, predictions in approaches.items():
        print(f"\n{name}")
        print(f"Accuracy: {accuracy_score(expected, predictions):.4f}")
        print(classification_report(expected, predictions, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
