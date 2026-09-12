"""Launch the Gradio comparison interface."""

from __future__ import annotations

import argparse

import gradio as gr

from misinformation_detection.model import analyse_text, train_model


def format_analysis(text, model):
    try:
        result = analyse_text(text, model)
    except ValueError as error:
        return str(error)

    indicators = (
        ", ".join(item.replace("_", " ").title() for item in result.rule_result.indicators)
        or "None detected"
    )
    return (
        f"Final Classification: {result.final_label}\n\n"
        f"ML Prediction: {result.ml_label}\n"
        f"ML Confidence: {result.ml_confidence:.2f}%\n"
        f"ML Misleading Probability: {result.misleading_probability:.2f}%\n\n"
        f"Rule-Based Result: {result.rule_result.label}\n"
        f"Rule-Based Risk: {result.rule_result.risk_percentage:.2f}%\n"
        f"Detected Indicators: {indicators}\n\n"
        "This prototype identifies patterns; it does not verify whether a claim is factually true."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/train.tsv")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    model = train_model(args.train_data)

    interface = gr.Interface(
        fn=lambda text: format_analysis(text, model),
        inputs=gr.Textbox(lines=6, label="Statement", placeholder="Enter a statement to analyse..."),
        outputs=gr.Textbox(lines=12, label="Analysis Results"),
        title="Misinformation Detection System",
        description="Compare TF-IDF Logistic Regression with interpretable linguistic-risk rules.",
        examples=[
            ["Cybersecurity experts advise users to enable two-factor authentication."],
            ["Climate change is a fake conspiracy created to manipulate the global economy."],
        ],
    )
    interface.launch(share=args.share)


if __name__ == "__main__":
    main()
