"""
train_risk_classifier.py

Trains a document risk classifier (low / medium / high), evaluates it
honestly on a held-out test set, extracts the top words driving each risk
category for explainability (brief step 5), and applies a tiered review
policy instead of a flat confidence threshold (brief step 6).

Why a tiered policy, not a flat threshold: a support ticket being
mis-routed costs a delay; a high-risk legal clause slipping through
uncaught can cost real money or legal exposure. Treating every prediction
below a confidence cutoff the same way, regardless of what's at stake,
ignores that the cost of being wrong differs by category. Instead:
  - High risk + low confidence  -> Escalate immediately (most urgent)
  - High risk + high confidence, OR any risk + low confidence -> Flag for review
  - Low/medium risk + high confidence -> Auto-clear
This reflects that a confident "high risk" call still deserves a second
look given the stakes, even though the model isn't unsure about it.

Run:  python train_risk_classifier.py
Out:  models/risk_classifier.joblib
      reports/evaluation_results.txt
      reports/explainability.txt
      reports/flagged_documents.csv
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CONFIDENCE_THRESHOLD = 0.60


def review_action(risk, confidence):
    if confidence < CONFIDENCE_THRESHOLD and risk == "high":
        return "Escalate immediately"
    elif confidence < CONFIDENCE_THRESHOLD or risk == "high":
        return "Flag for review"
    else:
        return "Auto-clear"


def top_words_per_class(vectorizer, model, n=8):
    feature_names = vectorizer.get_feature_names_out()
    lines = []
    for idx, label in enumerate(model.classes_):
        coeffs = model.coef_[idx]
        top_indices = np.argsort(coeffs)[-n:]
        lines.append(f"\nTop words signalling '{label.upper()}' risk:")
        for i in reversed(top_indices):
            lines.append(f"  {feature_names[i]}  (weight: {coeffs[i]:.2f})")
    return lines


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("data/real_documents.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_df["risk_level"])

    predictions = model.predict(X_test)
    acc = accuracy_score(test_df["risk_level"], predictions)

    report_lines = [f"Dataset size: {len(df)} documents ({len(train_df)} train / {len(test_df)} test)"]
    report_lines.append(f"\nAccuracy: {acc:.3f}")
    report_lines.append(classification_report(test_df["risk_level"], predictions, digits=3))

    labels_sorted = sorted(test_df["risk_level"].unique())
    cm = confusion_matrix(test_df["risk_level"], predictions, labels=labels_sorted)
    report_lines.append("Confusion matrix (rows = true, cols = predicted):")
    report_lines.append(str(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)))

    print("\n".join(report_lines))
    with open("reports/evaluation_results.txt", "w") as f:
        f.write("\n".join(report_lines))

    # Explainability (brief step 5)
    explain_lines = top_words_per_class(vectorizer, model)
    print("\n".join(explain_lines))
    with open("reports/explainability.txt", "w") as f:
        f.write("\n".join(explain_lines))

    # Tiered review policy (brief step 6)
    confidences = np.max(model.predict_proba(X_test), axis=1)
    review_df = test_df.copy()
    review_df["predicted_risk"] = predictions
    review_df["confidence"] = confidences.round(3)
    review_df["action"] = review_df.apply(
        lambda row: review_action(row["predicted_risk"], row["confidence"]), axis=1
    )

    action_counts = review_df["action"].value_counts()
    print("\nReview actions:")
    print(action_counts)

    flagged = review_df[review_df["action"] != "Auto-clear"]
    flagged[["doc_id", "text", "risk_level", "predicted_risk", "confidence", "action"]] \
        .to_csv("reports/flagged_documents.csv", index=False)

    joblib.dump({"vectorizer": vectorizer, "model": model},
                "models/risk_classifier.joblib")

    print(f"\nSaved model to models/risk_classifier.joblib")
    print(f"{len(flagged)} of {len(test_df)} documents flagged or escalated for human review")


if __name__ == "__main__":
    main()
