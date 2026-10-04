"""
app.py

Streamlit dashboard exposing the document risk classifier (brief step 6).
Unlike a flat "needs review" flag, this shows the tiered action the system
recommends (Auto-clear / Flag for review / Escalate immediately) alongside
the specific words that drove the risk prediction (brief step 5's
explainability), so a reviewer sees not just a verdict but why.

Three tabs:
  1. Check a clause     - paste one clause, see risk, confidence, action, key words
  2. Upload a document  - PDF / Word / text file is cleaned, split into clauses,
                          and every clause is scored (brief steps 2 and 6)
  3. Flagged documents  - clauses the model sent to human review on the test set,
                          plus class-wise performance (brief step 6)

Run:  streamlit run app.py
"""

import os

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from document_utils import (
    CONFIDENCE_THRESHOLD,
    extract_text,
    review_action as tiered_action,
    score_clauses,
    split_into_clauses,
)

st.set_page_config(page_title="Document Risk Classifier", page_icon="📄", layout="centered")

st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #1B2A4A;
        border-color: #1B2A4A;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #16213A;
        border-color: #16213A;
    }
    div[data-testid="stMetricValue"] {
            color: #4F8DF7;

    }
    .word-tag {
        display: inline-block;
        background-color: #FDF0D5;
        color: #8A5A00;
        border: 1px solid #E8B84B;
        border-radius: 14px;
        padding: 3px 12px;
        margin: 3px 4px 3px 0;
        font-size: 0.85rem;
        font-family: monospace;
    }
    h1 {
        color: #1B2A4A;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/risk_classifier.joblib")


ICONS = {"Escalate immediately": "🚨", "Flag for review": "⚠️", "Auto-clear": "✅"}


def review_action(risk, confidence):
    action = tiered_action(risk, confidence)
    return action, ICONS[action]


@st.cache_data
def class_performance(_vectorizer, _model):
    """Class-wise precision / recall / F1 on the same held-out test split used in training."""
    df = pd.read_csv("data/real_documents.csv")
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    preds = _model.predict(_vectorizer.transform(test_df["text"]))
    report = classification_report(test_df["risk_level"], preds, output_dict=True, digits=3)
    rows = {k: v for k, v in report.items() if k in ("high", "medium", "low")}
    table = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]].round(3)
    return table, round(report["accuracy"], 3)


def top_contributing_words(vectorizer, model, text, predicted_class, n=5):
    feature_names = vectorizer.get_feature_names_out()
    class_index = list(model.classes_).index(predicted_class)
    coeffs = model.coef_[class_index]

    doc_vector = vectorizer.transform([text]).toarray()[0]
    present_word_indices = np.nonzero(doc_vector)[0]

    scored = [(feature_names[i], coeffs[i]) for i in present_word_indices if coeffs[i] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


st.title("📄 AI Document Risk & Compliance Classifier")
st.caption("Check a clause, upload a whole document, and review what the model flagged for a human.")

bundle = load_model()
vectorizer, model = bundle["vectorizer"], bundle["model"]

tab_clause, tab_upload, tab_flagged = st.tabs(
    ["🔎 Check a clause", "📂 Upload a document", "🚩 Flagged documents"]
)

# ------------------------------------------------------------ tab 1 --------
with tab_clause:
    st.caption("Paste a contract clause or policy excerpt to see its risk level and why.")
    if "form_round" not in st.session_state:
        st.session_state.form_round = 0

    text_key = f"doc_input_{st.session_state.form_round}"

    doc_text = st.text_area(
        "Document text",
        placeholder="e.g. This agreement waives all rights to legal recourse in case of breach",
        height=100,
        key=text_key,
    )

    col_classify, col_clear = st.columns([1, 1])
    classify_clicked = col_classify.button("Assess Document", type="primary")
    clear_clicked = col_clear.button("🔄 Check Another Document")

    if clear_clicked:
        st.session_state.form_round += 1
        st.rerun()

    if classify_clicked and doc_text.strip():
        features = vectorizer.transform([doc_text])
        predicted_risk = model.predict(features)[0]
        confidence = float(model.predict_proba(features).max())
        action, icon = review_action(predicted_risk, confidence)

        col1, col2 = st.columns(2)
        col1.metric("Predicted risk level", predicted_risk.title())
        col2.metric("Confidence", f"{confidence:.0%}")

        st.progress(confidence)

        if action == "Escalate immediately":
            st.error(f"{icon} **{action}** — high risk predicted and the model is not confident. This needs immediate human attention.")
        elif action == "Flag for review":
            st.warning(f"{icon} **{action}** — either the risk is high, or the model isn't fully confident. A human should confirm.")
        else:
            st.success(f"{icon} **{action}** — low/medium risk with high confidence. No immediate action needed.")

        contributing = top_contributing_words(vectorizer, model, doc_text, predicted_risk)
        if contributing:
            st.markdown(f"**Words that drove the '{predicted_risk}' prediction:**")
            tags_html = "".join(f'<span class="word-tag">{w}</span>' for w, _ in contributing)
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.caption("No single word stood out strongly — this document may be genuinely ambiguous.")


# ------------------------------------------------------------ tab 2 --------
with tab_upload:
    st.caption(
        "Upload a contract (PDF, Word or text). The text is extracted, cleaned, "
        "split into clauses, and every clause is scored. Riskiest clauses come first."
    )
    uploaded = st.file_uploader("Contract file", type=["pdf", "docx", "txt"])

    if uploaded is not None:
        try:
            raw_text = extract_text(uploaded.getvalue(), uploaded.name)
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            raw_text = None

        if raw_text is None:
            pass  # the error above already explains what went wrong
        elif not raw_text.strip():
            st.warning(
                "No text could be extracted. If this is a scanned PDF (an image), "
                "it needs OCR first, which this app does not do."
            )
        else:
            clauses = split_into_clauses(raw_text)
            results = score_clauses(vectorizer, model, clauses)

            n_esc = int(results["action"].str.contains("Escalate").sum())
            n_flag = int(results["action"].str.contains("Flag").sum())
            n_clear = int(results["action"].str.contains("Auto-clear").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clauses found", len(results))
            c2.metric("🚨 Escalate", n_esc)
            c3.metric("⚠️ Flag", n_flag)
            c4.metric("✅ Auto-clear", n_clear)

            only_review = st.checkbox("Show only clauses that need review", value=True)
            view = results[~results["action"].str.contains("Auto-clear")] if only_review else results
            st.dataframe(
                view.rename(columns={
                    "clause": "Clause", "predicted_risk": "Risk",
                    "confidence": "Confidence", "action": "Action", "key_words": "Key words",
                }),
                hide_index=True,
            )
            st.download_button(
                "Download all results (CSV)",
                results.to_csv(index=False).encode("utf-8"),
                file_name="document_risk_results.csv",
                mime="text/csv",
            )

# ------------------------------------------------------------ tab 3 --------
with tab_flagged:
    st.caption(
        "Clauses from the held-out test set that the model sent to a human "
        "(from reports/flagged_documents.csv)."
    )
    if os.path.exists("reports/flagged_documents.csv"):
        flagged = pd.read_csv("reports/flagged_documents.csv")
        counts = flagged["action"].value_counts()

        m1, m2, m3 = st.columns(3)
        m1.metric("Flagged / escalated", len(flagged))
        m2.metric("🚨 Escalate immediately", int(counts.get("Escalate immediately", 0)))
        m3.metric("⚠️ Flag for review", int(counts.get("Flag for review", 0)))

        choice = st.selectbox("Filter by action", ["All"] + list(counts.index))
        shown = flagged if choice == "All" else flagged[flagged["action"] == choice]
        st.dataframe(
            shown.sort_values("confidence").rename(columns={
                "doc_id": "ID", "text": "Clause", "risk_level": "True risk",
                "predicted_risk": "Predicted risk", "confidence": "Confidence",
                "action": "Action",
            }),
            hide_index=True,
        )
    else:
        st.info("Run train_risk_classifier.py to create reports/flagged_documents.csv.")

    st.subheader("Class-wise performance")
    if os.path.exists("data/real_documents.csv"):
        perf, overall = class_performance(vectorizer, model)
        st.dataframe(perf)
        st.caption(f"Overall accuracy on the held-out test set: {overall:.1%}")

st.divider()
st.caption(
    "Tiered review policy: a confident high-risk prediction is still flagged "
    "(not auto-cleared), since the cost of a missed high-risk document is "
    "higher than a missed low-risk one. Only an unsure, high-risk prediction "
    "is escalated immediately."
)

with st.expander("About this project"):
    st.write(
        "Trained on 2,400 real contract clauses from the CUAD dataset "
        "(low/medium/high risk). Accuracy: 82.1% on a held-out test set. "
        "Risk levels come from a clause-category mapping described in "
        "RISK_CATEGORIES.md, so this is decision support, not legal advice."
    )
