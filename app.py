"""
app.py

Streamlit dashboard exposing the document risk classifier (brief step 6).
Unlike a flat "needs review" flag, this shows the tiered action the system
recommends (Auto-clear / Flag for review / Escalate immediately) alongside
the specific words that drove the risk prediction (brief step 5's
explainability), so a reviewer sees not just a verdict but why.

Run:  streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np

CONFIDENCE_THRESHOLD = 0.60

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


def review_action(risk, confidence):
    if confidence < CONFIDENCE_THRESHOLD and risk == "high":
        return "Escalate immediately", "🚨"
    elif confidence < CONFIDENCE_THRESHOLD or risk == "high":
        return "Flag for review", "⚠️"
    else:
        return "Auto-clear", "✅"


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
st.caption("Paste a contract clause or policy excerpt to see its risk level and why.")

bundle = load_model()
vectorizer, model = bundle["vectorizer"], bundle["model"]

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
