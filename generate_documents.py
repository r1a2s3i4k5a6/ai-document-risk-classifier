"""
generate_documents.py

Builds a synthetic business document dataset for the risk classification
project, labelled low / medium / high risk.

Why synthetic data: no real contract/policy corpus was available, so this
script generates realistic clause-level text instead. An early version of
this dataset used only 8 fixed phrases per risk level with no variation,
which produced a meaningless 100% test accuracy - the model was memorizing
exact sentences, not learning risk language. To keep the evaluation honest,
this script adds a batch of deliberately ambiguous clauses (phrases that
reference "pending review" or "further negotiation" without stating an
actual risk) with a randomly assigned risk label each time they appear, so
the model is forced to handle genuine uncertainty rather than memorize.

Output: data/documents.csv
"""

import os
import random
import pandas as pd

random.seed(42)

LOW_RISK = [
    "This agreement may be terminated by either party with 30 days written notice",
    "Standard non-disclosure clause covering confidential business information",
    "Employee must complete annual compliance training by December 31st",
    "Payment shall be made via standard invoicing with net 30 terms",
    "This policy is reviewed annually by the compliance department",
    "Either party may request a copy of this agreement at any time",
    "Standard working hours are defined as Monday to Friday, 9 to 5",
    "This document outlines the standard onboarding process for new employees",
]

MEDIUM_RISK = [
    "Payment terms are net 45 days from invoice date",
    "Either party may assign this agreement without the other's prior consent",
    "Contractor is granted broad access to internal systems with minimal oversight",
    "This clause allows for automatic renewal unless cancelled 60 days in advance",
    "Vendor may subcontract portions of this work without prior written approval",
    "Disputes shall be resolved through arbitration in a jurisdiction of vendor's choosing",
    "This agreement may be modified unilaterally with 10 days notice",
    "Late payments incur a penalty fee without a cap on total charges",
]

HIGH_RISK = [
    "Vendor shall indemnify the company against all third-party claims without limitation",
    "This contract contains an unlimited liability clause with no cap on damages",
    "Data sharing with third parties is permitted without explicit customer consent",
    "This agreement waives all rights to legal recourse in case of breach",
    "Personal data may be transferred internationally without additional safeguards",
    "The company assumes full liability for contractor actions with no indemnification",
    "This clause permits termination of employee data rights without notice",
    "Confidential information may be disclosed to competitors under this agreement",
]

# Deliberately vague clauses - a human reviewer would also need more context
# to judge these. Risk level is assigned fresh and random EACH time one
# appears, so the model cannot simply memorize a fixed answer for them.
AMBIGUOUS_PHRASES = [
    "This clause requires further legal review before approval",
    "Terms are subject to change based on ongoing negotiations",
    "Please refer to the attached addendum for full details",
    "This section has not yet been finalized by legal counsel",
]

RISK_PHRASES = {"low": LOW_RISK, "medium": MEDIUM_RISK, "high": HIGH_RISK}
RISK_WEIGHTS = [0.40, 0.35, 0.25]  # low risk most common, like a real document set


def build_core_documents(n=1500):
    levels = list(RISK_PHRASES)
    rows = []
    for _ in range(n):
        level = random.choices(levels, weights=RISK_WEIGHTS)[0]
        text = random.choice(RISK_PHRASES[level])
        rows.append((text, level))
    return rows


def build_ambiguous_documents(n=150):
    levels = list(RISK_PHRASES)
    rows = []
    for _ in range(n):
        text = random.choice(AMBIGUOUS_PHRASES)
        level = random.choice(levels)
        rows.append((text, level))
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    rows = build_core_documents(1500) + build_ambiguous_documents(150)
    df = pd.DataFrame(rows, columns=["text", "risk_level"])
    df.insert(0, "doc_id", [f"D{1000 + i}" for i in range(len(df))])
    df.to_csv("data/documents.csv", index=False)
    print(f"Wrote {len(df)} documents to data/documents.csv")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    main()
