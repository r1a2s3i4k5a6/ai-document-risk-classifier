"""
collect_dataset.py

Collects REAL contract clauses from CUAD (Contract Understanding Atticus
Dataset, CC BY 4.0) and labels each one low / medium / high risk using the
category->risk mapping defined in RISK_CATEGORIES.md.

Two ways to get the data:
  A) From a local CUAD_v1.json (most reliable). Download CUAD_v1.zip from
     https://zenodo.org/records/4595826 and unzip, then run:
         python collect_dataset.py --json path/to/CUAD_v1.json
  B) From Hugging Face (needs internet + `pip install datasets`):
         python collect_dataset.py

Output: data/real_documents.csv
        columns: doc_id, text, clause_category, risk_level, source
"""

import argparse
import json
import os
import re

import pandas as pd

# ---- Category -> risk mapping (documented in RISK_CATEGORIES.md) -----------
RISK_MAP = {
    "high": [
        "Uncapped Liability", "Liquidated Damages", "Non-Compete", "Exclusivity",
        "Change Of Control", "Covenant Not To Sue", "Ip Ownership Assignment",
        "Joint Ip Ownership", "Irrevocable Or Perpetual License",
        "Unlimited/All-You-Can-Eat-License",
    ],
    "medium": [
        "Cap On Liability", "Termination For Convenience", "Anti-Assignment",
        "Renewal Term", "Notice Period To Terminate Renewal", "Most Favored Nation",
        "No-Solicit Of Customers", "No-Solicit Of Employees", "Non-Disparagement",
        "Rofr/Rofo/Rofn", "Revenue/Profit Sharing", "Price Restrictions",
        "Minimum Commitment", "Volume Restriction", "Competitive Restriction Exception",
        "Audit Rights", "Post-Termination Services", "Source Code Escrow",
        "Insurance", "Warranty Duration", "Third Party Beneficiary",
        "License Grant", "Non-Transferable License",
        "Affiliate License-Licensor", "Affiliate License-Licensee",
    ],
    "low": [
        "Document Name", "Parties", "Agreement Date", "Effective Date",
        "Expiration Date", "Governing Law",
    ],
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


CATEGORY_TO_RISK = {norm(c): (c, risk) for risk, cats in RISK_MAP.items() for c in cats}
CATEGORY_RE = re.compile(r'related to "([^"]+)"')

MIN_CHARS, MAX_CHARS = 25, 1000
RISK_ORDER = {"high": 0, "medium": 1, "low": 2}


def rows_from_squad_json(path):
    """CUAD_v1.json is SQuAD-format: data -> paragraphs -> qas -> answers."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    for contract in data:
        for para in contract["paragraphs"]:
            for qa in para["qas"]:
                yield qa["question"], [a["text"] for a in qa["answers"]]


def rows_from_huggingface():
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset("theatticusproject/cuad-qa")
    for split in ds:
        for ex in ds[split]:
            yield ex["question"], ex["answers"]["text"]


def build(rows, per_class_cap, seed=42):
    records, skipped = [], set()
    for question, answers in rows:
        m = CATEGORY_RE.search(question)
        if not m:
            continue
        key = norm(m.group(1))
        if key not in CATEGORY_TO_RISK:
            skipped.add(m.group(1))
            continue
        category, risk = CATEGORY_TO_RISK[key]
        for text in answers:
            text = " ".join(text.split())  # collapse whitespace/newlines
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                records.append((text, category, risk))

    df = pd.DataFrame(records, columns=["text", "clause_category", "risk_level"])
    # same text under several categories -> keep the highest risk (conservative)
    df["_o"] = df["risk_level"].map(RISK_ORDER)
    df = df.sort_values("_o").drop_duplicates("text").drop(columns="_o")

    if skipped:
        print("Categories not in mapping (skipped):", sorted(skipped))

    # cap each class so no single risk level dominates
    df = pd.concat(
        [g.sample(min(len(g), per_class_cap), random_state=seed)
         for _, g in df.groupby("risk_level")]
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "doc_id", [f"R{1000 + i}" for i in range(len(df))])
    df["source"] = "CUAD v1 (CC BY 4.0)"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="path to local CUAD_v1.json")
    ap.add_argument("--cap", type=int, default=800, help="max clauses per risk level")
    args = ap.parse_args()

    rows = rows_from_squad_json(args.json) if args.json else rows_from_huggingface()
    df = build(rows, args.cap)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/real_documents.csv", index=False)
    print(f"Wrote {len(df)} clauses to data/real_documents.csv")
    print(df["risk_level"].value_counts())
    print(df["clause_category"].value_counts().head(10))


if __name__ == "__main__":
    main()
