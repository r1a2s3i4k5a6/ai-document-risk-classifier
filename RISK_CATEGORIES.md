# Risk Categories & Labeling Policy

## Source documents
Clause text is collected from **CUAD v1** (Contract Understanding Atticus
Dataset, The Atticus Project, CC BY 4.0): 510 real commercial contracts in
which lawyers marked clauses belonging to 41 categories.
Each clause is labeled by **its CUAD clause category**, which is mapped to a
risk level below.

## Risk levels

| Level | Definition | Typical reviewer action |
|---|---|---|
| **High** | Clause can create large, open-ended or hard-to-reverse financial, legal or operational exposure. | Mandatory legal review before signing |
| **Medium** | Clause restricts flexibility or shifts obligations, with bounded impact. Worth checking. | Review / confirm terms |
| **Low** | Descriptive or administrative information. Little or no exposure. | Auto-clear |

## Category -> risk mapping

**High**: Uncapped Liability, Liquidated Damages, Non-Compete, Exclusivity,
Change Of Control, Covenant Not To Sue, IP Ownership Assignment, Joint IP
Ownership, Irrevocable Or Perpetual License, Unlimited/All-You-Can-Eat-License

**Medium**: Cap On Liability, Termination For Convenience, Anti-Assignment,
Renewal Term, Notice Period To Terminate Renewal, Most Favored Nation,
No-Solicit Of Customers, No-Solicit Of Employees, Non-Disparagement,
ROFR/ROFO/ROFN, Revenue/Profit Sharing, Price Restrictions, Minimum Commitment,
Volume Restriction, Competitive Restriction Exception, Audit Rights,
Post-Termination Services, Source Code Escrow, Insurance, Warranty Duration,
Third Party Beneficiary, License Grant, Non-Transferable License,
Affiliate License-Licensor, Affiliate License-Licensee

**Low**: Document Name, Parties, Agreement Date, Effective Date,
Expiration Date, Governing Law

## Rationale for key choices
- **Uncapped Liability / Liquidated Damages / Covenant Not To Sue**: exposure
  is unlimited, pre-agreed, or waives legal remedies -> High.
- **Non-Compete / Exclusivity / Change Of Control**: restrict the business's
  future options -> High.
- **Cap On Liability**: it *limits* exposure, so it is Medium (needs checking
  that the cap is adequate) rather than High.
- **Governing Law**: standard boilerplate -> Low.

## Policy-specific classes (alternative)
The dataset also keeps the original `clause_category` column, so the model can
be trained to predict the clause type directly (for example, to route
"Termination" clauses to a different team) instead of, or in addition to, the
low/medium/high level.

## Limitations (state these in your report)
- The mapping is a **proposed policy** based on common contract-review
  practice, not legal advice. Review it with your mentor and adjust it.
- Risk depends on clause *wording and context* (a cap of $1 vs. $10M are
  both "Cap On Liability"). Category-based labels are a simplification.
- CUAD contains English commercial contracts only.
