# CPT Code Analysis

Analyze CPT code reimbursement patterns. TF-IDF clustering of procedure descriptions + statistical analysis of pricing across regions. This directly connects to work I did at Lantern analyzing pricing gaps across MSAs.

The core question was: do CPT codes that describe similar procedures cluster together based on their text descriptions, and do reimbursement rates for those clusters vary meaningfully by region? The answer to both is yes — and the regional disparities are larger than I expected.

## Approach

1. **TF-IDF Clustering** — Vectorize CPT descriptions, KMeans cluster, compare to actual clinical categories
2. **Reimbursement Analysis** — Distribution analysis, normality testing, regional comparisons
3. **Anomaly Detection** — Isolation Forest + z-score to flag unusual pricing

## Key Findings

- TF-IDF clusters align reasonably well with clinical categories (silhouette ~0.35)
- ASC codes in the South are ~15% below the national average
- Isolation Forest flagged 12 CPT codes with anomalous pricing across regions
- Reimbursement distributions are right-skewed (fail Shapiro-Wilk) — a few high-cost procedures pull the tail

## Run

```bash
python src/cpt_data.py
python src/tfidf_clustering.py
python src/reimbursement_analysis.py
python src/pricing_anomalies.py
```
