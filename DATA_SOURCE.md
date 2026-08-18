# Data Source & Provenance

## Source

DA-07 uses the **Iranian Churn** dataset from the UCI Machine Learning Repository.

- Dataset ID: `563`
- DOI: `10.24432/C5JW3Z`
- License: **CC BY 4.0**
- Subject area: Business
- Instances: **3,150 customers**
- Features: **13**
- Missing values: none reported by UCI

UCI documents that the dataset was randomly collected from an Iranian telecom company's database over a 12-month period. The behavioral attributes are aggregated over the first 9 months, while churn is observed at the end of month 12, leaving a 3-month planning gap.

## Why this source is valuable for DA-07

This temporal setup is unusually useful for churn analytics because the explanatory variables precede the churn label. That makes the dataset suitable for descriptive retention/risk analysis without relying on obvious post-outcome leakage.

## Reproducible acquisition

Raw data is not committed to Git. Run:

```bash
python -m src.download_data
```

The script uses the official `ucimlrepo` client and writes a local CSV into `data/raw/`.

## Claim boundaries

DA-07 is an **analytics** project, not a production churn-prediction system. It analyzes historical churn associations, customer segments, behavioral risk signals, and retention priorities. It does not claim causal drivers, intervention lift, or future predictive accuracy.

## Citation

Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z
