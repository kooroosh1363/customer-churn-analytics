from pathlib import Path

import pandas as pd

from .analytics import (
    age_analysis,
    clean_churn_data,
    complaint_analysis,
    descriptive_risk_profile,
    overall_kpis,
    status_analysis,
    tenure_analysis,
    usage_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "iranian_churn.csv"
OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = clean_churn_data(pd.read_csv(RAW))

    outputs = {
        "executive_summary.csv": overall_kpis(df),
        "tenure_churn.csv": tenure_analysis(df),
        "complaint_churn.csv": complaint_analysis(df),
        "status_churn.csv": status_analysis(df),
        "age_churn.csv": age_analysis(df),
        "usage_behavior.csv": usage_analysis(df),
        "descriptive_risk_segments.csv": descriptive_risk_profile(df),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False)

    print(outputs["executive_summary.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
