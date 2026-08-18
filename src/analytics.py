from __future__ import annotations

import numpy as np
import pandas as pd


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI column names/types while preserving the customer population."""
    out = df.copy()
    out.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns
    ]
    out["churned"] = pd.to_numeric(out["churn"], errors="coerce").eq(1)
    return out


def overall_kpis(df: pd.DataFrame) -> pd.DataFrame:
    customers = len(df)
    churned = int(df["churned"].sum())
    return pd.DataFrame([{
        "customers": customers,
        "churned_customers": churned,
        "retained_customers": customers - churned,
        "churn_rate_pct": round(100 * churned / customers, 2) if customers else 0.0,
        "retention_rate_pct": round(100 * (customers - churned) / customers, 2) if customers else 0.0,
    }])


def churn_breakdown(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    g = df.groupby(dimension, dropna=False).agg(
        customers=("churned", "size"),
        churned_customers=("churned", "sum"),
    ).reset_index()
    g["churn_rate_pct"] = (100 * g["churned_customers"] / g["customers"]).round(2)
    overall = df["churned"].mean()
    g["churn_rate_index"] = (g["churned_customers"] / g["customers"] / overall).round(2) if overall else 0.0
    return g.sort_values(["churn_rate_pct", "customers"], ascending=[False, False])


def tenure_analysis(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["tenure_band"] = pd.cut(
        pd.to_numeric(x["subscription_length"], errors="coerce"),
        bins=[-np.inf, 6, 12, 24, 36, 48, np.inf],
        labels=["<=6m", "7-12m", "13-24m", "25-36m", "37-48m", "49m+"],
    )
    return churn_breakdown(x, "tenure_band")


def usage_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare churned vs retained customers on behavior, not causal effects."""
    candidates = [
        "seconds_of_use", "frequency_of_use", "frequency_of_sms",
        "distinct_called_numbers", "customer_value"
    ]
    available = [c for c in candidates if c in df.columns]
    rows = []
    for col in available:
        for status, group in df.groupby("churned"):
            s = pd.to_numeric(group[col], errors="coerce")
            rows.append({
                "metric": col,
                "customer_status": "churned" if status else "retained",
                "customers": int(s.notna().sum()),
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
            })
    return pd.DataFrame(rows)


def complaint_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return churn_breakdown(df, "complains")


def status_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return churn_breakdown(df, "status")


def age_analysis(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["age_band"] = pd.cut(
        pd.to_numeric(x["age"], errors="coerce"),
        bins=[-np.inf, 24, 34, 44, 54, 64, np.inf],
        labels=["<=24", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    return churn_breakdown(x, "age_band")


def descriptive_risk_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Create transparent historical risk segments; this is not a prediction score."""
    x = df.copy()
    usage = pd.to_numeric(x.get("frequency_of_use"), errors="coerce")
    value = pd.to_numeric(x.get("customer_value"), errors="coerce")
    usage_cut = usage.median()
    value_cut = value.median()
    x["usage_level"] = np.where(usage < usage_cut, "lower_usage", "higher_usage")
    x["value_level"] = np.where(value < value_cut, "lower_value", "higher_value")
    x["complaint_flag"] = np.where(pd.to_numeric(x["complains"], errors="coerce").eq(1), "complaint", "no_complaint")
    g = x.groupby(["usage_level", "value_level", "complaint_flag"], dropna=False).agg(
        customers=("churned", "size"),
        churned_customers=("churned", "sum"),
    ).reset_index()
    g["churn_rate_pct"] = (100 * g["churned_customers"] / g["customers"]).round(2)
    return g.sort_values(["churn_rate_pct", "customers"], ascending=[False, False])
