import pandas as pd
import pytest

from src.analytics import clean_churn_data, churn_breakdown, overall_kpis, tenure_analysis


def sample_data():
    # Double spaces intentionally mirror the inconsistent spacing seen in the UCI headers.
    return pd.DataFrame({
        "Subscription  Length": [5, 12, 30, 40],
        "Complains": [1, 0, 1, 0],
        "Status": [1, 1, 2, 2],
        "Age": [22, 35, 50, 67],
        "Frequency of use": [2, 10, 4, 20],
        "Customer Value": [10, 50, 20, 80],
        "Churn": [1, 0, 1, 0],
    })


def test_population_and_churn_flag_are_preserved():
    x = clean_churn_data(sample_data())
    assert len(x) == 4
    assert int(x["churned"].sum()) == 2


def test_realistic_whitespace_normalizes_to_stable_schema():
    x = clean_churn_data(sample_data())
    assert "subscription_length" in x.columns
    tenure = tenure_analysis(x)
    assert int(tenure["customers"].sum()) == len(x)


def test_overall_rates_reconcile():
    x = clean_churn_data(sample_data())
    k = overall_kpis(x).iloc[0]
    assert k["customers"] == 4
    assert k["churned_customers"] == 2
    assert k["retained_customers"] == 2
    assert k["churn_rate_pct"] == 50.0
    assert k["retention_rate_pct"] == 50.0


def test_breakdown_reconciles_to_overall():
    x = clean_churn_data(sample_data())
    b = churn_breakdown(x, "complains")
    assert int(b["customers"].sum()) == len(x)
    assert int(b["churned_customers"].sum()) == int(x["churned"].sum())


def test_missing_required_schema_fails_loudly():
    broken = sample_data().drop(columns=["Customer Value"])
    with pytest.raises(ValueError, match="customer_value"):
        clean_churn_data(broken)
