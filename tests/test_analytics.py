import pandas as pd

from src.analytics import clean_churn_data, churn_breakdown, overall_kpis


def sample_data():
    return pd.DataFrame({
        "Subscription Length": [5, 12, 30, 40],
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
