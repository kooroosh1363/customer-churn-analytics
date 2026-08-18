# Data Dictionary

DA-07 uses the UCI **Iranian Churn** dataset. Each row represents one telecom customer.

## Core variables

| Variable | Analytical meaning |
|---|---|
| `call_failure` | Number of call failures |
| `complains` | Whether the customer has complained |
| `subscription_length` | Customer subscription/tenure length |
| `charge_amount` | Ordinal charge amount category |
| `seconds_of_use` | Total usage duration |
| `frequency_of_use` | Usage frequency |
| `frequency_of_sms` | SMS frequency |
| `distinct_called_numbers` | Number of distinct called numbers |
| `age_group` | Source age-group category |
| `tariff_plan` | Tariff-plan category |
| `status` | Source customer status category |
| `age` | Customer age |
| `customer_value` | Source customer-value measure |
| `churn` | Source churn label; 1 indicates churn |
| `churned` | Boolean analytical version of `churn` |

## Time semantics

According to UCI, explanatory behavior is aggregated over the first 9 months of a 12-month observation window and churn is measured at month 12. This separation matters: it gives the business a conceptual 3-month planning window between the behavioral measurement period and the churn outcome.

## Analytical caution

A higher historical churn rate among customers who complain, use the service less, or belong to another segment is an **association**. DA-07 does not label these variables as causal churn drivers without experimental or quasi-experimental evidence.

## Risk-profile semantics

`descriptive_risk_segments.csv` groups customers using transparent median splits for usage and customer value plus complaint status. The resulting churn rate is a historical segment statistic — **not an individual churn probability or predictive model score**.
