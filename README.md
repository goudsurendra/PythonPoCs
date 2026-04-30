# PythonPoCs

A collection of Python proof-of-concept scripts for data analysis and financial crime compliance.

---

## Scripts

### 1. UC Claims Analysis & Prediction (`UC_Claims_Analysis_Prediction.py`)

Time series analysis and ARIMA forecasting on weekly unemployment claims data.

**Input:** `UC_claims_made.xlsx` (place in the same directory as the script)

**Dependencies:**
```
pip install matplotlib pandas numpy statsmodels openpyxl
```

**Run:**
```bash
python UC_Claims_Analysis_Prediction.py
```

**Output:** Interactive plots for seasonal decomposition, stationarity tests, ACF/PACF, and ARIMA predictions.

---

### 2. Threshold Backtesting & Calibration Engine (`threshold_calibration.py`)

Backtests transaction monitoring rules across a range of thresholds to measure how alert volume responds to threshold changes. Helps compliance teams answer: *"Is our current threshold too tight or too loose?"*

**Inputs:**

| File | Description |
|---|---|
| `transactions.csv` | Account transactions with columns: `account_key`, `transaction_key`, `amount`, `debit_or_credit`, `transaction_type` |
| `thresholds.csv` | Monitoring rules with columns: `rule-name`, `customer segment`, `threshold` |

**Dependencies:**
```
pip install matplotlib pandas numpy openpyxl
```

**Run:**
```bash
python threshold_calibration.py
```

**Output:**
- Console report showing alert count and alert rate at 50%, 75%, 100%, 125%, and 150% of each configured threshold
- `calibration_output/threshold_sensitivity.png` — sensitivity curves per rule
- `calibration_output/alert_distribution.png` — account × rule flag heatmap

---

## Adding a New Rule to `thresholds.csv`

#### Step 1 — Add the rule to `thresholds.csv`

```
rule-name,customer segment,threshold
Monitoring of big payments,customer-segment1,10000
excessive card payments,customer-segment1,30000
your new rule name,customer-segment1,5000
```

#### Step 2 — Implement the rule logic in `threshold_calibration.py`

Open `threshold_calibration.py` and add a function that takes the full transactions DataFrame and a threshold value, and returns the set of `account_key` values that breach it:

```python
def _rule_your_new_rule(df_tx, threshold):
    # Example: flag accounts with more than `threshold` total credit transactions
    credits = df_tx[df_tx['debit_or_credit'].str.lower() == 'credit']
    agg = credits.groupby('account_key')['amount'].sum()
    return set(agg[agg >= threshold].index)
```

#### Step 3 — Register the function in `RULE_REGISTRY`

The key must exactly match the `rule-name` value in `thresholds.csv`:

```python
RULE_REGISTRY = {
    'Monitoring of big payments': _rule_large_single_transaction,
    'excessive card payments':    _rule_excessive_card_payments,
    'your new rule name':         _rule_your_new_rule,   # add this line
}
```

That's it. The calibration loop, console report, and charts all pick up the new rule automatically on the next run.
